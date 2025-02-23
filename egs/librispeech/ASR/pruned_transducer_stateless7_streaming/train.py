#!/usr/bin/env python3
# Copyright    2021-2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,)
#                                                       Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless7_streaming/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless7_streaming/exp \
  --full-libri 1 \
  --max-duration 300

# For mix precision training:

./pruned_transducer_stateless7_streaming/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming/exp \
  --full-libri 1 \
  --max-duration 550
"""


import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from asr_datamodule import LibriSpeechAsrDataModule
from decoder import Decoder
from joiner import Joiner
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model import Transducer
from optim import Eden, ScaledAdam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer
from beam_search import greedy_search_batch

from icefall import diagnostics
from icefall.checkpoint import load_checkpoint, remove_checkpoints
from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
from icefall.checkpoint import (
    save_checkpoint_with_global_batch_idx,
    update_averaged_model,
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.err import raise_grad_scale_is_too_small_error
from icefall.hooks import register_inf_check_hooks
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool, add_sos
from torch import amp
import editdistance
import random
import re
import itertools

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]

try:
    import k2
    logging.info("Successfully imported k2")
except Exception as e:
    logging.warning(f"Could not import k2: {e}")


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,4,3,2,4",
        help="Number of zipformer encoder layers, comma separated.",
    )

    parser.add_argument(
        "--feedforward-dims",
        type=str,
        default="1024,1024,2048,2048,1024",
        help="Feedforward dimension of the zipformer encoder layers, comma separated.",
    )

    parser.add_argument(
        "--nhead",
        type=str,
        default="8,8,8,8,8",
        help="Number of attention heads in the zipformer encoder layers.",
    )

    parser.add_argument(
        "--encoder-dims",
        type=str,
        default="384,384,384,384,384",
        help="Embedding dimension in the 2 blocks of zipformer encoder layers, comma separated",
    )

    parser.add_argument(
        "--attention-dims",
        type=str,
        default="192,192,192,192,192",
        help="""Attention dimension in the 2 blocks of zipformer encoder layers, comma separated;
        not the same as embedding dimension.""",
    )

    parser.add_argument(
        "--encoder-unmasked-dims",
        type=str,
        default="256,256,256,256,256",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "Must be <= each of encoder_dims.  Empirically, less than 256 seems to make performance "
        " worse.",
    )

    parser.add_argument(
        "--zipformer-downsampling-factors",
        type=str,
        default="1,2,4,8,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--cnn-module-kernels",
        type=str,
        default="31,31,31,31,31",
        help="Sizes of kernels in convolution modules",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--short-chunk-size",
        type=int,
        default=50,
        help="""Chunk length of dynamic training, the chunk size would be either
        max sequence length of current batch or uniformly sampled from (1, short_chunk_size).
        """,
    )

    parser.add_argument(
        "--num-left-chunks",
        type=int,
        default=4,
        help="How many left context can be seen in chunks when calculating attention.",
    )

    parser.add_argument(
        "--decode-chunk-len",
        type=int,
        default=32,
        help="The chunk size for decoding (in frames before subsampling)",
    )

    parser.add_argument('--use-xlsr', action='store_true', default=False,
                        help='Use XLSR encoder instead of the default Zipformer encoder')
    parser.add_argument('--xlsr-model-name', type=str, default='facebook/wav2vec2-xls-r-300m',
                        help='Pretrained XLSR model name to use with the XLSR encoder')

    parser.add_argument("--use-accelerate", type=str2bool, default=False, help="Use HF accelerate for multi GPU training")

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision (FP16) training.",
    )

    parser.add_argument(
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use bfloat16 precision training. If both --use-fp16 and --use-bf16 are True, bf16 takes precedence.",
    )

    parser.add_argument(
        "--streaming-regularization",
        type=float,
        default=0.1,
        help="Weight for streaming regularization loss",
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7_streaming/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_2500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.05, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network) part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=2000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 1.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech",
        choices=["librispeech", "estonian"],
        help="Dataset to use for training (librispeech or estonian)",
    )

    parser.add_argument(
        "--train-txt",
        type=str,
        default="Data/train.txt",
        help="Path to training text file for Estonian dataset",
    )

    parser.add_argument(
        "--val-txt",
        type=str,
        default="Data/val.txt",
        help="Path to validation text file for Estonian dataset",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training (used only with Estonian dataset)",
    )

    parser.add_argument(
        "--audio-base-path",
        type=str,
        default=None,
        help="Base path for audio files in Estonian dataset",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            # XLSR parameters from paper
            "use_xlsr": True,  # Enable XLSR by default
            "xlsr_model_name": "facebook/wav2vec2-xls-r-300m",
            "frame_duration": 0.025,  # 25ms per frame
            "frame_stride": 0.020,  # 20ms stride
            "subsampling_factor": 320,  # For wav2vec2/XLSR models
            
            # Chunk configurations from paper
            "decode_chunk_size": 5120,  # 320ms at 16kHz (best performing)
            "chunk_overlap": 2560,  # Half of decode_chunk_size
            "use_attention_sink": True,
            "attention_sink_size": 16,  # Paper's optimal setting
            "left_context_chunks": 1,  # Paper's optimal setting
            
            # Decoding parameters
            "max_states": 64,  # Maximum active FSA states
            "min_states": 16,  # Minimum active FSA states (max_states // 4)
            "beam": 20.0,  # Beam size for pruning
            "max_contexts": 8,  # Maximum right contexts
            "temperature": 1.0,  # Temperature for softmax scaling
            
            # Training parameters
            "streaming_regularization": 0.1,  # Weight for streaming regularization loss
            "batch_size": 4,  # Adjust based on your GPU memory
            "num_epochs": 30,
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "warm_step": 2000,
            "batch_idx_train": 0,  # Initialize batch index counter
            "reset_interval": 200,  # Reset interval for loss statistics
            "log_interval": 50,  # Log interval for training progress
            "valid_interval": 2000,  # Validation interval
            
            # Progressive unfreezing schedule
            "unfreeze_schedule": [
                (10, 8),   # At epoch 10, unfreeze 8 more layers
                (20, 12)   # At epoch 20, unfreeze all layers
            ],
            
            # Dataset parameters
            "dataset": "estonian",  # "librispeech" or "estonian"
            "train_txt": None,  # Path to train.txt for Estonian dataset
            "val_txt": None,  # Path to val.txt for Estonian dataset
            
            # Other parameters
            "feature_dim": 80,
            "vocab_size": None,  # Will be set after loading tokenizer
            "blank_id": None,  # Will be set after loading tokenizer
            "context_size": 2,
            "max_duration": 200.0,
            "random_seed": 42,
            "save_every_n": 2000,
            "keep_last_k": 20,
            "average_period": 100,
            "use_fp16": False,
            "epoch": 1,
            "return_encoder_output": False,
            "return_boundaries": False,
            "use_averaged_model": False,
            
            # Best metrics tracking
            "best_valid_wer": float("inf"),
            "best_valid_epoch": 0,
            "best_train_loss": float("inf"),
            "best_train_epoch": 0,
            
            # Experiment directory
            "exp_dir": Path("pruned_transducer_stateless7_streaming/exp"),
            "lang_dir": Path("data/lang_bpe_2500"),
            "vocab_file": "data/lang_bpe_2500/tokens.txt",
        }
    )
    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    if getattr(params, 'use_xlsr', False):
        from xlsr_encoder import XLSREncoder
        encoder = XLSREncoder(model_name=params.xlsr_model_name)
        return encoder

    # TODO: We can add an option to switch between Zipformer and Transformer
    def to_int_tuple(s: str):
        return tuple(map(int, s.split(",")))

    encoder = Zipformer(
        num_features=params.feature_dim,
        output_downsampling_factor=2,
        zipformer_downsampling_factors=to_int_tuple(
            params.zipformer_downsampling_factors
        ),
        encoder_dims=to_int_tuple(params.encoder_dims),
        attention_dim=to_int_tuple(params.attention_dims),
        encoder_unmasked_dims=to_int_tuple(params.encoder_unmasked_dims),
        nhead=to_int_tuple(params.nhead),
        feedforward_dim=to_int_tuple(params.feedforward_dims),
        cnn_module_kernels=to_int_tuple(params.cnn_module_kernels),
        num_encoder_layers=to_int_tuple(params.num_encoder_layers),
        num_left_chunks=params.num_left_chunks,
        short_chunk_size=params.short_chunk_size,
        decode_chunk_size=params.decode_chunk_len // 2,
    )
    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    if params.dataset == "estonian":
        from estonian_decoder import EstonianDecoder
        decoder = EstonianDecoder(
            vocab_size=params.vocab_size,
            decoder_dim=params.decoder_dim,
            blank_id=params.blank_id,
            context_size=2  # From paper
        )
    else:
        decoder = Decoder(
            vocab_size=params.vocab_size,
            decoder_dim=params.decoder_dim,
            blank_id=params.blank_id,
        )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    # Use XLSR encoder output dimension if use_xlsr is enabled
    encoder_dim = 1024 if getattr(params, "use_xlsr", False) else int(params.encoder_dims.split(",")[-1])
    joiner = Joiner(
        encoder_dim=encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)
    
    # Set encoder_dim based on whether XLSR is used
    encoder_dim = 1024 if getattr(params, "use_xlsr", False) else int(params.encoder_dims.split(",")[-1])
    
    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size
    )
    
    if params.use_xlsr:
        # Initialize decoder with Xavier instead of random
        torch.nn.init.xavier_uniform_(model.decoder.embedding.weight)
        # Freeze first N layers of XLSR
        for layer in model.encoder.model.encoder.layers[:10]:
            layer.requires_grad_(False)
            
    # Create decoding graph for Estonian if needed
    if params.dataset == "estonian":
        from estonian_decoder import create_estonian_token_table, create_estonian_decoding_graph
        token_table = create_estonian_token_table(params.vocab_file)
        decoding_graph = create_estonian_decoding_graph(
            token_table=token_table,
            num_tokens=params.vocab_size,
            device=torch.device("cpu")  # Will be moved to correct device later
        )
        # Store FSA objects in decoder
        model.decoder.token_table = token_table
        model.decoder.decoding_graph = decoding_graph
    
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    
    save_data = {
        "model": model.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "sampler": sampler.state_dict() if (sampler is not None and hasattr(sampler, 'state_dict')) else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "params": params,
    }
    if model_avg is not None:
        save_data["model_avg"] = model_avg.state_dict()
        
    torch.save(save_data, filename)

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def get_random_chunk_size() -> int:
    """Returns a random chunk size for training.
    The chunk sizes are sampled from [8, 16, 32] frames.
    """
    chunk_sizes = [8, 16, 32]  # in frames
    return random.choice(chunk_sizes)


def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute transducer loss given the model and its inputs.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y)

    with torch.set_grad_enabled(is_training):
        # Get encoder output
        encoder_out, encoder_out_lens = model.encoder(feature, feature_lens)
        
        # Get decoder output
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]
        
        # Add blank at the beginning
        blank_id = model.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)
        
        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(device)
        
        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = model.decoder(sos_y_padded)
        
        # Create boundary tensor
        batch_size = feature.size(0)
        boundary = torch.zeros((batch_size, 4), dtype=torch.int64, device=device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens
        
        # Forward pass through model
        simple_loss, pruned_loss = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
            boundary=boundary,  # Pass boundary tensor explicitly
        )

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()

    return loss, info


def decode_with_beam_search(
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
    decoding_graph: Optional[k2.Fsa] = None,
) -> List[str]:
    """
    Decode encoder output using beam search with FSA.
    Args:
        encoder_out: Encoder output (batch, time, dim)
        encoder_out_lens: Length of each sequence
        params: Parameters for decoding
        sp: SentencePiece tokenizer
        decoding_graph: Optional FSA decoding graph
    Returns:
        List of decoded strings
    """
    device = encoder_out.device
    batch_size = encoder_out.size(0)

    try:
        # Convert encoder output to log probabilities
        encoder_out = torch.nn.functional.log_softmax(encoder_out, dim=-1)

        # Create supervision segments on CPU as required by k2
        supervision_segments = torch.tensor(
            [[i, 0, encoder_out_lens[i].item()] for i in range(batch_size)],
            dtype=torch.int32,
            device="cpu"  # k2 requires this to be on CPU
        )

        # Move encoder output to CPU for k2
        encoder_out_cpu = encoder_out.detach().cpu()
        
        # Create dense FSA from encoder output
        dense_fsa = k2.DenseFsaVec(encoder_out_cpu, supervision_segments)

        if decoding_graph is None:
            # Create a simple CTC-like decoding graph
            decoding_graph = k2.trivial_graph(
                params.vocab_size - 1,  # -1 since k2 doesn't need the blank token
                device="cpu"  # Create on CPU first
            )

        # Ensure decoding graph is on CPU and sorted
        decoding_graph = decoding_graph.to("cpu")
        decoding_graph = k2.arc_sort(decoding_graph)

        # Intersect with decoding graph and find best path
        lattice = k2.intersect_dense_pruned(
            decoding_graph,
            dense_fsa,
            search_beam=params.beam,
            output_beam=params.beam,
            min_active_states=params.min_states,
            max_active_states=params.max_states
        )

        # Connect and sort the lattice
        lattice = k2.connect(lattice)
        lattice = k2.top_sort(lattice)

        # Get best path
        best_path = k2.shortest_path(lattice, use_double_scores=True)
        
        # Convert best path to token IDs
        hyps = []
        for i in range(batch_size):
            # Get labels (token IDs) from the best path
            labels = []
            # Use k2's proper methods to get labels
            if hasattr(best_path[i], 'labels'):
                # Some k2 versions provide labels directly
                labels = best_path[i].labels.tolist()
            else:
                # Otherwise extract from arcs
                fsa = best_path[i]
                aux_labels = fsa.aux_labels.tolist() if hasattr(fsa, 'aux_labels') else []
                labels = [arc.label for arc in fsa.arcs if arc.label != 0]
                if aux_labels:
                    labels = aux_labels
            
            # Remove consecutive duplicates and zeros
            labels = [x for x, _ in itertools.groupby(labels) if x != 0]
            
            # Convert token IDs to text using sentencepiece
            hyp = sp.decode(labels)
            hyps.append(hyp)

        return hyps

    except Exception as e:
        logging.warning(f"FSA decoding failed: {e}, falling back to greedy decoding")
        # Simple greedy decoding fallback
        hyps = []
        # Get most likely token at each timestep
        predictions = encoder_out.argmax(dim=-1)  # [batch, time]
        
        # Convert predictions to text for each sequence in batch
        for i in range(batch_size):
            # Get sequence up to its length
            sequence = predictions[i, :encoder_out_lens[i]]
            # Remove consecutive duplicates and zeros (CTC blank)
            sequence = [t.item() for t in sequence]
            sequence = [x for x, _ in itertools.groupby(sequence) if x != 0]
            # Decode to text
            hyp = sp.decode(sequence)
            hyps.append(hyp)
            
        return hyps


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()
    tot_loss = MetricsTracker()
    
    # Initialize WER computer
    wer = WERComputer()
    
    # Get device from model parameters
    device = next(model.parameters()).device
    
    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        
        # Compute WER for this batch
        if params.use_xlsr:
            # Get encoder output
            feature = batch["inputs"].to(device)
            feature_lens = batch["supervisions"]["num_frames"].to(device)
            
            # Non-streaming forward pass for validation
            encoder_out, encoder_out_lens = model.encoder(feature, feature_lens)
            
            # Decode with beam search
            hyps = decode_with_beam_search(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                params=params,
                sp=sp,
                decoding_graph=params.decoding_graph if hasattr(params, "decoding_graph") else None
            )
            
            # Get reference texts
            refs = batch["supervisions"]["text"]
            
            # Update WER stats
            for ref, hyp in zip(refs, hyps):
                wer.add_sentence(ref, hyp)
        
        tot_loss = tot_loss + loss_info
    
    if world_size > 1:
        tot_loss.reduce(loss.device)
        
    loss_value = tot_loss["loss"] / tot_loss["frames"]
    
    # Get final WER
    if params.use_xlsr:
        wer_stats = wer.get_stats()
        logging.info(f"WER: {wer_stats['wer']:.2f}%")
        logging.info(f"Number of words: {wer_stats['num_words']}")
        logging.info(f"Number of errors: {wer_stats['num_errors']}")
        
        if hasattr(params, "best_valid_wer"):
            if wer_stats["wer"] < params.best_valid_wer:
                params.best_valid_wer = wer_stats["wer"]
                params.best_valid_epoch = params.cur_epoch
    
    return tot_loss


class WERComputer:
    """Compute Word Error Rate."""
    def __init__(self):
        self.total_words = 0
        self.total_errors = 0
        
    def add_sentence(self, ref: str, hyp: str) -> None:
        """Add a sentence pair to WER computation.
        
        Args:
            ref: Reference text
            hyp: Hypothesis text
        """
        # Normalize texts
        ref = self._normalize_text(ref)
        hyp = self._normalize_text(hyp)
        
        # Split into words
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        # Compute Levenshtein distance
        distance = self._levenshtein_distance(ref_words, hyp_words)
        
        # Update statistics
        self.total_words += len(ref_words)
        self.total_errors += distance
    
    def get_stats(self) -> Dict[str, Union[float, int]]:
        """Get WER statistics.
        
        Returns:
            Dictionary containing:
                - wer: Word Error Rate as percentage
                - num_words: Total number of words
                - num_errors: Total number of errors
        """
        wer = (self.total_errors / self.total_words * 100) if self.total_words > 0 else float("inf")
        return {
            "wer": wer,
            "num_words": self.total_words,
            "num_errors": self.total_errors
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for WER computation."""
        # Convert to lowercase
        text = text.lower()
        # Remove multiple spaces
        text = " ".join(text.split())
        # Remove punctuation except apostrophes
        text = re.sub(r'[^\w\s\']', '', text)
        return text
    
    def _levenshtein_distance(self, s1: List[str], s2: List[str]) -> int:
        """Compute Levenshtein distance between two word lists."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    scheduler: LRSchedulerType,
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch."""
    params.batch_idx_train = 0
    model.train()
    
    # Progressive unfreezing for XLSR
    if params.use_xlsr:
        cur_epoch = params.cur_epoch
        # Initially freeze all encoder layers
        for layer in model.encoder.model.encoder.layers:
            layer.requires_grad_(False)
        
        # Unfreeze last 4 layers initially
        for layer in model.encoder.model.encoder.layers[-4:]:
            layer.requires_grad_(True)
        
        # Check unfreeze schedule
        for epoch, num_layers in params.unfreeze_schedule:
            if cur_epoch >= epoch:
                # Unfreeze specified number of layers from the end
                for layer in model.encoder.model.encoder.layers[-num_layers:]:
                    layer.requires_grad_(True)
    
    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            mixed_precision = None
            if params.use_bf16:
                mixed_precision = "bf16"
            elif params.use_fp16:
                mixed_precision = "fp16"

            with amp.autocast('cuda', enabled=mixed_precision is not None, dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            set_batch_count(model, params.batch_idx_train)
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except:  # noqa
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()
            if cur_grad_scale < 1.0 or (cur_grad_scale < 8.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                raise_grad_scale_is_too_small_error(cur_grad_scale)

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, "
                f"batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )

        # Run validation and WER evaluation every 500 batches
        if batch_idx % 500 == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss and WER")
            valid_info = compute_validation_loss(
                params=params,
                model=model,
                sp=sp,
                valid_dl=valid_dl,
                world_size=world_size,
            )
            model.train()
            logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
            logging.info(
                f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
            )
            if tb_writer is not None:
                valid_info.write_summary(
                    tb_writer, "train/valid_", params.batch_idx_train
                )

    loss_value = tot_loss["loss"] / tot_loss["frames"]

    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss

    # After model initialization
    if params.use_xlsr:
        # Gradually unfreeze layers
        for layer in model.encoder.model.encoder.layers:
            layer.requires_grad_(False)
            
        # Unfreeze last 4 layers initially
        for layer in model.encoder.model.encoder.layers[-4:]:
            layer.requires_grad_(True)
            
        # Schedule more layers to unfreeze later
        scheduler.unfreeze_schedule = [
            (10, 8),  # At epoch 10, unfreeze 8 more layers
            (20, 12)  # At epoch 20, unfreeze all
        ]


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))
    if params.full_libri is False:
        params.valid_interval = 1600

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    
    # For Estonian dataset, blank_id is 1 since k2.SymbolTable reserves 0 for <eps>
    if params.dataset == "estonian":
        params.blank_id = 1
    else:
        params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    parameters_names = []
    parameters_names.append(
        [name_param_pair[0] for name_param_pair in model.named_parameters()]
    )
    optimizer = ScaledAdam(
        model.parameters(),
        lr=params.base_lr,
        clipping_scale=2.0,
        parameters_names=parameters_names,
    )

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    if params.dataset == "estonian":
        from estonian_dataset import EstonianASRDataset, collate_fn
        logging.info("Using Estonian dataset")
        
        train_dataset = EstonianASRDataset(
            params.train_txt, 
            base_path=params.audio_base_path,
            min_duration=1.0,  # 1 second minimum
            max_duration=10.0  # 10 seconds maximum
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=params.seed,
        ) if world_size > 1 else None
        
        train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            persistent_workers=True,
        )

        valid_dataset = EstonianASRDataset(
            params.val_txt, 
            base_path=params.audio_base_path,
            min_duration=1.0,  # 1 second minimum
            max_duration=10.0  # 10 seconds maximum
        )
        valid_dl = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )

        # For Estonian dataset, we use the vocab_size and blank_id from the BPE model
        logging.info(f"Using vocab_size={params.vocab_size} and blank_id={params.blank_id} from BPE model")
    else:
        # Original Librispeech data loading code
        librispeech = LibriSpeechAsrDataModule(args)

        if params.full_libri:
            train_cuts = librispeech.train_all_shuf_cuts()
        else:
            train_cuts = librispeech.train_clean_100_cuts()

        def remove_short_and_long_utt(c: Cut):
            if c.duration < 1.0 or c.duration > 20.0:
                logging.warning(
                    f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
                )
                return False
            return True

        train_cuts = train_cuts.filter(remove_short_and_long_utt)

        if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
            sampler_state_dict = checkpoints["sampler"]
        else:
            sampler_state_dict = None

        train_dl = librispeech.train_dataloaders(
            train_cuts, sampler_state_dict=sampler_state_dict
        )

        valid_cuts = librispeech.dev_clean_cuts()
        valid_cuts += librispeech.dev_other_cuts()
        valid_dl = librispeech.valid_dataloaders(valid_cuts)

    # if not params.print_diagnostics:
    #     scan_pessimistic_batches_for_oom(
    #         model=model,
    #         train_dl=train_dl,
    #         optimizer=optimizer,
    #         sp=sp,
    #         params=params,
    #     )

    scaler = amp.GradScaler('cuda', enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        if hasattr(train_dl.sampler, 'set_epoch'):
            train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    y = sp.encode(supervisions["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params, sp=sp)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def run_accelerate(accelerator, args):
    # This function implements the training loop using HF Accelerate
    params = get_params()
    params.update(vars(args))
    if params.full_libri is False:
        params.valid_interval = 1600

    fix_random_seed(params.seed)
    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started (Accelerate Mode)")

    tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard") if (args.tensorboard and accelerator.is_main_process) else None

    # Set device from accelerator
    device = accelerator.device
    logging.info(f"Using Accelerator device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()
    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)
    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    optimizer = ScaledAdam(
        model.parameters(),
        lr=params.base_lr,
        clipping_scale=2.0,
        parameters_names=[ [name for name, _ in model.named_parameters()] ]
    )
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    # Data loading
    if params.dataset == "estonian":
        from estonian_dataset import EstonianASRDataset, collate_fn
        logging.info("Using Estonian dataset")
        train_dataset = EstonianASRDataset(params.train_txt, base_path=params.audio_base_path)
        train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            persistent_workers=True,
        )
        valid_dataset = EstonianASRDataset(params.val_txt, base_path=params.audio_base_path)
        valid_dl = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )
        logging.info(f"Using vocab_size={params.vocab_size} and blank_id={params.blank_id} from BPE model")
    else:
        # Original Librispeech data loading code
        librispeech = LibriSpeechAsrDataModule(args)
        if params.full_libri:
            train_cuts = librispeech.train_all_shuf_cuts()
        else:
            train_cuts = librispeech.train_clean_100_cuts()
        train_dl = librispeech.train_dataloaders(train_cuts)
        valid_cuts = librispeech.dev_clean_cuts()
        valid_cuts += librispeech.dev_other_cuts()
        valid_dl = librispeech.valid_dataloaders(valid_cuts)

    # Prepare with accelerator
    model, optimizer, scheduler, train_dl, valid_dl = accelerator.prepare(model, optimizer, scheduler, train_dl, valid_dl)

    model.to(device)

    # Optionally load checkpoint if available
    model_avg = None
    if accelerator.is_main_process:
        try:
            import copy
            model_avg = copy.deepcopy(model).to(torch.float64)
        except Exception as e:
            logging.warning(f"Could not create averaged model: {e}")

    # Main training loop
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)

        if hasattr(train_dl, 'sampler') and hasattr(train_dl.sampler, 'set_epoch'):
            train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch
        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=amp.GradScaler(enabled=params.use_fp16, init_scale=1.0),
            tb_writer=tb_writer,
            world_size=1,
            rank=0,
        )

        if accelerator.is_main_process:
            save_checkpoint(
                params=params,
                model=model,
                model_avg=model_avg,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler if hasattr(train_dl, 'sampler') else None,
                scaler=amp.GradScaler(enabled=params.use_fp16, init_scale=1.0),
                rank=0,
            )

    logging.info("Training completed (Accelerate Mode)!")


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    if args.use_accelerate:
        try:
            from accelerate import Accelerator
        except ImportError:
            raise ImportError("Please install hf accelerate (pip install accelerate) or set --use-accelerate False")
        accelerator = Accelerator()
        run_accelerate(accelerator, args)
    else:
        world_size = args.world_size
        assert world_size >= 1
        if world_size > 1:
            mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
        else:
            run(rank=0, world_size=1, args=args)

    torch.set_num_threads(1)
    torch.set_interop_threads(1)


if __name__ == "__main__":
    main()
