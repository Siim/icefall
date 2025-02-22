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
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
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
from icefall.utils import AttributeDict, MetricsTracker, setup_logger, str2bool
from torch import amp
import editdistance
import random

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

    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before updating weights",
    )

    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=8,
        help="Target batch size after gradient accumulation",
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
            # parameters for conformer
            "subsampling_factor": 4,
            "vgg_frontend": False,
            "use_feat_batchnorm": True,
            "feature_dim": 80,
            "nhead": 8,
            "attention_dim": 512,
            "num_decoder_layers": 6,
            
            # parameters for streaming (from paper)
            "decode_chunk_size": 5120,  # 320ms at 16kHz (optimal)
            "chunk_sizes": {
                "320ms": 5120,   # 16 frames
                "640ms": 10240,  # 32 frames
                "1280ms": 20480, # 64 frames
                "2560ms": 40960  # 128 frames
            },
            "use_attention_sink": True,
            "attention_sink_size": 16,  # Paper's optimal setting
            "left_context_chunks": 1,  # Paper's optimal setting
            "streaming_regularization": 0.1,  # Weight for streaming regularization
            
            # parameters for Noam
            "model_warm_step": 3000,  # arg given to model, not for lrate
            "env_info": get_env_info(),
            "use_xlsr": False,  # Whether to use XLSR encoder
            "xlsr_model_name": "facebook/wav2vec2-xls-r-300m",
            
            # parameters for loss
            "simple_loss_scale": 0.5,
            "prune_range": 5,
            "lm_scale": 0.25,
            "am_scale": 0.0,
            
            # parameters for decoding
            "search_beam": 20,
            "output_beam": 8,
            "min_active_states": 30,
            "max_active_states": 10000,
            "use_double_scores": True,
            
            # parameters for training
            "context_size": 2,
            "max_duration": 200.0,
            "random_seed": 42,
            "batch_size": 4,
            "num_epochs": 30,
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "exp_dir": Path("pruned_transducer_stateless7_streaming/exp"),
            "lang_dir": Path("data/lang_bpe_2500"),
            "vocab_file": "data/lang_bpe_2500/tokens.txt",  # Path to vocabulary file
            
            # Training parameters from paper
            "base_lr": 1e-3,
            "warmup_steps": 10000,
            "min_lr": 1e-5,
            "lr_decay": "linear",  # Linear decay after warmup
            
            # Optimizer settings from paper
            "adam_betas": (0.9, 0.98),
            "adam_eps": 1e-6,
            "weight_decay": 0.01,
            "grad_clip": 5.0,
            
            # Multi-chunk training
            "min_chunks": 2,  # Minimum chunks per sequence
            "max_chunks": 4,  # Maximum chunks per sequence
            
            # Other training parameters
            "lr": 1e-3,
            "weight_decay": 1e-6,
            "warm_step": 2000,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 2000,  # Match save_every_n for consistent evaluation
            "save_every_n": 2000,
            "keep_last_k": 20,
            "average_period": 100,
            "use_fp16": False,
            "epoch": 1,
            "return_encoder_output": False,  # used only during inference
            "return_boundaries": False,  # used only during inference
            "use_averaged_model": False,  # used only during inference
            "dataset": "librispeech",  # "librispeech" or "estonian"
            "train_txt": None,  # Path to train.txt for Estonian dataset
            "val_txt": None,  # Path to val.txt for Estonian dataset
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
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)

    texts = batch["supervisions"]["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y).to(device)

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss = None, None
        streaming_loss = torch.tensor([0.0], device=device)

        if is_training and params.use_xlsr:
            # Get random chunk size during training
            chunk_size = random.choice(list(params.chunk_sizes.values()))
            chunks = model.encoder.prepare_chunks(feature, chunk_size)
            encoder_out_chunks = []
            prev_chunk_last = None

            # Process each chunk
            for chunk in chunks:
                chunk_len = torch.tensor([chunk.shape[1]], dtype=torch.int32, device=device)
                if prev_chunk_last is None:
                    # First chunk
                    chunk_out, _, states = model.encoder.streaming_forward(chunk, chunk_len, None)
                    prev_chunk_last = chunk_out[:, -1:, :]
                else:
                    # Subsequent chunks - compute streaming regularization
                    chunk_out, _, states = model.encoder.streaming_forward(chunk, chunk_len, states)
                    curr_chunk_first = chunk_out[:, :1, :]
                    # Compute transition loss between chunks
                    streaming_loss += F.mse_loss(prev_chunk_last, curr_chunk_first)
                    prev_chunk_last = chunk_out[:, -1:, :]
                
                encoder_out_chunks.append(chunk_out)

            # Concatenate chunks
            encoder_out = torch.cat(encoder_out_chunks, dim=1)
            
            # Scale streaming loss by number of chunks
            if len(chunks) > 1:
                streaming_loss = streaming_loss / (len(chunks) - 1)
        else:
            # Regular forward pass for validation or non-XLSR
            encoder_out, _ = model.encoder(feature, feature_lens)

        # Get decoder output
        decoder_out = model.decoder(y)
        
        # Compute joiner output
        joiner_out = model.joiner(encoder_out, decoder_out)

        # Compute losses
        simple_loss = model.simple_loss(joiner_out, y)
        
        if params.batch_idx_train > params.model_warm_step:
            pruned_loss = model.pruned_loss(joiner_out, y)

        # Combine losses
        loss = simple_loss
        if pruned_loss is not None:
            loss += pruned_loss
        
        # Add streaming regularization if training with XLSR
        if is_training and params.use_xlsr:
            loss += params.streaming_regularization * streaming_loss

    assert loss.requires_grad == is_training

    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    info["loss"] = loss.detach().cpu().item()
    if simple_loss is not None:
        info["simple_loss"] = simple_loss.detach().cpu().item()
    if pruned_loss is not None:
        info["pruned_loss"] = pruned_loss.detach().cpu().item()
    if streaming_loss.item() != 0:
        info["streaming_loss"] = streaming_loss.detach().cpu().item()

    return loss, info


def decode_batch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    max_decode_samples: int = 4,
) -> List[Tuple[str, str, float]]:
    """Decode a random subset of the batch and compute WER for monitoring."""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Select up to max_decode_samples from the batch
        indices = torch.randperm(len(batch["supervisions"]["text"]))[:max_decode_samples]
        
        # Prepare mini-batch for decoding
        texts = [batch["supervisions"]["text"][i] for i in indices]
        features = batch["inputs"][indices].to(device)
        feature_lens = batch["supervisions"]["num_frames"][indices].to(device)
        
        # Add right context for streaming
        feature_lens += params.attention_sink_size
        feature = torch.nn.functional.pad(
            features,
            pad=(0, 0, 0, params.attention_sink_size),
            value=LOG_EPS,
        )
        
        # Get encoder output with streaming settings
        if hasattr(model.encoder, 'streaming_forward'):
            # Initialize streaming state
            states = model.encoder.get_init_state(device)
            
            # Process in chunks
            current = 0
            encoder_out_chunks = []
            chunk_overlap = params.decode_chunk_len // 2
            
            while current < feature.size(1):
                end = min(current + params.decode_chunk_len, feature.size(1))
                chunk = feature[:, current:end]
                chunk_len = torch.tensor([chunk.shape[1]], dtype=torch.int32, device=device)
                
                # Process chunk with streaming forward
                chunk_out, chunk_lens, states = model.encoder.streaming_forward(
                    chunk, 
                    chunk_len, 
                    states
                )
                encoder_out_chunks.append(chunk_out)
                
                # Move to next chunk considering overlap
                if end == feature.size(1):  # Last chunk
                    break
                current = end - chunk_overlap
            
            # Concatenate chunks
            encoder_out = torch.cat(encoder_out_chunks, dim=1)
            encoder_out_lens = torch.tensor([encoder_out.size(1)], dtype=torch.int32, device=device)
        else:
            # Fallback to regular forward pass
            encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)
        
        # Log encoder output statistics
        logging.info(f"\nEncoder output stats:")
        logging.info(f"Shape: {encoder_out.shape}")
        logging.info(f"Mean: {encoder_out.mean().item():.3f}")
        logging.info(f"Std: {encoder_out.std().item():.3f}")
        logging.info(f"Min: {encoder_out.min().item():.3f}")
        logging.info(f"Max: {encoder_out.max().item():.3f}")
        
        hyps = []
        # Use FSA decoding for Estonian
        if params.dataset == "estonian" and hasattr(model, "decoding_graph"):
            from estonian_decoder import fast_beam_search_one_best
            hyp_tokens = fast_beam_search_one_best(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=40.0,  # Paper's optimal setting
                max_states=128,  # Paper's optimal setting
                max_contexts=16,  # Paper's optimal setting
                decoding_graph=model.decoding_graph.to(device)
            )
            for hyp in hyp_tokens:
                hyps.append([model.decoder.token_table[i] for i in hyp])
        else:
            # Fallback to greedy search for quick monitoring
            hyp_tokens = greedy_search_batch(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens
            )
            for tokens in hyp_tokens:
                text = sp.decode(tokens)
                hyps.append(text.split())
        
        total_words = 0
        total_errors = 0
        results = []
        
        for i, hyp_words in enumerate(hyps):
            ref = texts[i]
            ref_words = ref.split()
            # Compute WER for this sample
            errors = editdistance.eval(ref_words, hyp_words)
            wer = errors / len(ref_words) * 100 if ref_words else 0
            
            total_words += len(ref_words)
            total_errors += errors
            
            logging.info(f"\nReference: {ref}")
            logging.info(f"Hypothesis: {' '.join(hyp_words)}")
            logging.info(f"Sample WER: {wer:.1f}%")
            results.append((ref, ' '.join(hyp_words), wer))
            
        # Log overall WER for this batch
        if total_words > 0:
            batch_wer = (total_errors / total_words) * 100
            logging.info(f"\nBatch WER: {batch_wer:.1f}% (Total words: {total_words}, Total errors: {total_errors})")
    
    model.train()
    return results


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

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info
        
        # Decode samples when we would save a checkpoint
        if batch_idx == 0:
            logging.info(f"\nDecoding samples from validation batch {batch_idx}")
            decode_batch(params, model, sp, batch)

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


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
    model.train()

    tot_loss = MetricsTracker()

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

    # Initialize gradient accumulation
    optimizer.zero_grad()
    accumulation_steps = getattr(params, "gradient_accumulation_steps", 1)
    
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
                # Scale loss by accumulation steps
                loss = loss / accumulation_steps
                
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()
            
            # Only step optimizer after accumulating enough gradients
            if (batch_idx + 1) % accumulation_steps == 0:
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

    optimizer = get_optimizer(model, params)

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
        
        train_dataset = EstonianASRDataset(params.train_txt, base_path=params.audio_base_path)
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

        valid_dataset = EstonianASRDataset(params.val_txt, base_path=params.audio_base_path)
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

    scaler = amp.GradScaler(enabled=params.use_fp16 or params.use_bf16, init_scale=1.0)
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

        if not params.print_diagnostics:
            scan_pessimistic_batches_for_oom(
                model=model,
                train_dl=train_dl,
                optimizer=optimizer,
                sp=sp,
                params=params,
            )

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

    optimizer = get_optimizer(model, params)
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


def get_optimizer(model: nn.Module, params: AttributeDict) -> ScaledAdam:
    parameters_names = []
    parameters_names.append(
        [name_param_pair[0] for name_param_pair in model.named_parameters()]
    )
    
    optimizer = ScaledAdam(
        model.parameters(),
        lr=params.base_lr,
        betas=params.adam_betas,
        eps=params.adam_eps,
        clipping_scale=2.0,
        parameters_names=parameters_names
    )
    return optimizer


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
