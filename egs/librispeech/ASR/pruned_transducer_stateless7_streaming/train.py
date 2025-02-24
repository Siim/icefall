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
import editdistance
import math
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
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
from xlsr_encoder import XLSREncoder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

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
from beam_search import (
    beam_search,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    fast_beam_search_nbest_oracle,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
    modified_beam_search_lm_rescore,
    modified_beam_search_lm_rescore_LODR,
    modified_beam_search_lm_shallow_fusion,
    modified_beam_search_LODR,
)

LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]

LOG_EPS = math.log(1e-10)  # Small value for padding in log space


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

    parser.add_argument(
        "--use-xlsr",
        type=str2bool,
        default=True,
        help="Whether to use XLSR encoder instead of Zipformer.",
    )

    parser.add_argument(
        "--xlsr-model-name",
        type=str,
        default="TalTechNLP/xls-r-300m-et",
        help="Name of the XLSR model to use from HuggingFace.",
    )

    parser.add_argument(
        "--attention-sink-size",
        type=int,
        default=16,
        help="Number of frames to use for attention sink (paper's optimal setting).",
    )

    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=5120,
        help="Chunk size for decoding in samples (320ms at 16kHz).",
    )

    parser.add_argument(
        "--left-context-chunks",
        type=int,
        default=1,
        help="Number of left context chunks to use (paper's optimal setting).",
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
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
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
        default="Data/train_list.txt",
        help="Path to training data list file for Estonian dataset",
    )

    parser.add_argument(
        "--val-txt",
        type=str,
        default="Data/val_list.txt",
        help="Path to validation data list file for Estonian dataset",
    )

    parser.add_argument(
        "--audio-base-path",
        type=str,
        default=None,
        help="Base path for audio files in Estonian dataset",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for training",
    )

    parser.add_argument(
        "--streaming-regularization",
        type=float,
        default=0.1,
        help="Weight for streaming regularization loss",
    )

    parser.add_argument(
        "--ctc-epochs",
        type=int,
        default=5,
        help="Number of epochs for CTC pre-training",
    )

    add_model_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - encoder_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,
            
            # XLSR specific parameters
            "use_xlsr": True,
            "xlsr_model_name": "TalTechNLP/xls-r-300m-et",
            "attention_sink_size": 16,  # Paper's optimal setting
            "decode_chunk_size": 5120,  # 320ms at 16kHz
            "left_context_chunks": 1,   # Paper's optimal setting
            
            # Frame parameters from paper
            "frame_duration": 0.025,  # 25ms per frame
            "frame_stride": 0.020,   # 20ms stride
            
            # Original parameters
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
            "warm_step": 2000,
            "env_info": get_env_info(),
            "ctc_epochs": 5,
        }
    )

    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    if getattr(params, 'use_xlsr', False):
        from xlsr_encoder import XLSREncoder
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        # Initialize cur_epoch if not present
        if not hasattr(params, 'cur_epoch'):
            params.cur_epoch = 0

        # Initialize ctc_epochs if not present
        if not hasattr(params, 'ctc_epochs'):
            params.ctc_epochs = 5  # Default to 5 epochs of CTC pre-training

        # Load processor first
        try:
            processor = Wav2Vec2Processor.from_pretrained(params.xlsr_model_name)
            logging.info(f"Successfully loaded XLSR processor")
            params.wav2vec2_processor = processor
        except Exception as e:
            logging.warning(f"Could not load processor: {str(e)}")

        # Load appropriate model based on training phase
        if params.cur_epoch <= params.ctc_epochs:
            # During CTC pre-training, use Wav2Vec2ForCTC
            model = Wav2Vec2ForCTC.from_pretrained(params.xlsr_model_name)
            logging.info(f"Using Wav2Vec2ForCTC for pre-training (epoch {params.cur_epoch}/{params.ctc_epochs})")
        else:
            # After pre-training, use base model
            model = Wav2Vec2Model.from_pretrained(params.xlsr_model_name)
            logging.info(f"Using base Wav2Vec2Model for transducer training")

        # Create XLSR encoder with streaming capabilities
        encoder = XLSREncoder(
            model=model,
            decode_chunk_size=params.decode_chunk_len,
            chunk_overlap=params.decode_chunk_len // 2,
            use_attention_sink=True,
            attention_sink_size=16,  # Paper's optimal setting
            frame_duration=0.025,    # 25ms per frame
            frame_stride=0.020,      # 20ms stride
            min_chunk_size=2560,     # 160ms at 16kHz
            max_chunk_size=20480,    # 1280ms at 16kHz
            left_context_chunks=1     # Paper's optimal setting
        )
        return encoder
    else:
        # Original Zipformer code...
        def to_int_tuple(s: str):
            return tuple(map(int, s.split(",")))

        encoder = Zipformer(
            num_features=params.feature_dim,
            output_downsampling_factor=2,
            zipformer_downsampling_factors=to_int_tuple(params.zipformer_downsampling_factors),
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
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    # For XLSR encoder, output dim is fixed at 1024 for XLS-R 300M
    encoder_dim = 1024 if params.use_xlsr else int(params.encoder_dims.split(",")[-1])
    
    joiner = Joiner(
        encoder_dim=encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    # For XLSR encoder, output dim is fixed at 1024 for XLS-R 300M
    encoder_dim = 1024 if params.use_xlsr else int(params.encoder_dims.split(",")[-1])

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
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
    
    # Handle sampler state dict
    sampler_state = None
    if sampler is not None and hasattr(sampler, 'state_dict'):
        sampler_state = sampler.state_dict()
    
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler_state,  # Pass the state instead of the sampler
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute transducer loss given the model and batch.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    feature = feature.to(device)
    
    # Fix input shape for XLSR - remove extra dimensions
    if feature.ndim == 4:  # [batch, channel, time, extra_dim]
        feature = feature.squeeze(-1)  # Remove last dimension
    if feature.ndim == 3:  # [batch, channel, time]
        feature = feature.squeeze(1)  # Remove channel dimension
    assert feature.ndim == 2, f"Expected 2D input (batch, time), got shape {feature.shape}"
    
    # Get supervisions
    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    texts = supervisions["text"]
    tokens = supervisions["tokens"].to(device)
    token_lens = supervisions["token_lens"].to(device)
    
    # Calculate output lengths after XLSR's downsampling
    xlsr_downsample = 320  # XLSR/wav2vec2 downsampling factor
    output_lens = torch.div(feature_lens, xlsr_downsample, rounding_mode='floor')
    output_lens = torch.maximum(output_lens, torch.ones_like(output_lens))  # Ensure at least 1 frame
    
    # Compute with appropriate gradient context
    with torch.set_grad_enabled(is_training):
        # Use CTC loss for pre-training if in early epochs
        if params.cur_epoch <= params.ctc_epochs:
            # Get CTC outputs directly from Wav2Vec2ForCTC
            if isinstance(model.encoder.model, Wav2Vec2ForCTC):
                # Create attention mask for proper padding
                attention_mask = torch.ones_like(feature, dtype=torch.long)
                for i in range(attention_mask.size(0)):
                    attention_mask[i, feature_lens[i]:] = 0
                
                # Forward through CTC model to get logits
                outputs = model.encoder.model(
                    feature,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Get logits and project to our vocabulary size if needed
                logits = outputs.logits  # [batch, time, wav2vec2_vocab_size]
                
                # Project to our vocabulary size using the model's projection layer
                if isinstance(model, DDP):
                    logits = model.module.simple_am_proj(outputs.hidden_states[-1])
                else:
                    logits = model.simple_am_proj(outputs.hidden_states[-1])
                
                # Ensure output lengths don't exceed logits length
                max_logit_len = logits.size(1)
                output_lens = torch.minimum(output_lens, torch.tensor(max_logit_len, device=output_lens.device))
                
                if is_training:
                    # During training, compute CTC loss
                    log_probs = torch.log_softmax(logits, dim=-1)
                    ctc_loss = torch.nn.functional.ctc_loss(
                        log_probs.transpose(0, 1),  # (T, B, V)
                        tokens,
                        output_lens,
                        token_lens,
                        blank=params.blank_id,
                        reduction='sum',
                        zero_infinity=True
                    )
                    simple_loss = ctc_loss
                    pruned_loss = ctc_loss
                    wer = 0.0  # Don't compute WER during training
                else:
                    # During validation, compute WER
                    with torch.no_grad():
                        # Get predictions
                        predictions = torch.argmax(logits, dim=-1)
                        
                        # Convert predictions to text for WER calculation
                        hyp_texts = []
                        for i, length in enumerate(output_lens):
                            # Get sequence without padding
                            pred = predictions[i, :length].tolist()
                            
                            # CTC decoding - remove repeated and blanks
                            decoded = []
                            prev = -1
                            for token in pred:
                                if token != params.blank_id and token != prev:
                                    decoded.append(token)
                                prev = token
                            
                            # Convert to text
                            text = sp.decode(decoded)
                            hyp_texts.append(text)
                        
                        # Calculate WER
                        total_words = sum(len(ref.split()) for ref in texts)
                        total_errors = sum(editdistance.eval(hyp.split(), ref.split()) 
                                        for hyp, ref in zip(hyp_texts, texts))
                        wer = 100.0 * total_errors / total_words if total_words > 0 else float('inf')
                        
                        # Compute loss for validation metrics
                        log_probs = torch.log_softmax(logits, dim=-1)
                        ctc_loss = torch.nn.functional.ctc_loss(
                            log_probs.transpose(0, 1),  # (T, B, V)
                            tokens,
                            output_lens,
                            token_lens,
                            blank=params.blank_id,
                            reduction='sum',
                            zero_infinity=True
                        )
                        simple_loss = ctc_loss
                        pruned_loss = ctc_loss
            else:
                raise ValueError("Encoder model must be Wav2Vec2ForCTC during pre-training epochs")
        else:
            # Use transducer loss after pre-training
            encoder_out, encoder_out_lens = model.encoder(feature, feature_lens)
            
            if isinstance(model, DDP):
                encoder_out = model.module.encoder_proj(encoder_out)
            else:
                encoder_out = model.encoder_proj(encoder_out)
                
            simple_loss, pruned_loss = model(
                x=feature,
                x_lens=feature_lens,
                y=k2.RaggedTensor(tokens),
                prune_range=params.prune_range,
                am_scale=params.am_scale,
                lm_scale=params.lm_scale,
            )
            wer = 0.0  # Don't compute WER during training
        
        # Scale losses
        s = params.simple_loss_scale
        simple_loss_scale = (
            s
            if "warm_step" not in params
            else (
                s
                if params.batch_idx_train > params.warm_step
                else float(params.batch_idx_train) / params.warm_step * s
            )
        )
        pruned_loss_scale = (
            1.0
            if "warm_step" not in params
            else (
                1.0
                if params.batch_idx_train > params.warm_step
                else float(params.batch_idx_train) / params.warm_step
            )
        )
        
        loss = simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss
    
    info = MetricsTracker()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()
    
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()
    info["wer"] = wer
    
    return loss, info


def decode_one_batch_hyps(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
) -> Tuple[List[str], List[str]]:
    """Get hypotheses and predictions for one batch."""
    device = next(model.parameters()).device
    feature = batch["inputs"].to(device)
    supervisions = batch["supervisions"]
    feature_lens = supervisions["num_frames"].to(device)
    
    # Get encoder output with streaming settings
    with torch.no_grad():
        # Add right context padding for streaming
        right_context = params.decode_chunk_len // 2
        feature_lens_pad = feature_lens + right_context
        feature_pad = torch.nn.functional.pad(
            feature,
            pad=(0, right_context),  # Only pad the time dimension
            value=LOG_EPS,
        )
        
        # Create attention mask for proper padding
        attention_mask = torch.ones(feature_pad.shape[:2], dtype=torch.long, device=device)  # [batch, time]
        for i in range(attention_mask.shape[0]):  # Iterate over batch dimension
            if i < feature_lens_pad.shape[0]:  # Check if index is valid
                attention_mask[i, feature_lens_pad[i]:] = 0
        
        # Get encoder output
        if isinstance(model.encoder.model, Wav2Vec2ForCTC):
            # For CTC pre-training, use the CTC model's forward pass
            outputs = model.encoder.model(
                feature_pad,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            encoder_out = outputs.hidden_states[-1]
            encoder_out_lens = torch.div(feature_lens_pad, 320, rounding_mode='floor')  # XLSR downsample factor
            encoder_out_lens = torch.maximum(encoder_out_lens, torch.ones_like(encoder_out_lens))
            
            # Project to vocab size
            if isinstance(model, DDP):
                logits = model.module.simple_am_proj(encoder_out)  # [B, T, V]
            else:
                logits = model.simple_am_proj(encoder_out)  # [B, T, V]
            
            # Get CTC predictions
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Decode each sequence in batch
            hyp_tokens = []
            for b in range(log_probs.size(0)):
                # Get sequence without padding
                sequence_log_probs = log_probs[b, :encoder_out_lens[b]]  # [T, V]
                
                # Get best path
                best_path = torch.argmax(sequence_log_probs, dim=-1)  # [T]
                
                # CTC decoding - collapse repeated tokens and remove blanks
                decoded = []
                prev = -1
                for token in best_path:
                    token = token.item()
                    if token != params.blank_id and token != prev:
                        decoded.append(token)
                    prev = token
                
                hyp_tokens.append(decoded)
                
                # Debug logging
                if b == 0:  # Log first sequence in batch
                    logging.debug(f"Sequence {b} length: {encoder_out_lens[b]}")
                    logging.debug(f"Raw best path: {best_path[:10].tolist()}")  # First 10 tokens
                    logging.debug(f"Decoded tokens: {decoded[:10]}")  # First 10 decoded tokens
                    
        else:
            # Regular encoder forward pass
            encoder_out, encoder_out_lens = model.encoder(feature_pad, feature_lens_pad)
            
            # Project encoder output
            if isinstance(model, DDP):
                encoder_out = model.module.encoder_proj(encoder_out)
            else:
                encoder_out = model.encoder_proj(encoder_out)
            
            # Use transducer decoding
            hyp_tokens = greedy_search_batch(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
            )
    
    # Convert token IDs to text
    hyps = []
    preds = []
    
    # Get ground truth
    texts = supervisions["text"]
    
    # Process each item in the batch
    for i in range(len(texts)):
        # Ground truth
        hyps.append(texts[i])
        
        # Prediction - handle both tensor and list outputs
        pred_tokens = hyp_tokens[i]
        if isinstance(pred_tokens, torch.Tensor):
            pred_tokens = pred_tokens.tolist()
        elif not isinstance(pred_tokens, list):
            pred_tokens = list(pred_tokens)
        
        # Debug logging for first sequence
        if i == 0:
            logging.debug(f"Prediction tokens before filtering: {pred_tokens[:10]}")
            
        # Remove any padding or special tokens
        filtered_tokens = []
        for token in pred_tokens:
            if token != params.blank_id:
                filtered_tokens.append(token)
        
        # Debug logging for first sequence
        if i == 0:
            logging.debug(f"Prediction tokens after filtering: {filtered_tokens[:10]}")
            
        # Convert to text
        try:
            pred = sp.decode(filtered_tokens)
        except Exception as e:
            logging.warning(f"Failed to decode tokens {filtered_tokens}: {str(e)}")
            pred = ""
        
        # Debug logging for first sequence
        if i == 0:
            logging.debug(f"Decoded prediction: {pred}")
            
        preds.append(pred)
    
    return hyps, preds


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

        if batch_idx % 100 == 0:
            logging.info(f"batch {batch_idx}, validation loss: {loss_info}")
            
            # Get hypotheses and predictions for the first few examples
            hyps, preds = decode_one_batch_hyps(params, model, sp, batch)
            num_to_print = min(3, len(hyps))
            
            logging.info("\nExample comparisons:")
            logging.info("-" * 80)
            for i in range(num_to_print):
                logging.info(f"HYP[{i}]: {hyps[i]}")
                logging.info(f"PRD[{i}]: {preds[i]}")
                logging.info("-" * 80)

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["frames"]
    if loss_value < params.best_valid_loss:
        params.best_valid_loss = loss_value
        params.best_valid_epoch = params.cur_epoch

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

    for batch_idx, batch in enumerate(train_dl):
        params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            with torch.amp.autocast('cuda', enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

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
                sampler=train_dl.sampler if params.dataset == "librispeech" else None,
                scaler=scaler,
                rank=rank,
            )
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it
            cur_grad_scale = scaler._scale.item()
            if cur_grad_scale < 1.0 or (cur_grad_scale < 8.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                raise_grad_scale_is_too_small_error(cur_grad_scale)

        if batch_idx % params.log_interval == 0:
            cur_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, loss[{loss_info}], "
                f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                f"lr: {cur_lr:.2e}, "
                f"grad_scale: {scaler.get_scale():.1f}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar(
                    "train/learning_rate", cur_lr, params.batch_idx_train
                )

                # Write all loss components to tensorboard
                for k, v in loss_info.items():
                    tb_writer.add_scalar(f"train/{k}", v, params.batch_idx_train)

        if batch_idx % params.valid_interval == 0 and not params.print_diagnostics:
            logging.info("Computing validation loss")
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
    
    # Initialize training parameters
    params.cur_epoch = max(1, params.start_epoch)  # Start from epoch 1 or start_epoch
    params.batch_idx_train = 0
    
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

    # <blk> is defined in local/train_bpe_model.py
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
        
        # Get processor from params (set during encoder initialization)
        processor = getattr(params, 'wav2vec2_processor', None)
        if processor is None:
            logging.warning("No XLSR processor found, will use basic audio normalization")
        
        train_dataset = EstonianASRDataset(
            params.train_txt, 
            base_path=params.audio_base_path, 
            sp=sp,
            processor=processor
        )
        train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )
        
        valid_dataset = EstonianASRDataset(
            params.val_txt, 
            base_path=params.audio_base_path, 
            sp=sp,
            processor=processor
        )
        valid_dl = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )
    else:
        librispeech = LibriSpeechAsrDataModule(args)
        
        if params.mini_libri:
            train_cuts = librispeech.train_clean_5_cuts()
        else:
            if params.full_libri:
                train_cuts = librispeech.train_all_shuf_cuts()
            else:
                train_cuts = librispeech.train_clean_100_cuts()

        if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
            sampler_state_dict = checkpoints["sampler"]
        else:
            sampler_state_dict = None

        train_dl = librispeech.train_dataloaders(
            train_cuts, sampler_state_dict=sampler_state_dict
        )

        if params.mini_libri:
            valid_cuts = librispeech.dev_clean_2_cuts()
        else:
            valid_cuts = librispeech.dev_clean_cuts()
            valid_cuts += librispeech.dev_other_cuts()
        valid_dl = librispeech.valid_dataloaders(valid_cuts)

    if params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            sp=sp,
            params=params,
        )

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        
        if params.dataset != "estonian":
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
            sampler=train_dl.sampler if params.dataset != "estonian" else None,
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


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()