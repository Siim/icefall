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
import time
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union, List
import random

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
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
from beam_search import greedy_search_batch  # Original greedy search
from xlsr_beam_search import beam_search_batch  # Import our new beam search implementation
from xlsr_greedy_search import greedy_search_batch as xlsr_greedy_search_batch  # Import our enhanced greedy search

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

    parser.add_argument(
        "--pre-train-epochs",
        type=int,
        default=5,
        help="Number of epochs to pre-train without streaming before introducing chunks",
    )

    parser.add_argument(
        "--pre-train-lr",
        type=float,
        default=0.0001,
        help="Learning rate during pre-training phase",
    )

    parser.add_argument(
        "--streaming-start-epoch",
        type=int,
        default=6,
        help="Epoch to start using streaming (after pre-training)",
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
        "--use-bf16",
        type=str2bool,
        default=False,
        help="Whether to use bfloat16 precision training (requires GPU with BF16 support).",
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
        "--pretrain-epochs",
        type=int,
        default=5,
        help="Number of epochs for pre-training phase with full-sequence processing."
    )

    parser.add_argument(
        "--progressive-epochs",
        type=int,
        default=10,
        help="Number of epochs for progressive training stage transitioning to streaming."
    )

    parser.add_argument(
        "--streaming-epochs",
        type=int,
        default=15,
        help="Number of epochs for fully streaming training phase."
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
            "valid_interval": 500,
            
            # Paper's training configuration
            "pretrain_epochs": 5,  # Pre-training phase
            "streaming_epochs": 15, # Streaming phase
            "beam_size": 4,        # Paper's beam width for decoding
            
            # Learning rate schedule from paper
            "warmup_steps": 500,
            "base_lr": 1.25e-3,
            
            # Chunk configurations from paper
            "chunk_sizes": {
                "320ms": 5120,   # 16 frames (best performing)
                "640ms": 10240,  # 32 frames
                "1280ms": 20480, # 64 frames
                "2560ms": 40960  # 128 frames
            },
            
            # Frame parameters (from paper)
            "frame_duration": 0.025,  # 25ms per frame
            "frame_stride": 0.020,    # 20ms stride
            "downsample_factor": 320, # For wav2vec2/XLSR models
            
            # Attention sink parameters (from paper)
            "attention_sink_size": 16,  # Paper's optimal setting
            "left_context_chunks": 1,   # Paper's optimal setting
            
            # Memory management
            "batch_size": {
                "non_streaming": 4,  # Larger as no chunking overhead
                "streaming": 2       # Smaller due to chunk processing
            },
            "gradient_accumulation_steps": {
                "non_streaming": 4,
                "streaming": 8
            },
            "max_duration": {
                "non_streaming": 10.0,  # seconds
                "streaming": 8.0        # seconds
            },
            
            # Original parameters
            "feature_dim": 80,
            "subsampling_factor": 4,
            "warm_step": 2000,
            "env_info": get_env_info(),
            
            # Add BF16 related params
            "use_bf16": False,  # Default to False, will be updated from args
        }
    )

    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    if getattr(params, 'use_xlsr', False):
        # Create XLSR encoder with streaming capabilities built-in
        encoder = XLSREncoder(
            model_name=params.xlsr_model_name,
            decode_chunk_size=params.decode_chunk_size,  # 320ms at 16kHz
            chunk_overlap=params.decode_chunk_size // 2,  # Half of chunk size
            use_attention_sink=True,
            attention_sink_size=params.attention_sink_size,  # 16 frames (paper's optimal)
            frame_duration=params.frame_duration,  # 25ms per frame
            frame_stride=params.frame_stride,    # 20ms stride
            min_chunk_size=2560,   # 160ms at 16kHz (16 frames)
            max_chunk_size=20480,  # 1280ms at 16kHz (128 frames)
            left_context_chunks=params.left_context_chunks  # 1 chunk (paper's optimal)
        )
        # Verify the encoder is properly initialized
        assert isinstance(encoder, XLSREncoder), f"Expected XLSREncoder, got {type(encoder)}"
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


def validate_one_sample(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    sample: dict,
) -> Tuple[str, str, float]:
    """
    Validate a single sample with proper batch handling.
    
    Args:
        params: Model parameters
        model: The transducer model
        sp: SentencePieceProcessor
        sample: A single sample dict with batch dimension 1
        
    Returns:
        (reference, prediction, wer): The ground truth, prediction, and WER
    """
    model.eval()
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Process inputs - ensure batch size is exactly 1
        feature = sample["inputs"].to(device)
        feature_lens = sample["supervisions"]["num_frames"].to(device)
        
        # Verify dimensions
        assert feature.size(0) == 1, f"Expected batch size 1, got {feature.size(0)}"
        assert feature_lens.size(0) == 1, f"Expected lens size 1, got {feature_lens.size(0)}"
        
        # Get reference text
        reference = sample["supervisions"]["text"][0]
        
        # Reset XLSR encoder states to ensure clean inference
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'reset_streaming_state'):
            if isinstance(model, DDP):
                model.module.encoder.reset_streaming_state()
            else:
                model.encoder.reset_streaming_state()
        
        # Try CTC decoding first for comparison using the pretrained model
        try:
            from xlsr_ctc_decoder import PretrainedXLSRCTCDecoder
            
            logging.info("Trying CTC decoding with pretrained model for comparison")
            ctc_decoder = PretrainedXLSRCTCDecoder(
                model_name=params.xlsr_model_name,
                blank_id=params.blank_id
            ).to(device)
            
            # Decode directly from audio using the pretrained CTC model
            ctc_predictions = ctc_decoder.decode_from_audio(feature)
            ctc_prediction = ctc_predictions[0]  # First (only) result
            
            logging.info(f"CTC PREDICTION: {ctc_prediction}")
        except Exception as e:
            logging.error(f"Error during pretrained CTC decoding: {str(e)}")
            ctc_prediction = "CTC_DECODE_ERROR"
                
        # Process with encoder (non-streaming for validation simplicity)
        encoder_out, encoder_out_lens = model.encoder(
            feature, feature_lens, is_pre_training=True
        )
        
        # Project encoder output
        if isinstance(model, DDP):
            encoder_out = model.module.encoder_proj(encoder_out)
        else:
            encoder_out = model.encoder_proj(encoder_out)
        
        # Always use greedy search for validation during early training
        if params.cur_epoch <= 2 or params.batch_idx_train < 500:
            # Use enhanced XLSR-specific greedy search for early training
            logging.info("Using enhanced XLSR greedy search for early validation")
            try:
                prediction_tokens = xlsr_greedy_search_batch(
                    model=model,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    blank_penalty=2.0,  # Stronger blank penalty for early training
                    repetition_penalty=2.0  # Repetition penalty to prevent loops
                )[0]  # Take first (only) result
            except Exception as e:
                logging.error(f"Error during enhanced greedy validation: {str(e)}")
                return reference, "GREEDY_DECODE_ERROR", 100.0
        else:
            # Use beam search with adjusted parameters for better stability
            try:
                logging.info("Using beam search for validation")
                prediction_tokens = beam_search_batch(
                    model=model,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    beam_size=4,
                    blank_penalty=5.0,  # High penalty to prevent blank token dominance
                )[0]  # Take first (only) result
            except Exception as e:
                logging.error(f"Beam search failed, falling back to enhanced greedy: {str(e)}")
                prediction_tokens = xlsr_greedy_search_batch(
                    model=model,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    blank_penalty=2.0
                )[0]
            
        # Convert tokens to text
        if isinstance(prediction_tokens, torch.Tensor):
            prediction_tokens = prediction_tokens.tolist()
            
        # Remove blank tokens from the end
        while prediction_tokens and prediction_tokens[-1] == params.blank_id:
            prediction_tokens.pop()
            
        # Limit prediction length to avoid extremely long outputs
        max_length = 100
        if len(prediction_tokens) > max_length:
            logging.warning(f"Truncating long prediction from {len(prediction_tokens)} to {max_length} tokens")
            prediction_tokens = prediction_tokens[:max_length]
            
        # Convert to text
        prediction = sp.decode(prediction_tokens)
        
        # Calculate WER
        ref_words = reference.split()
        pred_words = prediction.split()
        errors = editdistance.eval(ref_words, pred_words)
        wer = 100.0 * errors / max(1, len(ref_words))
        
        # Also calculate WER for CTC prediction for comparison
        if ctc_prediction != "CTC_DECODE_ERROR":
            ctc_pred_words = ctc_prediction.split()
            ctc_errors = editdistance.eval(ref_words, ctc_pred_words)
            ctc_wer = 100.0 * ctc_errors / max(1, len(ref_words))
            logging.info(f"CTC WER: {ctc_wer:.2f}%")
        
        # Log more detailed token information
        logging.info(f"Number of tokens in prediction: {len(prediction_tokens)}")
        
        return reference, prediction, wer


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    tb_writer: Optional[SummaryWriter] = None,
) -> MetricsTracker:
    """Run minimal validation with proper batch handling."""
    model.eval()
    
    # Create a placeholder metrics tracker
    tot_loss = MetricsTracker()
    tot_loss["frames"] = 1  # Avoid division by zero
    tot_loss["loss"] = 0.0
    
    # Clear GPU memory before validation
    torch.cuda.empty_cache()
    
    # Load and process a random sample
    try:
        # Get a single random sample directly from dataset
        val_dataset = valid_dl.dataset
        if len(val_dataset) == 0:
            logging.warning("Empty validation dataset, skipping validation")
            return tot_loss
        
        # Get a random sample
        rand_idx = random.randint(0, len(val_dataset) - 1)
        single_sample = val_dataset[rand_idx]
        
        # Make sure it has batch dimension
        if 'inputs' in single_sample and single_sample['inputs'].ndim == 1:
            single_sample['inputs'] = single_sample['inputs'].unsqueeze(0)
        
        if 'supervisions' in single_sample and 'num_frames' in single_sample['supervisions']:
            if isinstance(single_sample['supervisions']['num_frames'], int):
                single_sample['supervisions']['num_frames'] = torch.tensor([single_sample['supervisions']['num_frames']])
            elif isinstance(single_sample['supervisions']['num_frames'], torch.Tensor) and single_sample['supervisions']['num_frames'].ndim == 0:
                single_sample['supervisions']['num_frames'] = single_sample['supervisions']['num_frames'].unsqueeze(0)
        
        # Ensure text is in a list
        if 'supervisions' in single_sample and 'text' in single_sample['supervisions']:
            if isinstance(single_sample['supervisions']['text'], str):
                single_sample['supervisions']['text'] = [single_sample['supervisions']['text']]
        
        logging.info(f"Validating with sample {rand_idx} from dataset")
        
        # Validate sample with our clean validation function
        reference, prediction, wer = validate_one_sample(
            params=params,
            model=model,
            sp=sp,
            sample=single_sample
        )
        
        # Display the result
        logging.info("\nRandom validation sample:")
        logging.info("-" * 80)
        logging.info(f"REFERENCE: {reference}")
        logging.info(f"PREDICTION: {prediction}")
        logging.info(f"WER: {wer:.2f}%")
        logging.info("-" * 80)
        
        if tb_writer is not None:
            tb_writer.add_scalar('valid/wer_sample', wer, params.batch_idx_train)
            
    except Exception as e:
        logging.warning(f"Error during minimal validation: {str(e)}")
        logging.warning("Exception details:", exc_info=True)  # Print full stack trace
        
    # Clean up GPU memory
    torch.cuda.empty_cache()
    
    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    sp: spm.SentencePieceProcessor,
    scaler: Optional[torch.amp.GradScaler],
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch following paper's approach:
    1. Pre-training phase: Full sequence processing
    2. Progressive streaming phase: Gradually decrease chunk size
    3. Final streaming phase: Fixed optimal chunk size
    """
    model.train()
    tot_loss = MetricsTracker()
    
    # Determine training phase
    is_pre_training = params.cur_epoch <= params.pretrain_epochs
    
    # Set batch size based on phase
    if isinstance(params.batch_size, dict):
        curr_batch_size = params.batch_size["non_streaming" if is_pre_training else "streaming"]
    else:
        # If batch_size is an integer (from command line args), use it directly
        curr_batch_size = params.batch_size
    
    phase = (
        "Pre-training" if is_pre_training
        else "Progressive streaming" if params.cur_epoch <= params.pretrain_epochs + 5
        else "Final streaming"
    )
    
    logging.info(
        f"Epoch {params.cur_epoch}: {phase} phase, batch_size={curr_batch_size}"
    )
    
    # Special debugging for the first epoch
    if params.cur_epoch == 1 and params.batch_idx_train < 5:
        # Set more verbose logging for first few batches
        logging.getLogger().setLevel(logging.DEBUG)
        
        # Check model parameter norms to detect initialization issues
        logging.info("Checking model parameter norms:")
        encoder_norms = []
        decoder_norms = []
        joiner_norms = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_norm = param.norm().item()
                if 'encoder' in name:
                    encoder_norms.append((name, param_norm))
                elif 'decoder' in name:
                    decoder_norms.append((name, param_norm))
                elif 'joiner' in name:
                    joiner_norms.append((name, param_norm))
                    
                # Log extreme values
                if param_norm > 1000 or param_norm < 1e-6:
                    logging.warning(f"Parameter {name} has extreme norm: {param_norm}")
        
        # Log summary stats
        if encoder_norms:
            enc_avg = sum(n for _, n in encoder_norms) / len(encoder_norms)
            logging.info(f"Encoder average param norm: {enc_avg:.6f}")
        if decoder_norms:
            dec_avg = sum(n for _, n in decoder_norms) / len(decoder_norms)
            logging.info(f"Decoder average param norm: {dec_avg:.6f}")
        if joiner_norms:
            join_avg = sum(n for _, n in joiner_norms) / len(joiner_norms)
            logging.info(f"Joiner average param norm: {join_avg:.6f}")
    
    for batch_idx, batch in enumerate(train_dl):
        try:
            params.batch_idx_train += 1
            optimizer.zero_grad()
            
            # Extra debugging for first few batches
            if params.cur_epoch == 1 and batch_idx < 2:
                logging.debug(f"Batch {batch_idx} input shape: {batch['inputs'].shape}")
                text_samples = batch["supervisions"]["text"][:3]  # First 3 examples
                logging.info(f"Sample texts: {text_samples}")
                
                # Check for any suspicious values in inputs
                inputs = batch["inputs"]
                if torch.isnan(inputs).any():
                    logging.error("NaN values detected in batch inputs!")
                if torch.isinf(inputs).any():
                    logging.error("Inf values detected in batch inputs!")
                    
                # Log input stats
                logging.debug(f"Input min: {inputs.min().item()}, max: {inputs.max().item()}, mean: {inputs.mean().item()}")
            
            # Use compute_loss_with_amp for proper precision handling
            loss, loss_info = compute_loss_with_amp(
                params=params,
                model=model,
                sp=sp,
                batch=batch,
                is_training=True,
                is_pre_training=is_pre_training,
                scaler=scaler
            )
            
            # Handle mixed precision training
            if params.use_fp16 and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 5.0, 2.0)
                scaler.step(optimizer)
                scaler.update()
            elif params.use_bf16:
                # BF16 doesn't need a scaler, just use autocast context
                loss.backward()
                clip_grad_norm_(model.parameters(), 5.0, 2.0)
                optimizer.step()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), 5.0, 2.0)
                optimizer.step()
            
            tot_loss = tot_loss + loss_info
            
            if batch_idx % params.log_interval == 0:
                current_lr = list(optimizer.param_groups)[0]['lr']
                logging.info(
                    f'Epoch {params.cur_epoch}, batch {batch_idx}, '
                    f'batch size {curr_batch_size}, loss {loss:.4f}, '
                    f'current lr {current_lr:.2e}, '
                    f'memory used: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB'
                )
                
                # Reset to info level after early batches
                if params.cur_epoch == 1 and params.batch_idx_train >= 5:
                    logging.getLogger().setLevel(logging.INFO)
                
                if tb_writer is not None:
                    tb_writer.add_scalar('train/loss', loss, params.batch_idx_train)
                    tb_writer.add_scalar('train/simple_loss', loss_info["simple_loss"], params.batch_idx_train)
                    if "pruned_loss" in loss_info:
                        tb_writer.add_scalar('train/pruned_loss', loss_info["pruned_loss"], params.batch_idx_train)
                    tb_writer.add_scalar('train/lr', current_lr, params.batch_idx_train)
                    tb_writer.add_scalar('train/batch_size', curr_batch_size, params.batch_idx_train)
            
            # Run validation less frequently to save memory - validate every 100 batches
            # or only at specific batch indices during early training
            should_validate = (
                batch_idx > 0 and 
                ((batch_idx % 50 == 0) or  # Run validation more frequently
                 (params.cur_epoch <= 2 and batch_idx in [10, 30, 50]))  # More validation points in first epochs
            )
            
            if should_validate and not params.print_diagnostics and rank == 0:  # Only rank 0 does validation
                # Clear memory before validation
                torch.cuda.empty_cache()
                
                logging.info("Running quick validation (single random sample)")
                try:
                    valid_info = compute_validation_loss(
                        params=params,
                        model=model,
                        sp=sp,
                        valid_dl=valid_dl,
                        world_size=world_size,
                        tb_writer=tb_writer,
                    )
                    model.train()  # Switch back to training mode
                    logging.info(f"Validation complete. Memory used: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB")
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.warning(
                            f"OOM during validation, something went very wrong. "
                            f"Memory usage: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB"
                        )
                        torch.cuda.empty_cache()
                        model.train()  # Ensure we're back in training mode
                    else:
                        raise e
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Reduce batch size if OOM
                if curr_batch_size > 1:
                    curr_batch_size = max(1, curr_batch_size // 2)
                    logging.warning(
                        f'OOM error - reducing batch size to {curr_batch_size}. '
                        f'Memory usage: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB'
                    )
                    torch.cuda.empty_cache()
                    continue
                else:
                    logging.error(
                        'OOM error with batch size 1 - cannot reduce further. '
                        f'Memory usage: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB'
                    )
                    raise e
            else:
                raise e
    
    logging.info(f'Mean loss: {tot_loss["loss"] / tot_loss["frames"]:.4f}')
    
    # After pre-training phase, evaluate with different chunk sizes following the paper's approach
    if params.cur_epoch == params.pretrain_epochs and rank == 0:
        logging.info("Pre-training complete. Evaluating streaming configurations...")
        model.eval()
        with torch.no_grad():
            # Evaluate with different chunk sizes (using the paper's settings)
            # The paper uses chunk sizes of 320ms, 640ms, 1280ms, and 2560ms
            streaming_results = {}
            
            for name, size in params.chunk_sizes.items():
                logging.info(f"Evaluating with chunk size {name} ({size} samples)")
                
                metrics = evaluate_streaming(
                    params=params,
                    model=model,
                    valid_dl=valid_dl,
                    chunk_size=size,
                    sp=sp
                )
                
                streaming_results[name] = metrics
                logging.info(f"Chunk size {name}: WER = {metrics['wer']:.2f}%")
                
                if tb_writer is not None:
                    tb_writer.add_scalar(
                        f'eval/wer_{name}',
                        metrics['wer'],
                        params.batch_idx_train
                    )
            
            # Find optimal chunk size based on WER
            best_chunk_size = None
            best_wer = float('inf')
            
            for name, metrics in streaming_results.items():
                if metrics['wer'] < best_wer:
                    best_wer = metrics['wer']
                    best_chunk_size = name
            
            logging.info(f"Best performing chunk size: {best_chunk_size} with WER = {best_wer:.2f}%")
    
    # Save checkpoint at the end of epoch
    save_checkpoint(
        params=params,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=train_dl.sampler,
        scaler=scaler,
        rank=rank,
    )


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
    # We want to keep our validation interval at 500 regardless of dataset
    # if params.full_libri is False:
    #     params.valid_interval = 1600

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

    # Paper uses warmup followed by decay
    def lr_scheduler(step: int, epoch: int) -> float:
        # Warmup phase
        if step < params.warmup_steps:
            return step / params.warmup_steps
        # Decay phase based on steps and epochs
        decay_factor = 0.05 * (params.num_epochs - epoch) / params.num_epochs
        return max(0.05, decay_factor)  # Don't let LR go below 5% of base_lr

    scheduler = Eden(
        optimizer,
        lr_batches=params.lr_batches,
        lr_epochs=params.lr_epochs,
        warmup_batches=params.warmup_steps,
    )

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"])

    if checkpoints and "scheduler" in checkpoints:
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
        
        train_dataset = EstonianASRDataset(params.train_txt, base_path=params.audio_base_path, sp=sp)
        train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )
        
        valid_dataset = EstonianASRDataset(params.val_txt, base_path=params.audio_base_path, sp=sp)
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

    # Update to use new GradScaler API with proper precision type
    precision_type = 'bf16' if params.use_bf16 else ('fp16' if params.use_fp16 else None)
    if precision_type == 'fp16':
        scaler = torch.amp.GradScaler(
            'cuda',
            enabled=True,
            init_scale=1.0
        )
    else:
        scaler = None
        
    if checkpoints and "grad_scaler" in checkpoints and scaler is not None:
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
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=valid_dl,
            sp=sp,
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
            # Set up mixed precision context
            precision_type = 'bf16' if params.use_bf16 else ('fp16' if params.use_fp16 else None)
            if precision_type == 'bf16':
                # BF16 autocast
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                    loss, _ = compute_loss(
                        params=params,
                        model=model,
                        sp=sp,
                        batch=batch,
                        is_training=True,
                        is_pre_training=False
                    )
            else:
                # FP16 or no autocast
                with torch.amp.autocast(device_type='cuda', enabled=params.use_fp16):
                    loss, _ = compute_loss(
                        params=params,
                        model=model,
                        sp=sp,
                        batch=batch,
                        is_training=True,
                        is_pre_training=False
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


def reduce_metrics(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    """Reduce metrics across distributed processes.
    
    Args:
        metrics: Dictionary of metrics to reduce
        device: Device to perform reduction on
        
    Returns:
        Reduced metrics dictionary
    """
    if torch.distributed.is_initialized():
        reduced_metrics = {}
        for k, v in metrics.items():
            tensor = torch.tensor([v], device=device)
            torch.distributed.all_reduce(tensor)
            reduced_metrics[k] = tensor.item() / torch.distributed.get_world_size()
        return reduced_metrics
    return metrics


def find_optimal_chunk_size(
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    sp: spm.SentencePieceProcessor,
) -> int:
    """Benchmarks different chunk sizes and returns optimal one.
    
    Args:
        params: Model parameters
        model: The model to test
        valid_dl: Validation dataloader
        sp: Sentence piece processor
        
    Returns:
        Optimal chunk size in samples
    """
    logging.info("Benchmarking chunk sizes to find optimal configuration...")
    results = {}
    
    # Test a range of chunk sizes
    for name, size in params.chunk_sizes.items():
        start_time = time.time()
        metrics = evaluate_streaming(
            params=params,
            model=model,
            valid_dl=valid_dl,
            chunk_size=size,
            sp=sp
        )
        elapsed = time.time() - start_time
        
        # Calculate efficiency score (lower WER and latency is better)
        efficiency = metrics["wer"] * (metrics["latency"] ** 0.5)
        results[name] = {
            "wer": metrics["wer"],
            "latency": metrics["latency"],
            "efficiency": efficiency,
            "size": size
        }
        
        logging.info(f"Chunk size {name}: WER={metrics['wer']:.2f}%, "
                     f"Latency={metrics['latency']:.3f}s, "
                     f"Efficiency={efficiency:.3f}")
    
    # Find most efficient configuration
    optimal = min(results.items(), key=lambda x: x[1]["efficiency"])
    logging.info(f"Optimal chunk size: {optimal[0]} with efficiency {optimal[1]['efficiency']:.3f}")
    
    return optimal[1]["size"]


def evaluate_streaming(
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    chunk_size: int,
    sp: spm.SentencePieceProcessor,
) -> Dict[str, float]:
    """Evaluate model with streaming inference using the paper's approach."""
    model.eval()
    total_words = 0
    total_errors = 0
    total_samples = 0
    batch_metrics = []
    
    # Reset streaming states at the beginning
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'reset_streaming_state'):
        if isinstance(model, DDP):
            model.module.encoder.reset_streaming_state()
        else:
            model.encoder.reset_streaming_state()
    
    for batch_idx, batch in enumerate(valid_dl):
        try:
            # Process one utterance at a time to avoid batch size issues
            batch_size = batch["inputs"].size(0)
            for idx in range(batch_size):
                with torch.no_grad():
                    # Extract single sample
                    single_sample = {
                        "inputs": batch["inputs"][idx:idx+1],
                        "supervisions": {
                            "num_frames": batch["supervisions"]["num_frames"][idx:idx+1],
                            "text": [batch["supervisions"]["text"][idx]],
                        }
                    }
                    
                    device = next(model.parameters()).device
                    feature = single_sample["inputs"].to(device)
                    feature_lens = single_sample["supervisions"]["num_frames"].to(device)
                    reference = single_sample["supervisions"]["text"][0]
                    
                    # Reset states for each utterance
                    if hasattr(model, 'encoder') and hasattr(model.encoder, 'reset_streaming_state'):
                        if isinstance(model, DDP):
                            model.module.encoder.reset_streaming_state()
                        else:
                            model.encoder.reset_streaming_state()
                    
                    # Process audio in chunks following the paper's approach
                    encoder_out_chunks = []
                    pos = 0
                    
                    # Get buffer and state setup
                    chunk_overlap = chunk_size // 2  # Paper used 50% overlap
                    attention_sink_size = params.attention_sink_size
                    states = None
                    
                    # Initialize variables for attention sink
                    attention_sink = None
                    
                    # Process audio in chunks
                    while pos < feature.size(1):
                        end_pos = min(pos + chunk_size, feature.size(1))
                        current_chunk = feature[:, pos:end_pos]
                        
                        # Add attention sink if available
                        if attention_sink is not None:
                            # Paper's approach: prepend attention sink
                            current_chunk = torch.cat([attention_sink, current_chunk], dim=1)
                        
                        # Process chunk with appropriate precision
                        if params.use_bf16:
                            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                                # Get encoder output for this chunk
                                chunk_out, chunk_lens, states = model.encoder.streaming_forward(
                                    current_chunk,
                                    torch.tensor([current_chunk.size(1)], device=device),
                                    states
                                )
                        else:
                            with torch.amp.autocast(device_type='cuda', enabled=params.use_fp16):
                                # Get encoder output for this chunk
                                chunk_out, chunk_lens, states = model.encoder.streaming_forward(
                                    current_chunk,
                                    torch.tensor([current_chunk.size(1)], device=device),
                                    states
                                )
                        
                        # Update attention sink from end of current output
                        if attention_sink_size > 0:
                            attention_sink = chunk_out[:, -attention_sink_size:]
                        
                        # Store output
                        if pos > 0:
                            # Remove overlap from output for subsequent chunks
                            overlap_frames = chunk_overlap // model.encoder.downsample_factor
                            chunk_out = chunk_out[:, overlap_frames:]
                            
                        encoder_out_chunks.append(chunk_out)
                        
                        # Move position forward with overlap
                        pos = end_pos - chunk_overlap
                    
                    # Concatenate chunks
                    encoder_out = torch.cat(encoder_out_chunks, dim=1)
                    
                    # Calculate output length (after downsampling)
                    encoder_out_lens = ((feature_lens.float() / model.encoder.downsample_factor).floor()).to(torch.int64)
                    encoder_out_lens = torch.maximum(encoder_out_lens, torch.ones_like(encoder_out_lens))
                    
                    # Project encoder output
                    if isinstance(model, DDP):
                        encoder_out = model.module.encoder_proj(encoder_out)
                    else:
                        encoder_out = model.encoder_proj(encoder_out)
                    
                    # Use XLSR-specific greedy search for decoding as it's more robust
                    hyp_tokens = xlsr_greedy_search_batch(
                        model=model,
                        encoder_out=encoder_out,
                        encoder_out_lens=encoder_out_lens,
                        blank_penalty=2.0,  # Stronger blank penalty to prevent blank token dominance
                        repetition_penalty=1.5  # Moderate repetition penalty
                    )[0]
                    
                    # Convert to text
                    if isinstance(hyp_tokens, torch.Tensor):
                        hyp_tokens = hyp_tokens.tolist()
                    
                    # Remove blank tokens
                    while hyp_tokens and hyp_tokens[-1] == params.blank_id:
                        hyp_tokens.pop()
                    
                    prediction = sp.decode(hyp_tokens)
                    
                    # Calculate WER
                    ref_words = reference.split()
                    pred_words = prediction.split()
                    errors = editdistance.eval(ref_words, pred_words)
                    total_errors += errors
                    total_words += len(ref_words)
                    total_samples += 1
                    
                    # Log sample metrics
                    sample_wer = 100.0 * errors / max(1, len(ref_words))
                    batch_metrics.append({
                        'reference': reference,
                        'prediction': prediction, 
                        'wer': sample_wer,
                        'words': len(ref_words),
                        'errors': errors
                    })
                    
                    # Log occasionally
                    if total_samples % 10 == 0:
                        current_wer = 100.0 * total_errors / max(1, total_words)
                        logging.info(f"Processed {total_samples} samples, current WER: {current_wer:.2f}%")
        
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {str(e)}")
            logging.error("Exception details:", exc_info=True)
            continue
    
    # Calculate final metrics
    wer = 100.0 * total_errors / max(1, total_words)
    
    # Display some sample predictions
    logging.info("\nSample predictions:")
    for i, metric in enumerate(batch_metrics[:5]):  # Show first 5 samples
        logging.info(f"Sample {i}:")
        logging.info(f"  Reference: {metric['reference']}")
        logging.info(f"  Prediction: {metric['prediction']}")
        logging.info(f"  WER: {metric['wer']:.2f}%")
    
    # Return metrics
    metrics = {
        "wer": wer,
        "total_samples": total_samples,
        "total_words": total_words,
        "total_errors": total_errors,
        "chunk_size_ms": chunk_size / 16,  # Convert samples to ms
        "latency": chunk_size / 16000.0  # Estimated latency in seconds
    }
    
    return metrics


def compute_loss_with_amp(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    is_pre_training: bool = True,
    scaler: Optional[torch.amp.GradScaler] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """Compute loss with automatic mixed precision.
    
    Args:
        Same as compute_loss with additional scaler parameter
        
    Returns:
        Same as compute_loss
    """
    # Use appropriate precision based on parameters
    if params.use_bf16:
        # Use BF16 precision
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            loss, info = compute_loss(
                params=params,
                model=model,
                sp=sp,
                batch=batch,
                is_training=is_training,
                is_pre_training=is_pre_training,
                chunk_size=chunk_size
            )
    else:
        # Use FP16 or full precision
        with torch.amp.autocast(device_type='cuda', enabled=params.use_fp16):
            loss, info = compute_loss(
                params=params,
                model=model,
                sp=sp,
                batch=batch,
                is_training=is_training,
                is_pre_training=is_pre_training,
                chunk_size=chunk_size
            )
    
    # Return the unscaled loss - scaler.scale will be applied in training loop if needed
    return loss, info


def compute_loss(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    is_pre_training: bool = True,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """Compute loss following paper's approach with progressive chunk sizes.
    
    Args:
        params: Model parameters
        model: The model to train
        sp: Sentence piece processor
        batch: A batch of data
        is_training: Whether this is a training batch
        is_pre_training: Whether this is pre-training phase
        chunk_size: Optional override for chunk size
    
    Returns:
        (loss, MetricsTracker) containing loss and statistics
    """
    device = next(model.parameters()).device
    feature = batch["inputs"].to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)
    
    # Get supervisions
    supervisions = batch["supervisions"]
    texts = supervisions["text"]
    y = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(y).to(device)
    
    if is_pre_training:
        # Pre-training phase: full sequence processing without chunking
        encoder_out, encoder_out_lens = model.encoder(
            x=feature,
            x_lens=feature_lens,
            is_pre_training=True
        )
    else:
        # Determine chunk size based on training phase
        if chunk_size is None:
            if params.cur_epoch <= params.pretrain_epochs + 5:  # 5 epochs transition
                # Progressive chunk size selection
                progress = (params.cur_epoch - params.pretrain_epochs) / 5
                curr_chunk_size = int(
                    params.chunk_sizes["2560ms"] * (1 - progress) + 
                    params.chunk_sizes["320ms"] * progress
                )
            else:
                # Use optimal chunk size from paper
                curr_chunk_size = params.chunk_sizes["320ms"]
        else:
            curr_chunk_size = chunk_size
            
        # Process in chunks with attention sink
        encoder_out = process_streaming_chunks(
            model=model,
            feature=feature,
            chunk_size=curr_chunk_size,
            attention_sink_size=params.attention_sink_size,
            left_context_chunks=params.left_context_chunks
        )
        # Calculate encoder output lengths based on chunk processing
        encoder_out_lens = ((feature_lens.float() / model.encoder.downsample_factor).floor()).to(torch.int64)
        encoder_out_lens = torch.maximum(encoder_out_lens, torch.ones_like(encoder_out_lens))
    
    # Project encoder output if using XLSR
    if hasattr(model, 'encoder_proj'):
        encoder_out = model.encoder_proj(encoder_out)
    
    # Compute transducer loss
    simple_loss, pruned_loss = model(
        x=encoder_out,  # Using encoder output instead of raw audio
        x_lens=encoder_out_lens,
        y=y,
        prune_range=params.prune_range,
        am_scale=params.am_scale,
        lm_scale=params.lm_scale,
    )
    
    # Combine losses according to paper, with adjustments for early training
    if is_pre_training:
        # During early training, focus more on simple loss for better convergence
        if params.batch_idx_train < 500:
            # First 500 batches: focus entirely on simple loss
            loss = simple_loss
        else:
            # Gradually transition to the balanced loss
            loss = simple_loss
    else:
        # Use more balanced loss scale for streaming phase
        simple_scale = params.simple_loss_scale
        loss = simple_scale * simple_loss + (1 - simple_scale) * pruned_loss
    
    assert loss.requires_grad == is_training
    
    info = MetricsTracker()
    info["frames"] = feature.size(0)
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    if not is_pre_training:
        info["pruned_loss"] = pruned_loss.detach().cpu().item()
        if is_training and hasattr(params, "cur_epoch"):
            info["chunk_size"] = curr_chunk_size if not is_pre_training else 0
    
    return loss, info


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