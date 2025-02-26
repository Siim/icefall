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
from lhotse.dataset.dataloading import DataLoader
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
from beam_search import greedy_search_batch  # For compatibility
from xlsr_greedy_search import greedy_search_batch as xlsr_greedy_search_batch  # XLSR-specific implementation

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


def evaluate_streaming(
    params: AttributeDict,
    model: nn.Module,
    valid_dl: torch.utils.data.DataLoader,
    chunk_size: int,
    sp: spm.SentencePieceProcessor,
) -> Dict[str, float]:
    """Evaluate model with streaming inference using greedy search.
    Note: We use greedy search for quick validation during training.
    For actual model evaluation, use beam search with width=4 as per paper.
    """
    model.eval()
    total_words = 0
    total_errors = 0
    total_duration = 0.0
    total_processing_time = 0.0
    total_tokens_predicted = 0
    
    for batch_idx, batch in enumerate(valid_dl):
        try:
            with torch.no_grad():
                feature = batch["inputs"].to(next(model.parameters()).device)
                feature_lens = batch["supervisions"]["num_frames"].to(feature.device)
                texts = batch["supervisions"]["text"]
                
                # Calculate audio duration in seconds
                audio_duration = feature.size(1) / 16000  # 16kHz sampling rate
                total_duration += audio_duration
                
                # Process in chunks with timing
                start_time = time.time()
                
                encoder_out = process_streaming_chunks(
                    model=model,
                    feature=feature,
                    feature_lens=feature_lens,
                    chunk_size=chunk_size,
                    attention_sink_size=params.attention_sink_size,
                    left_context_chunks=params.left_context_chunks
                )
                
                # Use XLSR-specific greedy search for decoding - handles batch size mismatches
                hyp_tokens = xlsr_greedy_search_batch(
                    model=model,
                    encoder_out=encoder_out,
                    encoder_out_lens=feature_lens,
                )
                
                # Measure processing time
                processing_time = time.time() - start_time
                total_processing_time += processing_time
                
                # Convert predictions to text
                hyps = []
                for tokens in hyp_tokens:
                    if isinstance(tokens, torch.Tensor):
                        tokens = tokens.tolist()
                    hyps.append(sp.decode(tokens))
                    total_tokens_predicted += len(tokens)
                
                # Calculate WER
                for hyp, ref in zip(hyps, texts):
                    hyp_words = hyp.split()
                    ref_words = ref.split()
                    total_words += len(ref_words)
                    total_errors += editdistance.eval(hyp_words, ref_words)
                
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {str(e)}")
            continue
            
        if batch_idx % 10 == 0:
            wer = 100.0 * total_errors / max(1, total_words)
            rtf = total_processing_time / max(0.001, total_duration)
            logging.info(f"Batch {batch_idx}, current WER: {wer:.2f}%, RTF: {rtf:.2f}x")
    
    # Calculate final metrics
    wer = 100.0 * total_errors / max(1, total_words)
    
    # Real-time factor (RTF) - lower is better
    rtf = total_processing_time / max(0.001, total_duration)
    
    # Per-token latency in milliseconds
    per_token_latency = 1000 * total_processing_time / max(1, total_tokens_predicted)
    
    # First-token latency estimate (assuming first chunk processing time)
    first_token_latency = 1000 * chunk_size / 16000  # milliseconds
    
    metrics = {
        "wer": wer,
        "rtf": rtf,
        "per_token_latency_ms": per_token_latency,
        "first_token_latency_ms": first_token_latency,
        "chunk_size_ms": chunk_size / 16,  # Convert samples to ms
        "total_processing_time": total_processing_time,
        "total_audio_duration": total_duration,
    }
    
    logging.info(f"Streaming evaluation results:")
    logging.info(f"  WER: {wer:.2f}%")
    logging.info(f"  Real-time factor: {rtf:.2f}x")
    logging.info(f"  Per-token latency: {per_token_latency:.1f} ms")
    logging.info(f"  First-token latency (estimated): {first_token_latency:.1f} ms")
    logging.info(f"  Chunk size: {chunk_size / 16:.1f} ms")
    
    return metrics


def get_random_chunk_size(base_size=8000, variance=0.5, min_size=2560, max_size=20480):
    """Get a random chunk size for multi-chunk training
    
    Args:
        base_size: Base chunk size in samples (default: 8000 for 500ms at 16kHz)
        variance: Variance factor for randomization (default: 0.5)
        min_size: Minimum allowed chunk size (default: 2560 for 160ms at 16kHz)
        max_size: Maximum allowed chunk size (default: 20480 for 1280ms at 16kHz)
    
    Returns:
        Random chunk size in samples
    """
    # Choose a random factor between 1-variance and 1+variance
    factor = 1.0 + (random.random() * 2 - 1) * variance
    # Calculate chunk size and ensure it's within bounds
    chunk_size = int(base_size * factor)
    chunk_size = max(min_size, min(max_size, chunk_size))
    # Ensure chunk size is divisible by 320 (Wav2vec2 downsampling factor)
    chunk_size = (chunk_size // 320) * 320
    return chunk_size


def process_streaming_chunks(
    model: nn.Module,
    feature: torch.Tensor,
    feature_lens: torch.Tensor,
    chunk_size: int,
    attention_sink_size: int,
    left_context_chunks: int,
    multi_chunk_training: bool = False
) -> torch.Tensor:
    """Process input features in streaming mode with chunks following paper's approach.
    
    Args:
        model: The model to use
        feature: Input features (batch, time)
        feature_lens: Lengths of features
        chunk_size: Size of each chunk in samples
        attention_sink_size: Number of frames for attention sink
        left_context_chunks: Number of left context chunks
        multi_chunk_training: Whether to use randomized chunk sizes during training
    
    Returns:
        Encoder output tensor
    """
    device = feature.device
    batch_size = feature.size(0)
    
    # Initialize states and outputs
    states = None
    chunk_outputs = []
    
    # Calculate chunk parameters
    chunk_overlap = chunk_size // 2  # 50% overlap as per paper
    
    # Process each sequence in the batch
    for i in range(batch_size):
        seq_len = feature_lens[i]
        seq = feature[i, :seq_len].unsqueeze(0)  # Add batch dimension back
        
        pos = 0
        seq_outputs = []
        left_context = None
        sink_cache = None
        
        while pos < seq.size(1):
            # Randomize chunk size if multi-chunk training is enabled
            if multi_chunk_training and random.random() > 0.3:  # 70% chance to randomize
                current_chunk_size = get_random_chunk_size(
                    base_size=chunk_size, 
                    variance=0.5
                )
                current_overlap = current_chunk_size // 2
            else:
                current_chunk_size = chunk_size
                current_overlap = chunk_overlap
                
            # Get current chunk boundaries
            end_pos = min(pos + current_chunk_size, seq.size(1))
            chunk = seq[:, pos:end_pos]
            
            # Add left context if available
            if left_context is not None:
                chunk = torch.cat([left_context, chunk], dim=1)
            
            # Add attention sink if available
            if sink_cache is not None:
                chunk = torch.cat([sink_cache, chunk], dim=1)
            
            # Process chunk through encoder
            chunk_out, chunk_lens, new_states = model.encoder.streaming_forward(
                chunk,
                torch.tensor([chunk.size(1)], device=device),
                states
            )
            
            # Verify chunk_out has valid dimensions and is not empty
            if chunk_out.size(1) == 0:
                # If we got an empty output, use a minimal fallback
                # Create a small non-zero output based on chunk size
                min_frames = max(1, int(chunk.size(1) / model.encoder.downsample_factor))
                logging.warning(f"Empty encoder output detected. Using fallback with {min_frames} frames.")
                fallback_dim = model.encoder.output_dim
                chunk_out = torch.zeros((1, min_frames, fallback_dim), device=device)
            
            # Update cache for next iteration
            if attention_sink_size > 0:
                # Cache end of current chunk for attention sink
                sink_size = attention_sink_size * model.encoder.downsample_factor
                sink_cache = chunk[:, -sink_size:].detach().clone()
                
            # Save left context for next chunk
            if left_context_chunks > 0:
                left_context = chunk[:, -current_overlap:].detach().clone()
            
            # Add to outputs only if not empty
            if chunk_out.size(1) > 0:
                seq_outputs.append(chunk_out)
            else:
                logging.warning(f"Skipping empty chunk at position {pos}")
                
            # Move to next position with overlap
            pos = end_pos - current_overlap
        
        # Concatenate all outputs for this sequence if not empty
        if len(seq_outputs) > 0:
            seq_encoder_out = torch.cat(seq_outputs, dim=1)
            chunk_outputs.append(seq_encoder_out)
        else:
            # Create fallback output with minimal dimension
            logging.warning(f"No valid chunks for sequence {i}. Using fallback output.")
            fallback_dim = model.encoder.output_dim
            seq_encoder_out = torch.zeros((1, 1, fallback_dim), device=device)
            chunk_outputs.append(seq_encoder_out)
    
    # Return batch of outputs - check if we have any valid outputs
    if len(chunk_outputs) > 0:
        return torch.cat(chunk_outputs, dim=0)
    else:
        # Emergency fallback with minimum size tensor
        logging.error("No valid outputs from any sequence. Using emergency fallback.")
        fallback_dim = model.encoder.output_dim
        return torch.zeros((batch_size, 1, fallback_dim), device=device)


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
            feature_lens=feature_lens,
            chunk_size=curr_chunk_size,
            attention_sink_size=params.attention_sink_size,
            left_context_chunks=params.left_context_chunks,
            multi_chunk_training=False
        )
        # Calculate encoder output lengths based on chunk processing
        encoder_out_lens = ((feature_lens.float() / model.encoder.downsample_factor).floor()).to(torch.int64)
        encoder_out_lens = torch.maximum(encoder_out_lens, torch.ones_like(encoder_out_lens))
    
    # Project encoder output if using XLSR
    if hasattr(model, 'encoder_proj'):
        encoder_out = model.encoder_proj(encoder_out)
    
    # Compute transducer loss
    # Important: We provide the encoder output directly to the model
    # instead of the raw audio, as XLSR encoder has already processed it
    simple_loss, pruned_loss = model(
        x=encoder_out,  # Using encoder output instead of raw audio
        x_lens=encoder_out_lens,
        y=y,
        prune_range=params.prune_range,
        am_scale=params.am_scale,
        lm_scale=params.lm_scale,
    )
    
    # Combine losses according to paper
    if is_pre_training:
        # During pre-training, use only simple loss for better convergence
        loss = simple_loss
    else:
        # During streaming, combine both losses with scaling
        loss = params.simple_loss_scale * simple_loss + (1 - params.simple_loss_scale) * pruned_loss
    
    assert loss.requires_grad == is_training
    
    info = MetricsTracker()
    info["frames"] = feature.size(0)
    info["loss"] = loss.detach().cpu().item()
    if not is_pre_training:
        info["simple_loss"] = simple_loss.detach().cpu().item()
        info["pruned_loss"] = pruned_loss.detach().cpu().item()
        if is_training and hasattr(params, "cur_epoch"):
            info["chunk_size"] = curr_chunk_size if not is_pre_training else 0
    
    return loss, info


def compute_loss_with_amp(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    scaler: Optional[torch.amp.GradScaler],
    is_pre_training: bool = True,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """Compute loss with automatic mixed precision.
    
    Args:
        Same as compute_loss with additional scaler
        
    Returns:
        Same as compute_loss
    """
    with torch.cuda.amp.autocast(enabled=params.use_fp16):
        loss, info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=is_training,
            is_pre_training=is_pre_training,
            chunk_size=chunk_size
        )
    
    # Scale loss for mixed precision training
    if is_training and params.use_fp16 and scaler is not None:
        return loss, info  # Return unscaled loss - scaler.scale will be applied in training loop
    
    return loss, info


def decode_one_batch_hyps(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
) -> Tuple[List[str], List[str]]:
    """Get hypotheses and predictions for one batch.
    
    Args:
        params: Model parameters
        model: The model to use for decoding
        sp: SentencePieceProcessor for converting ids to text
        batch: A batch of data
        
    Returns:
        (hyps, preds): Lists of ground truth and predicted texts
    """
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
            pad=(0, 0, 0, right_context),
            value=LOG_EPS,
        )
        
        # Get encoder output
        encoder_out, encoder_out_lens = model.encoder(feature_pad, feature_lens_pad)
        
        # Project encoder output
        if isinstance(model, DDP):
            encoder_out = model.module.encoder_proj(encoder_out)
        else:
            encoder_out = model.encoder_proj(encoder_out)
        
        # Use XLSR-specific greedy search for decoding - handles batch size mismatches
        hyp_tokens = xlsr_greedy_search_batch(
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
            
        # Remove any padding or special tokens
        while pred_tokens and pred_tokens[-1] == params.blank_id:
            pred_tokens.pop()
            
        # Convert to text
        pred = sp.decode(pred_tokens)
        preds.append(pred)
    
    return hyps, preds


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
    tb_writer: Optional[SummaryWriter] = None,
) -> MetricsTracker:
    """Run minimal validation by randomly sampling a single example."""
    model.eval()
    
    # Create a placeholder metrics tracker
    tot_loss = MetricsTracker()
    tot_loss["frames"] = 1  # Avoid division by zero
    tot_loss["loss"] = 0.0
    
    # Clear GPU memory before validation
    torch.cuda.empty_cache()
    
    # Pick a random batch from validation set
    try:
        # Convert dataloader to list of batches first to enable random selection
        # But limit to first 10 batches to save memory
        val_batches = []
        for i, batch in enumerate(valid_dl):
            if i >= 10:  # Only consider first 10 batches
                break
            val_batches.append(batch)
            
        if not val_batches:
            logging.warning("No validation batches available, skipping validation")
            return tot_loss
            
        # Randomly select a batch
        random_batch = random.choice(val_batches)
        batch_size = random_batch["inputs"].size(0)
        
        # Handle case where batch size might be 1
        if batch_size == 1:
            random_idx = 0
        else:
            random_idx = random.randint(0, batch_size - 1)
        
        logging.info(f"Validating with sample from batch with size {batch_size}, using index {random_idx}")
        
        # Extract just the single sample
        single_sample = {
            "inputs": random_batch["inputs"][random_idx:random_idx+1],
            "supervisions": {
                "num_frames": random_batch["supervisions"]["num_frames"][random_idx:random_idx+1],
                "text": [random_batch["supervisions"]["text"][random_idx]],
            }
        }
        
        with torch.no_grad():
            # Get hypothesis and prediction
            hyps, preds = decode_one_batch_hyps(params, model, sp, single_sample)
            
            # Make sure we have results before accessing them
            if len(hyps) > 0 and len(preds) > 0:
                # Calculate WER for this single sample
                hyp = hyps[0]
                pred = preds[0]
                hyp_words = hyp.split()
                pred_words = pred.split()
                errors = editdistance.eval(hyp_words, pred_words)
                wer = 100.0 * errors / max(1, len(hyp_words))
                
                # Display the result
                logging.info("\nRandom validation sample:")
                logging.info("-" * 80)
                logging.info(f"REFERENCE: {hyp}")
                logging.info(f"PREDICTION: {pred}")
                logging.info(f"WER: {wer:.2f}%")
                logging.info(f"Word errors: {errors}/{len(hyp_words)}")
                logging.info("-" * 80)
                
                if tb_writer is not None:
                    tb_writer.add_scalar('valid/wer_sample', wer, params.batch_idx_train)
            else:
                logging.warning("No results returned from decode_one_batch_hyps")
                
            # Free memory
            del single_sample
            if 'hyps' in locals(): del hyps
            if 'preds' in locals(): del preds
            torch.cuda.empty_cache()
            
    except Exception as e:
        logging.warning(f"Error during minimal validation: {str(e)}")
        logging.warning("Exception details:", exc_info=True)  # Print full stack trace
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
    
    for batch_idx, batch in enumerate(train_dl):
        try:
            params.batch_idx_train += 1
            optimizer.zero_grad()
            
            # Use compute_loss_with_amp for proper fp16 handling
            loss, loss_info = compute_loss_with_amp(
                params=params,
                model=model,
                sp=sp,
                batch=batch,
                is_training=True,
                scaler=scaler
            )
            
            # Handle FP16 training with scaler
            if params.use_fp16 and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 5.0, 2.0)
                scaler.step(optimizer)
                scaler.update()
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
                
                if tb_writer is not None:
                    tb_writer.add_scalar('train/loss', loss, params.batch_idx_train)
                    tb_writer.add_scalar('train/lr', current_lr, params.batch_idx_train)
                    tb_writer.add_scalar('train/batch_size', curr_batch_size, params.batch_idx_train)
                    
                    # Add streaming specific metrics
                    if not is_pre_training:
                        tb_writer.add_scalar(
                            'train/chunk_size',
                            loss_info["chunk_size"],
                            params.batch_idx_train
                        )
                        
                        # Compare streaming vs non-streaming every 10 log intervals
                        if batch_idx % (params.log_interval * 10) == 0:
                            device = next(model.parameters()).device
                            with torch.no_grad():
                                # Get non-streaming output for comparison
                                non_streaming_out, _ = model.encoder(
                                    x=batch["inputs"].to(device),
                                    x_lens=batch["supervisions"]["num_frames"].to(device),
                                    is_pre_training=True
                                )
                                
                                # Get streaming output
                                streaming_out = process_streaming_chunks(
                                    model=model,
                                    feature=batch["inputs"].to(device),
                                    feature_lens=batch["supervisions"]["num_frames"].to(device),
                                    chunk_size=params.chunk_sizes["320ms"],
                                    attention_sink_size=params.attention_sink_size,
                                    left_context_chunks=params.left_context_chunks,
                                    multi_chunk_training=False
                                )
                                
                                # Compare outputs
                                output_diff = (streaming_out - non_streaming_out).abs()
                                tb_writer.add_scalar(
                                    'train/streaming_max_diff',
                                    output_diff.max().item(),
                                    params.batch_idx_train
                                )
                                tb_writer.add_scalar(
                                    'train/streaming_mean_diff',
                                    output_diff.mean().item(),
                                    params.batch_idx_train
                                )
                    
            # Run validation less frequently to save memory - validate every 100 batches
            # or only at specific batch indices during early training
            should_validate = (
                batch_idx > 0 and 
                ((batch_idx % 50 == 0) or  # Run validation more frequently
                 (params.cur_epoch <= 2 and batch_idx in [10, 30, 50]))  # More validation points in first epochs
            )
            
            if should_validate and not params.print_diagnostics:
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
    
    # After pre-training phase, evaluate with different chunk sizes
    if params.cur_epoch == params.pretrain_epochs:
        logging.info("Pre-training complete. Evaluating streaming configurations...")
        model.eval()
        with torch.no_grad():
            for name, size in params.chunk_sizes.items():
                metrics = evaluate_streaming(
                    params=params,
                    model=model,
                    valid_dl=valid_dl,
                    chunk_size=size,
                    sp=sp
                )
                logging.info(f"Chunk size {name}: WER = {metrics['wer']:.2f}%")
                if tb_writer is not None:
                    tb_writer.add_scalar(
                        f'eval/wer_{name}',
                        metrics['wer'],
                        params.batch_idx_train
                    )
    
    save_checkpoint(
        params=params,
        model=model,
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


def run(rank: int, world_size: int, args: argparse.Namespace):
    """
    Args:
      rank: The rank of the current process in the distributed training.
      world_size: Number of processes participating in the training.
      args: Command-line arguments parsed by argparse.
    """
    fix_random_seed(args.seed)
    
    if world_size > 1:
        setup_dist(rank=rank, world_size=world_size, master_port=args.master_port)
    
    setup_logger(f"{args.exp_dir}/log/log-train")
    logging.info("Training started")
    logging.info(f"Rank: {rank}/{world_size}")
    
    # Display training parameters as key-value pairs
    params = get_params()
    for key, value in vars(args).items():
        params[key] = value
    
    # Update params with necessary values
    params.update({
        "env_info": get_env_info(),
        "start_epoch": args.start_epoch, 
        "num_epochs": args.num_epochs,
        "start_batch": args.start_batch,
        "use_fp16": args.use_fp16,
        "exp_dir": args.exp_dir,
        "tensorboard": args.tensorboard,
        "dataset": args.dataset,
        "use_xlsr": args.use_xlsr,
        "xlsr_model_name": args.xlsr_model_name,
        "streaming_start_epoch": args.streaming_start_epoch,
        "pre_train_epochs": args.pre_train_epochs,
        "pre_train_lr": args.pre_train_lr,
        "decode_chunk_size": args.decode_chunk_size,
        "attention_sink_size": args.attention_sink_size,
        "left_context_chunks": args.left_context_chunks,
    })
    
    if params.tensorboard and rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None
    
    # Save key configurations to a file
    if rank == 0:
        with open(f"{params.exp_dir}/configs.yml", "w") as file:
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    file.write(f"{key}: {value}\n")
            for key, value in vars(args).items():
                if isinstance(value, (str, int, float, bool)):
                    file.write(f"{key}: {value}\n")
    
    logging.info(f"Loading BPE model: {args.bpe_model}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    
    # <blankid> is defined in the spm model
    params.blank_id = sp.piece_to_id("<blankid>")
    params.vocab_size = sp.get_piece_size()
    
    logging.info(f"Vocab size: {params.vocab_size}")
    logging.info(f"Blank ID: {params.blank_id}")
    
    # Initialize/Load dataset
    if args.dataset == "librispeech":
        if not torch.cuda.is_available():
            logging.warning("No GPU detected, using CPU for training!")
            
        dm = LibriSpeechAsrDataModule(args)
        train_cuts = dm.train_cuts()
        
        # For debug
        if args.full_libri == 1:
            logging.info("Using full Librispeech")
        else:
            # Only use 100h training data
            logging.info("Using Librispeech train-clean-100")
            train_cuts = train_cuts.filter(lambda cut: cut.recording.id.startswith("train-clean-100"))
        
        if args.enable_musan:
            # Enable MUSAN augmentation
            logging.info("Enable MUSAN augmentation")
            train_cuts = load_manifest_lazy(
                    Path("data/manifests/librispeech_cuts_musan.jsonl.gz")
            )  # noqa
        
        # Sample up to args.max_duration seconds
        if args.max_duration > 0:
            logging.info(
                f"Sample up to {args.max_duration} seconds from each cut in the training set"
            )
            train_cuts = train_cuts.cut_into_windows(
                duration=args.max_duration
            ).cut_into_windows(duration=args.max_duration)
        
        def remove_short_and_long_utt(c: Cut):
            # Remove utterances with duration less than 1 second or more than 20 seconds
            return 1.0 <= c.duration <= 20.0
            
        train_cuts = train_cuts.filter(remove_short_and_long_utt)
        
        # Create validation dataloaders
        valid_cuts = dm.dev_cuts()
        test_cuts = dm.test_cuts()
    elif args.dataset == "estonian":
        # Set up Estonian dataset from files
        from estonian_dataset import EstonianDataModule, EstonianCutSet
        
        # Create Estonian datamodule
        estonia_dm = EstonianDataModule(
            train_list=args.train_txt,
            val_list=args.val_txt,
            audio_base_path=args.audio_base_path
        )
        
        # Get train and validation data
        train_cuts = estonia_dm.train_cuts()
        valid_cuts = estonia_dm.val_cuts()
        test_cuts = valid_cuts  # Use validation data as test since we don't have separate test set
        
        logging.info(f"Estonian dataset loaded with {len(train_cuts)} training and {len(valid_cuts)} validation samples")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Create training sampler
    train_sampler = CutSampler(
        cuts=train_cuts,
        max_duration=params.max_duration["non_streaming"] if isinstance(params.max_duration, dict) else params.max_duration,
        shuffle=True,
    )
    
    logging.info(f"Number of training samples: {len(train_cuts)}")
    logging.info(f"Number of validation samples: {len(valid_cuts)}")
    
    # Create training dataloader
    train_dl = DataLoader(
        EstonianCutSet(train_cuts) if args.dataset == "estonian" else train_cuts,
        batch_size=None,
        sampler=train_sampler,
        num_workers=4,
        persistent_workers=True,
    )
    
    # Create validation dataloader with a smaller batch
    valid_sampler = CutSampler(
        cuts=valid_cuts,
        max_duration=params.max_duration["non_streaming"] if isinstance(params.max_duration, dict) else params.max_duration,
        shuffle=False,
    )
    
    valid_dl = DataLoader(
        EstonianCutSet(valid_cuts) if args.dataset == "estonian" else valid_cuts,
        batch_size=None,
        sampler=valid_sampler,
        num_workers=1,
        persistent_workers=False,
    )
    
    # Create model, optimizer, scheduler, etc.
    model = get_transducer_model(params)
    
    if params.print_diagnostics:
        diagnostic = diagnostics.attach_diagnostics(model)
    
    if params.inf_check:
        register_inf_check_hooks(model)
    
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    optimizer = ScaledAdam(
        model.parameters(),
        lr=params.base_lr,
        clipping_scale=2.0,
        parameters_names=list(name for name, _ in model.named_parameters()),
    )
    
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)
    
    scaler = GradScaler(enabled=params.use_fp16)
    
    if params.start_batch > 0 or params.start_epoch > 1:
        load_checkpoint_if_available(
            params=params,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    
    # If in early training, run pessimistic OOM check
    if params.start_epoch == 1 and params.start_batch == 0:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            sp=sp,
            params=params,
        )
    
    # Main training loop
    for epoch in range(params.start_epoch, params.num_epochs + 1):
        if params.start_batch == 0:
            set_batch_count(model, 0)
            fix_random_seed(params.seed + epoch - 1)
        
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
    
    # Final cleanup
    logging.info("Training finished")
    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


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