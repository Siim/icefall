#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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
(1) greedy search
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method greedy_search

(2) beam search (not recommended)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method beam_search \
    --beam-size 4

(3) modified beam search
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method modified_beam_search \
    --beam-size 4

(4) fast beam search (one best)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method fast_beam_search \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64

(5) fast beam search (nbest)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method fast_beam_search_nbest \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(6) fast beam search (nbest oracle WER)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method fast_beam_search_nbest_oracle \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64 \
    --num-paths 200 \
    --nbest-scale 0.5

(7) fast beam search (with LG)
./pruned_transducer_stateless7_streaming/decode.py \
    --epoch 28 \
    --avg 15 \
    --exp-dir ./pruned_transducer_stateless7_streaming/exp \
    --max-duration 600 \
    --decode-chunk-len 32 \
    --decoding-method fast_beam_search_nbest_LG \
    --beam 20.0 \
    --max-contexts 8 \
    --max-states 64
"""


import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
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
from train import add_model_arguments, get_params, get_transducer_model

from icefall import LmScorer, NgramLm
from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

LOG_EPS = math.log(1e-10)


class StreamingEncoder:
    """Streaming encoder wrapper for inference"""
    def __init__(
        self, 
        encoder: nn.Module, 
        chunk_size: int = 5120,  # 320ms at 16kHz (paper's best performing)
        chunk_overlap: int = None,  # Will be set to chunk_size // 2
        use_attention_sink: bool = True,
        attention_sink_size: int = 16,  # Paper's optimal setting
        frame_duration: float = 0.025,  # 25ms per frame
        frame_stride: float = 0.020,  # 20ms stride
        min_chunk_size: int = 2560,  # 160ms at 16kHz (16 frames)
        max_chunk_size: int = 20480,  # 1280ms at 16kHz (128 frames)
        left_context_chunks: int = 1,  # Paper's optimal setting
    ):
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else chunk_size // 2
        self.use_attention_sink = use_attention_sink
        self.attention_sink_size = attention_sink_size
        self.frame_duration = frame_duration
        self.frame_stride = frame_stride
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.left_context_chunks = left_context_chunks
        self.reset()

    def reset(self):
        """Reset all streaming state variables"""
        if hasattr(self.encoder, 'reset_streaming_state'):
            self.encoder.reset_streaming_state()
        self.cached_features = None
        self.cached_len = 0
        self.current_chunk_size = self.chunk_size
        self.last_chunk_latency = 0
        self.streaming_state = [None, None, None]  # Updated to [left_context, audio_sink, last_chunk_output]
        self.attention_sink_cache = None
        self.context_cache = None
        self.left_context_buffer = []  # Store left context chunks

    def forward(self, x: torch.Tensor, x_lens: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Streaming forward pass with proper context and attention sink handling
        Args:
            x: Input tensor (batch=1, time)
            x_lens: Length of input (optional in streaming mode)
        Returns:
            (output, output_lens)
        """
        assert x.size(0) == 1, "Streaming only supports batch size 1"
        
        # Handle cached features if any
        if self.cached_features is not None:
            x = torch.cat([self.cached_features, x], dim=1)
            self.cached_features = None

        current_offset = 0
        outputs_list = []
        
        while current_offset < x.size(1):
            chunk_start_time = time.time()
            
            # Use current chunk size
            end_idx = min(current_offset + self.current_chunk_size, x.size(1))
            chunk = x[:, current_offset:end_idx]
            
            # Add left context if available
            if self.left_context_buffer:
                context = self.left_context_buffer[-self.left_context_chunks:]
                chunk = torch.cat([*context, chunk], dim=1)
            
            # Add attention sink if enabled
            if self.use_attention_sink and self.attention_sink_cache is not None:
                chunk = torch.cat([self.attention_sink_cache, chunk], dim=1)
            
            # Process chunk
            chunk_len = torch.tensor([chunk.shape[1]], dtype=torch.int32)
            
            # Check if the encoder has streaming_forward method (for XLSR)
            if hasattr(self.encoder, 'streaming_forward'):
                # Use streaming_forward with state for XLSR encoder
                chunk_out, chunk_lens, updated_state = self.encoder.streaming_forward(
                    chunk, chunk_len, self.streaming_state
                )
                # Update streaming state with the new state
                self.streaming_state = updated_state
            else:
                # Fallback to regular forward for other encoders
                chunk_out, chunk_lens = self.encoder(chunk, chunk_len)
            
            # Update attention sink cache
            if self.use_attention_sink:
                sink_size = self.attention_sink_size * self.encoder.downsample_factor
                self.attention_sink_cache = chunk[:, -sink_size:]
            
            # Update left context buffer
            self.left_context_buffer.append(chunk.clone())
            if len(self.left_context_buffer) > self.left_context_chunks:
                self.left_context_buffer.pop(0)
            
            # If not the last chunk, cache overlap portion
            if end_idx < x.size(1):
                self.cached_features = x[:, end_idx - self.chunk_overlap:end_idx]
                # Only keep non-overlapping portion of output
                overlap_frames = self.chunk_overlap // self.encoder.downsample_factor
                chunk_out = chunk_out[:, :-overlap_frames]
            
            outputs_list.append(chunk_out)
            current_offset = end_idx - self.chunk_overlap
            
            # Measure chunk latency and adjust size if needed
            chunk_latency = time.time() - chunk_start_time
            self.last_chunk_latency = chunk_latency

        # Concatenate all chunks
        if outputs_list:
            outputs = torch.cat(outputs_list, dim=1)
            output_lens = torch.tensor([outputs.size(1)], dtype=torch.int32) if x_lens is not None else None
            return outputs, output_lens
        else:
            return None, None

def get_encoder_model(params: AttributeDict) -> nn.Module:
    if getattr(params, 'use_xlsr', False):
        from xlsr_encoder import XLSREncoder
        # Create XLSR encoder with streaming capabilities built-in
        encoder = XLSREncoder(
            model_name=params.xlsr_model_name,
            decode_chunk_size=params.decode_chunk_len,  # Now using decode_chunk_size consistently
            chunk_overlap=params.decode_chunk_len // 2,
            adaptive_chunk_size=getattr(params, 'adaptive_chunk_size', True),
            min_chunk_size=getattr(params, 'min_chunk_size', 4000),
            max_chunk_size=getattr(params, 'max_chunk_size', 12000),
            latency_tolerance=getattr(params, 'latency_tolerance', 0.1)
        )
        return encoder
    else:
        # Original Zipformer code...
        pass


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=9,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=False,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7_streaming/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - fast_beam_search
          - fast_beam_search_nbest
          - fast_beam_search_nbest_oracle
          - fast_beam_search_nbest_LG
        If you use fast_beam_search_nbest_LG, you have to specify
        `--lang-dir`, which should contain `LG.pt`.
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search,
        fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--use-shallow-fusion",
        type=str2bool,
        default=False,
        help="""Use neural network LM for shallow fusion.
        If you want to use LODR, you will also need to set this to true
        """,
    )

    parser.add_argument(
        "--lm-type",
        type=str,
        default="rnn",
        help="Type of NN lm",
        choices=["rnn", "transformer"],
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.3,
        help="""The scale of the neural network LM
        Used only when `--use-shallow-fusion` is set to True.
        """,
    )

    parser.add_argument(
        "--tokens-ngram",
        type=int,
        default=2,
        help="""The order of the ngram lm.
        """,
    )

    parser.add_argument(
        "--backoff-id",
        type=int,
        default=500,
        help="ID of the backoff symbol in the ngram LM",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech",
        choices=["librispeech", "estonian"],
        help="Dataset to use for decoding (librispeech or estonian)",
    )

    parser.add_argument(
        "--test-txt",
        type=str,
        default="Data/test.txt",
        help="Path to test text file for Estonian dataset",
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
        default=4,
        help="Batch size for decoding (used only with Estonian dataset)",
    )

    add_model_arguments(parser)

    return parser


def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict."""
    device = model.device
    feature = batch["inputs"].to(device)
    feature_lens = batch["supervisions"]["num_frames"].to(device)
    
    # Add right context for streaming
    feature_lens += 30
    feature = torch.nn.functional.pad(
        feature,
        pad=(0, 0, 0, 30),
        value=LOG_EPS,
    )
    
    # Get encoder output
    encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens)
    
    hyps = []
    
    # Use Estonian FSA-based decoding if available
    if params.dataset == "estonian" and hasattr(model, "decoding_graph"):
        from estonian_decoder import fast_beam_search_one_best
        hyp_tokens = fast_beam_search_one_best(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_states=params.max_states,
            max_contexts=params.max_contexts,
            decoding_graph=model.decoding_graph.to(device)
        )
        for hyp in hyp_tokens:
            hyps.append([word_table[i] for i in hyp])
    else:
        # Original decoding methods
        if params.decoding_method == "fast_beam_search":
            hyp_tokens = fast_beam_search_one_best(
                model=model,
                decoding_graph=decoding_graph,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=params.beam,
                max_contexts=params.max_contexts,
                max_states=params.max_states,
            )
            for hyp in sp.decode(hyp_tokens):
                hyps.append(hyp.split())
        elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
            hyp_tokens = greedy_search_batch(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
            )
            for hyp in sp.decode(hyp_tokens):
                hyps.append(hyp.split())
        elif params.decoding_method == "modified_beam_search":
            hyp_tokens = modified_beam_search(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=params.beam_size,
            )
            for hyp in sp.decode(hyp_tokens):
                hyps.append(hyp.split())
        else:
            batch_size = encoder_out.size(0)
            for i in range(batch_size):
                encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
                if params.decoding_method == "greedy_search":
                    hyp = greedy_search(
                        model=model,
                        encoder_out=encoder_out_i,
                        max_sym_per_frame=params.max_sym_per_frame,
                    )
                elif params.decoding_method == "beam_search":
                    hyp = beam_search(
                        model=model,
                        encoder_out=encoder_out_i,
                        beam=params.beam_size,
                    )
                else:
                    raise ValueError(f"Unsupported decoding method: {params.decoding_method}")
                hyps.append(sp.decode(hyp).split())
    
    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    elif "fast_beam_search" in params.decoding_method:
        key = f"beam_{params.beam}_"
        key += f"max_contexts_{params.max_contexts}_"
        key += f"max_states_{params.max_states}"
        if "nbest" in params.decoding_method:
            key += f"_num_paths_{params.num_paths}_"
            key += f"nbest_scale_{params.nbest_scale}"
            if "LG" in params.decoding_method:
                key += f"_ngram_lm_scale_{params.ngram_lm_scale}"
        return {key: hyps}
    else:
        return {f"beam_size_{params.beam_size}": hyps}


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    LM: Optional[LmScorer] = None,
    ngram_lm=None,
    ngram_lm_scale: float = 0.0,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    """Decode dataset.

    Args:
      dl:
        PyTorch's dataloader containing the dataset to decode.
      params:
        It is returned by :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
      ngram_lm:
        A n-gram LM to be used for LODR.
    Returns:
      Return a dict, whose key may be "greedy_search" if greedy search
      is used, or it may be "beam_7" if beam size of 7 is used.
      Its value is a list of tuples. Each tuple contains two elements:
      The first is the reference transcript, and the second is the
      predicted result.
    """
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 50
    else:
        log_interval = 20

    results = defaultdict(list)
    for batch_idx, batch in enumerate(dl):
        texts = batch["supervisions"]["text"]
        
        # Handle different dataset formats
        if params.dataset == "estonian":
            # For Estonian dataset, use the audio file paths as IDs
            cut_ids = [path.split('/')[-1].replace('.wav', '') for path in batch["supervisions"]["audio_paths"]]
        else:
            # For Librispeech dataset, use the cut IDs
            cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
            word_table=word_table,
            batch=batch,
        )

        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = params.res_dir / f"recogs-{test_set_name}-{params.suffix}.txt"
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = params.res_dir / f"errs-{test_set_name}-{params.suffix}.txt"
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = params.res_dir / f"wer-summary-{test_set_name}-{params.suffix}.txt"
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)


@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    LmScorer.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "fast_beam_search",
        "fast_beam_search_nbest",
        "fast_beam_search_nbest_LG",
        "fast_beam_search_nbest_oracle",
        "modified_beam_search",
        "modified_beam_search_LODR",
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_lm_rescore",
        "modified_beam_search_lm_rescore_LODR",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    params.suffix += f"-streaming-chunk-size-{params.decode_chunk_len}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
        if "nbest" in params.decoding_method:
            params.suffix += f"-nbest-scale-{params.nbest_scale}"
            params.suffix += f"-num-paths-{params.num_paths}"
            if "LG" in params.decoding_method:
                params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_shallow_fusion:
        params.suffix += f"-{params.lm_type}-lm-scale-{params.lm_scale}"

        if "LODR" in params.decoding_method:
            params.suffix += (
                f"-LODR-{params.tokens_ngram}gram-scale-{params.ngram_lm_scale}"
            )

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)
    
    # Check chunk size based on encoder type
    if getattr(params, 'use_xlsr', False):
        # For XLSR, chunk size is in samples
        assert hasattr(model.encoder, 'decode_chunk_size'), "XLSR encoder missing decode_chunk_size attribute"
        assert model.encoder.decode_chunk_size == params.decode_chunk_len, (
            f"XLSR chunk size mismatch: encoder={model.encoder.decode_chunk_size}, "
            f"params={params.decode_chunk_len}"
        )
    else:
        # For Zipformer, chunk size is in frames
        assert model.encoder.decode_chunk_size == params.decode_chunk_len // 2, (
            model.encoder.decode_chunk_size,
            params.decode_chunk_len,
        )

    if not params.use_averaged_model:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
        elif params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        if params.iter > 0:
            filenames = find_checkpoints(params.exp_dir, iteration=-params.iter)[
                : params.avg + 1
            ]
            if len(filenames) == 0:
                raise ValueError(
                    f"No checkpoints found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            elif len(filenames) < params.avg + 1:
                raise ValueError(
                    f"Not enough checkpoints ({len(filenames)}) found for"
                    f" --iter {params.iter}, --avg {params.avg}"
                )
            filename_start = filenames[-1]
            filename_end = filenames[0]
            logging.info(
                "Calculating the averaged model over iteration checkpoints"
                f" from {filename_start} (excluded) to {filename_end}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )
        else:
            assert params.avg > 0, params.avg
            start = params.epoch - params.avg
            assert start >= 1, start
            filename_start = f"{params.exp_dir}/epoch-{start}.pt"
            filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
            logging.info(
                f"Calculating the averaged model over epoch range from "
                f"{start} (excluded) to {params.epoch}"
            )
            model.to(device)
            model.load_state_dict(
                average_checkpoints_with_averaged_model(
                    filename_start=filename_start,
                    filename_end=filename_end,
                    device=device,
                )
            )

    model.to(device)
    model.eval()

    # only load the neural network LM if required
    if params.use_shallow_fusion or params.decoding_method in (
        "modified_beam_search_lm_rescore",
        "modified_beam_search_lm_rescore_LODR",
        "modified_beam_search_lm_shallow_fusion",
        "modified_beam_search_LODR",
    ):
        LM = LmScorer(
            lm_type=params.lm_type,
            params=params,
            device=device,
            lm_scale=params.lm_scale,
        )
        LM.to(device)
        LM.eval()
    else:
        LM = None

    # only load N-gram LM when needed
    if params.decoding_method == "modified_beam_search_lm_rescore_LODR":
        try:
            import kenlm
        except ImportError:
            print("Please install kenlm first. You can use")
            print(" pip install https://github.com/kpu/kenlm/archive/master.zip")
            print("to install it")
            import sys

            sys.exit(-1)
        ngram_file_name = str(params.lang_dir / f"{params.tokens_ngram}gram.arpa")
        logging.info(f"lm filename: {ngram_file_name}")
        ngram_lm = kenlm.Model(ngram_file_name)
        ngram_lm_scale = None  # use a list to search

    elif params.decoding_method == "modified_beam_search_LODR":
        lm_filename = f"{params.tokens_ngram}gram.fst.txt"
        logging.info(f"Loading token level lm: {lm_filename}")
        ngram_lm = NgramLm(
            str(params.lang_dir / lm_filename),
            backoff_id=params.backoff_id,
            is_binary=False,
        )
        logging.info(f"num states: {ngram_lm.lm.num_states}")
        ngram_lm_scale = params.ngram_lm_scale
    else:
        ngram_lm = None
        ngram_lm_scale = None

    if "fast_beam_search" in params.decoding_method:
        if params.decoding_method == "fast_beam_search_nbest_LG":
            lexicon = Lexicon(params.lang_dir)
            word_table = lexicon.word_table
            lg_filename = params.lang_dir / "LG.pt"
            logging.info(f"Loading {lg_filename}")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        else:
            word_table = None
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None
        word_table = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True

    if params.dataset == "estonian":
        from estonian_dataset import EstonianASRDataset, collate_fn
        logging.info("Using Estonian dataset")
        
        test_dataset = EstonianASRDataset(params.test_txt, base_path=params.audio_base_path)
        test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=params.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )
        
        test_sets = ["test"]
        test_dls = [test_dl]
    else:
        librispeech = LibriSpeechAsrDataModule(args)
        test_clean_cuts = librispeech.test_clean_cuts()
        test_other_cuts = librispeech.test_other_cuts()
        test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
        test_other_dl = librispeech.test_dataloaders(test_other_cuts)
        test_sets = ["test-clean", "test-other"]
        test_dls = [test_clean_dl, test_other_dl]

    import time
    for test_set, test_dl in zip(test_sets, test_dls):
        start = time.time()
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            word_table=word_table,
            decoding_graph=decoding_graph,
            LM=LM,
            ngram_lm=ngram_lm,
            ngram_lm_scale=ngram_lm_scale,
        )
        logging.info(f"Elapsed time for {test_set}: {time.time() - start}")

        save_results(
            params=params,
            test_set_name=test_set,
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
