#!/usr/bin/env python3
# Copyright    2024                      (authors: Siim Haugas)
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
This script evaluates the XLSR-Transducer model in streaming mode with different chunk sizes.
"""

import argparse
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import sentencepiece as spm
import torch
import torchaudio
from model import Transducer
from beam_search import beam_search, fast_beam_search_one_best

# Set up logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    
    parser.add_argument(
        "--test-list",
        type=str,
        default="Data/val_list.txt",
        help="Path to the test file list",
    )
    
    parser.add_argument(
        "--sp-model",
        type=str,
        default="Data/lang_bpe_2500/bpe.model",
        help="Path to the SentencePiece model",
    )
    
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="5120,10240,20480,40960",
        help="Comma-separated list of chunk sizes to evaluate",
    )
    
    parser.add_argument(
        "--left-context-chunks",
        type=int,
        default=1,
        help="Number of left context chunks",
    )
    
    parser.add_argument(
        "--use-attention-sink",
        type=str2bool,
        default=True,
        help="Whether to use attention sinks",
    )
    
    parser.add_argument(
        "--attention-sink-size",
        type=int,
        default=16,
        help="Number of frames to use as attention sink",
    )
    
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=100,
        help="Number of test samples to evaluate",
    )
    
    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size for beam search",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    
    return parser

def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, any]]:
    """Load model checkpoint."""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model" not in checkpoint:
        raise ValueError("Checkpoint does not contain model state")
    
    model_state = checkpoint["model"]
    args_dict = checkpoint.get("args", {})
    
    return model_state, args_dict

def load_model(model_state: Dict[str, torch.Tensor], args_dict: Dict[str, any], device: torch.device) -> Transducer:
    """Load transducer model from checkpoint state."""
    from xlsr_encoder import XLSREncoder
    
    # Create encoder
    encoder = XLSREncoder(
        model_name=args_dict.get("xlsr_model_name", "facebook/wav2vec2-large-xlsr-53"),
        decode_chunk_size=args_dict.get("xlsr_chunk_size", 8000),
        use_attention_sink=args_dict.get("xlsr_use_attention_sink", True),
        attention_sink_size=args_dict.get("xlsr_attention_sink_size", 16),
        context_frames=args_dict.get("xlsr_left_context_chunks", 1) * 10,
    )
    
    # Create decoder
    from decoder import Decoder
    decoder = Decoder(
        vocab_size=args_dict.get("vocab_size", 2500),
        decoder_dim=args_dict.get("decoder_dim", 512),
        blank_id=0,
    )
    
    # Create joiner
    from joiner import Joiner
    joiner = Joiner(
        encoder_dim=encoder.output_dim,
        decoder_dim=args_dict.get("decoder_dim", 512),
        joiner_dim=args_dict.get("joiner_dim", 512),
        vocab_size=args_dict.get("vocab_size", 2500),
    )
    
    # Create transducer model
    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=encoder.output_dim,
        decoder_dim=args_dict.get("decoder_dim", 512),
        joiner_dim=args_dict.get("joiner_dim", 512),
        vocab_size=args_dict.get("vocab_size", 2500),
    )
    
    # Load state dict
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    return model

def load_test_data(test_list: str, max_samples: int) -> List[Dict[str, str]]:
    """Load test data from file list."""
    logging.info(f"Loading test data from {test_list}")
    
    items = []
    with open(test_list, 'r', encoding='utf-8') as f:
        for line in f:
            if len(items) >= max_samples:
                break
                
            parts = line.strip().split('|')
            if len(parts) >= 2:
                audio_path, text = parts[0], parts[1]
                
                # Check if file exists
                if not os.path.exists(audio_path):
                    logging.warning(f"File not found: {audio_path}, skipping")
                    continue
                
                items.append({
                    "audio_path": audio_path,
                    "text": text
                })
    
    logging.info(f"Loaded {len(items)} test samples")
    return items

def process_audio(audio_path: str, device: torch.device) -> torch.Tensor:
    """Process audio file for model input."""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Normalize
    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()
    
    return waveform.to(device)

def evaluate_streaming(
    model: Transducer,
    sp: spm.SentencePieceProcessor,
    test_data: List[Dict[str, str]],
    chunk_size: int,
    left_context_chunks: int,
    use_attention_sink: bool,
    attention_sink_size: int,
    beam_size: int,
    device: torch.device,
) -> float:
    """Evaluate model in streaming mode with specified chunk size."""
    logging.info(f"Evaluating with chunk size: {chunk_size} samples ({chunk_size/16000:.2f}s)")
    
    # Configure encoder for streaming
    model.encoder.decode_chunk_size = chunk_size
    model.encoder.use_attention_sink = use_attention_sink
    model.encoder.attention_sink_size = attention_sink_size
    model.encoder.context_frames = left_context_chunks * 10  # 10 frames per chunk
    
    total_errors = 0
    total_words = 0
    
    for i, item in enumerate(test_data):
        audio_path = item["audio_path"]
        reference = item["text"]
        
        # Process audio
        waveform = process_audio(audio_path, device)
        
        # Reset encoder streaming state
        model.encoder.reset_streaming_state()
        
        # Process in chunks
        audio_len = waveform.size(1)
        pos = 0
        streaming_state = None
        hyps = []
        
        # Initialize decoder state
        decoder_state = model.decoder.get_init_state()
        
        while pos < audio_len:
            # Get current chunk
            chunk_end = min(pos + chunk_size, audio_len)
            chunk = waveform[:, pos:chunk_end]
            chunk_len = torch.tensor([chunk.size(1)], device=device)
            
            # Process chunk
            encoder_out, encoder_out_lens, next_state = model.encoder.streaming_forward(
                chunk, chunk_len, streaming_state
            )
            
            # Update streaming state
            streaming_state = next_state
            
            # Beam search on current chunk
            hyp = beam_search(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=beam_size,
                decoder_state=decoder_state,
            )
            
            # Update decoder state
            decoder_state = hyp.decoder_state
            
            # Add to hypothesis
            hyps.append(hyp.tokens.tolist())
            
            # Update position
            pos = chunk_end
        
        # Combine chunks and remove duplicates
        combined_hyp = []
        prev_token = -1
        for chunk_tokens in hyps:
            for token in chunk_tokens:
                if token != prev_token and token != 0:  # Skip blanks and duplicates
                    combined_hyp.append(token)
                prev_token = token
        
        # Convert to text
        hypothesis = sp.decode(combined_hyp)
        
        # Calculate WER
        errors = calculate_errors(reference, hypothesis)
        words = len(reference.split())
        
        total_errors += errors
        total_words += words
        
        if i < 5 or i % 20 == 0:
            logging.info(f"Sample {i+1}/{len(test_data)}")
            logging.info(f"Reference: {reference}")
            logging.info(f"Hypothesis: {hypothesis}")
            logging.info(f"WER: {errors/words:.4f} ({errors}/{words})")
    
    wer = total_errors / total_words if total_words > 0 else 1.0
    logging.info(f"Overall WER: {wer:.4f} ({total_errors}/{total_words})")
    
    return wer

def calculate_errors(ref: str, hyp: str) -> int:
    """Calculate word error rate between reference and hypothesis."""
    # Normalize text
    ref = ref.lower().strip()
    hyp = hyp.lower().strip()
    
    # Split into words
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    # Calculate edit distance
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,  # deletion
                    dp[i][j-1] + 1,  # insertion
                    dp[i-1][j-1] + 1  # substitution
                )
    
    return dp[m][n]

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    
    # Load checkpoint
    model_state, args_dict = load_checkpoint(args.checkpoint, device)
    
    # Load model
    model = load_model(model_state, args_dict, device)
    
    # Load SentencePiece model
    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model)
    
    # Load test data
    test_data = load_test_data(args.test_list, args.num_test_samples)
    
    # Parse chunk sizes
    chunk_sizes = [int(size) for size in args.chunk_sizes.split(",")]
    
    # Evaluate with different chunk sizes
    results = {}
    for chunk_size in chunk_sizes:
        wer = evaluate_streaming(
            model=model,
            sp=sp,
            test_data=test_data,
            chunk_size=chunk_size,
            left_context_chunks=args.left_context_chunks,
            use_attention_sink=args.use_attention_sink,
            attention_sink_size=args.attention_sink_size,
            beam_size=args.beam_size,
            device=device,
        )
        results[chunk_size] = wer
    
    # Print summary
    logging.info("\nEvaluation Summary:")
    logging.info(f"Model: {args.checkpoint}")
    logging.info(f"Left context chunks: {args.left_context_chunks}")
    logging.info(f"Attention sink: {'Enabled' if args.use_attention_sink else 'Disabled'}")
    if args.use_attention_sink:
        logging.info(f"Attention sink size: {args.attention_sink_size}")
    
    logging.info("\nResults by chunk size:")
    for chunk_size, wer in results.items():
        logging.info(f"Chunk size: {chunk_size} samples ({chunk_size/16000:.2f}s) - WER: {wer:.4f}")

if __name__ == "__main__":
    main() 