#!/usr/bin/env python3

import argparse
import logging
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torchaudio
from beam_search import modified_beam_search
from model import Transducer

from icefall.utils import AttributeDict, setup_logger, str2bool

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to load"
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        required=True,
        help="Path to the BPE model"
    )

    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to the audio file to decode"
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="Blank symbol penalty for beam search"
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size for modified beam search"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax"
    )

    parser.add_argument(
        "--use-xlsr",
        type=str2bool,
        default=True,
        help="Whether to use XLSR encoder"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5120,  # 320ms at 16kHz
        help="Chunk size for streaming decoding"
    )

    parser.add_argument(
        "--left-context-chunks",
        type=int,
        default=1,
        help="Number of left context chunks"
    )

    parser.add_argument(
        "--attention-sink-size",
        type=int,
        default=16,
        help="Number of frames for attention sink"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="log",
        help="Directory for saving logs"
    )

    return parser

def load_checkpoint(
    filename: str,
    device: torch.device,
) -> Tuple[AttributeDict, torch.nn.Module]:
    """Load checkpoint and return model parameters and model."""
    assert filename.is_file(), f"{filename} does not exist!"

    checkpoint = torch.load(filename, map_location=device)

    if "model" in checkpoint:
        model_state_dict = checkpoint["model"]
    else:
        model_state_dict = checkpoint

    # Convert model state dict to AttributeDict
    params = AttributeDict()
    params.update(checkpoint.get("params", {}))
    
    # Add default parameters without parsing command line
    default_args = {
        "vocab_size": checkpoint.get("vocab_size", 2500),  # Default BPE vocab size
        "blank_id": 0,  # Default blank ID
        "context_size": checkpoint.get("context_size", 2),
        "decoder_dim": checkpoint.get("decoder_dim", 512),
        "joiner_dim": checkpoint.get("joiner_dim", 512),
        "use_xlsr": True,  # Default to using XLSR
        
        # XLSR specific parameters
        "xlsr_model_name": "facebook/wav2vec2-large-xlsr-53",  # Correct model name from train_xlsr.sh
        "decode_chunk_size": 5120,  # 320ms at 16kHz
        "frame_duration": 0.025,  # 25ms per frame
        "frame_stride": 0.020,    # 20ms stride
        "downsample_factor": 320, # For wav2vec2/XLSR models
        "context_frames": 10,     # Default 10 additional context frames
        "transition_frames": 5,   # Default 5 frames for smooth transition
        
        # Additional parameters required by the encoder
        "attention_sink_size": 16,  # 16 frames (paper's optimal)
    }
    params.update(default_args)

    # Create model
    from train import get_transducer_model
    model = get_transducer_model(params)
    
    # Load state dict with weights_only=True to address the warning
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    return params, model

def normalize_audio(
    audio: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """Normalize and resample audio if needed."""
    # Resample if necessary
    if sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(
            audio, 
            sample_rate, 
            target_sample_rate
        )

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Normalize to [-1, 1]
    if audio.abs().max() > 1.0:
        audio = audio / audio.abs().max()

    # Ensure audio is in the format [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add channel dimension
        
    return audio

def process_streaming_chunks(
    model: torch.nn.Module,
    feature: torch.Tensor,
    chunk_size: int,
    attention_sink_size: int,
    left_context_chunks: int,
    device: torch.device,
) -> torch.Tensor:
    """Process audio in streaming mode with overlapping chunks."""
    # Handle different input shapes
    if feature.dim() == 2:  # [batch_size, seq_len]
        batch_size, seq_len = feature.shape
    elif feature.dim() == 3:  # [batch_size, channels, seq_len]
        batch_size, channels, seq_len = feature.shape
        # Reshape to [batch_size, seq_len] by taking the first channel
        feature = feature[:, 0, :]
    else:
        raise ValueError(f"Unexpected feature shape: {feature.shape}")
    
    # Calculate left context size
    left_context_size = left_context_chunks * chunk_size
    
    # Initialize attention sink if enabled
    if attention_sink_size > 0:
        attention_sink = torch.zeros(
            (batch_size, attention_sink_size, model.encoder.output_dim),
            device=device
        )
    else:
        attention_sink = None
    
    # Process chunks
    outputs = []
    cached_left_context = None
    
    for chunk_start in range(0, seq_len, chunk_size):
        # Extract current chunk
        chunk_end = min(chunk_start + chunk_size, seq_len)
        current_chunk = feature[:, chunk_start:chunk_end]
        
        # Add left context if available
        if cached_left_context is not None:
            context_size = min(left_context_size, cached_left_context.size(1))
            with_context = torch.cat([
                cached_left_context[:, -context_size:], 
                current_chunk
            ], dim=1)
        else:
            context_size = min(left_context_size, chunk_start)
            if context_size > 0:
                left_pad = feature[:, chunk_start-context_size:chunk_start]
                with_context = torch.cat([left_pad, current_chunk], dim=1)
            else:
                with_context = current_chunk
        
        # Create fake lengths for this chunk
        chunk_lens = torch.tensor([with_context.size(1)] * batch_size, device=device)
        
        # Process chunk
        chunk_out, _ = model.encoder(with_context, chunk_lens)
        
        # Apply attention sink if enabled
        if attention_sink is not None:
            chunk_out = torch.cat([attention_sink, chunk_out], dim=1)
            chunk_result = chunk_out[:, attention_sink_size:]
            sink_end = min(chunk_out.size(1), attention_sink_size)
            attention_sink = chunk_out[:, -sink_end:]
        else:
            left_context_frames = context_size // model.encoder.downsample_factor
            chunk_result = chunk_out[:, left_context_frames:]
        
        outputs.append(chunk_result)
        cached_left_context = current_chunk
    
    # Concatenate all outputs
    if outputs:
        combined_output = torch.cat(outputs, dim=1)
        expected_length = (seq_len // model.encoder.downsample_factor) + 1
        if combined_output.size(1) > expected_length:
            combined_output = combined_output[:, :expected_length]
        return combined_output
    else:
        return torch.zeros((batch_size, 0, model.encoder.output_dim), device=device)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Create log directory if it doesn't exist
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logger with a filename based on the audio file being processed
    audio_name = Path(args.audio_file).stem
    log_file = log_dir / f"decode_{audio_name}.log"
    setup_logger(log_file)
    
    # Load BPE model
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpe_model)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model
    params, model = load_checkpoint(Path(args.checkpoint), device)
    logging.info("Model loaded successfully")
    
    # Load and normalize audio
    logging.info(f"Processing audio file: {args.audio_file}")
    audio, sample_rate = torchaudio.load(args.audio_file)
    audio = normalize_audio(audio, sample_rate)
    logging.info(f"Audio shape after normalization: {audio.shape}")
    
    # Process in streaming mode
    with torch.no_grad():
        # Move audio to device
        audio = audio.to(device)
        
        # Process through encoder in chunks
        encoder_out = process_streaming_chunks(
            model=model,
            feature=audio,
            chunk_size=args.chunk_size,
            attention_sink_size=args.attention_sink_size,
            left_context_chunks=args.left_context_chunks,
            device=device
        )
        logging.info(f"Encoder output shape: {encoder_out.shape}")
        
        # Create encoder output lengths
        encoder_out_lens = torch.tensor(
            [encoder_out.size(1)],
            device=device,
            dtype=torch.int64
        )
        
        # Decode with beam search
        logging.info(f"Decoding with beam size: {args.beam_size}")
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=args.beam_size,
            temperature=args.temperature,
            blank_penalty=args.blank_penalty
        )
        
        # Convert tokens to text
        hyps = []
        for h in hyp_tokens:
            # Check if h is a tensor or a list
            if isinstance(h, torch.Tensor):
                h_list = h.tolist()
            else:
                h_list = h
            hyps.append(sp.decode(h_list))
        
        # Print the transcription
        print(f"\nTranscription: {hyps[0]}")
        logging.info(f"Transcription: {hyps[0]}")

if __name__ == "__main__":
    main() 