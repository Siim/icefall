#!/usr/bin/env python3

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import k2
import sentencepiece as spm
import torch
import torchaudio
from beam_search import modified_beam_search

# Add the parent directory to the path to import from train.py
sys.path.insert(0, str(Path(__file__).parent))
from train import get_transducer_model

# Import AttributeDict from icefall
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
    """Load checkpoint from file.
    
    Args:
        filename: Path to the checkpoint file
        device: Device to load the model to
        
    Returns:
        Tuple of (params, model)
    """
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location=device)
    
    if "params" in checkpoint:
        params = checkpoint["params"]
    else:
        params = AttributeDict(
            {
                "vocab_size": 2500,  # Default for Estonian BPE
                "blank_id": 0,
                "context_size": 2,
                "use_xlsr": True,
                "xlsr_model_name": "facebook/wav2vec2-large-xlsr-53",  # Correct model name
                "decode_chunk_size": 8000,
                "attention_sink_size": 16,
                "decoder_dim": 512,
                "joiner_dim": 512,
                "blank_penalty": 0.0,
            }
        )
    
    # Add XLSR-specific parameters if they don't exist
    if not hasattr(params, "frame_duration"):
        params.frame_duration = 0.025  # 25ms per frame
    
    if not hasattr(params, "frame_stride"):
        params.frame_stride = 0.020  # 20ms stride
    
    if not hasattr(params, "context_frames"):
        params.context_frames = 10  # Default context frames
    
    if not hasattr(params, "transition_frames"):
        params.transition_frames = 5  # Default transition frames
    
    if not hasattr(params, "downsample_factor"):
        params.downsample_factor = 320  # For wav2vec2/XLSR models
    
    # Create model
    model = get_transducer_model(params)
    
    if "model" in checkpoint:
        logging.info("Loading model from checkpoint")
        model.load_state_dict(checkpoint["model"], strict=False)
    
    model.to(device)
    model.eval()
    
    return params, model

def normalize_audio(
    audio: torch.Tensor,
    sample_rate: int,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """Normalize audio to [-1, 1] range and resample if needed.
    
    Args:
        audio: Audio tensor of shape (channels, time) or (time,)
        sample_rate: Original sample rate
        target_sample_rate: Target sample rate (default: 16kHz)
        
    Returns:
        Normalized audio tensor of shape (batch=1, time)
    """
    # Ensure audio is 2D: (channels, time)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # Add channel dimension
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        logging.info(f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz")
        audio = torchaudio.functional.resample(
            audio, 
            orig_freq=sample_rate, 
            new_freq=target_sample_rate
        )
    
    # Convert to mono if stereo
    if audio.size(0) > 1:
        logging.info(f"Converting {audio.size(0)} channels to mono")
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Normalize to [-1, 1]
    if audio.abs().max() > 0:
        audio = audio / audio.abs().max()
    
    # Ensure shape is (batch=1, time)
    if audio.dim() == 2 and audio.size(0) == 1:
        # Shape is already (1, time)
        pass
    else:
        # Reshape to (1, time)
        audio = audio.reshape(1, -1)
    
    logging.info(f"Normalized audio shape: {audio.shape}, range: [{audio.min():.2f}, {audio.max():.2f}]")
    
    return audio

def process_streaming_chunks(
    model: torch.nn.Module,
    feature: torch.Tensor,
    chunk_size: int,
    attention_sink_size: int,
    left_context_chunks: int,
    device: torch.device,
) -> torch.Tensor:
    """Process audio in non-streaming mode since model is in pre-training phase.
    
    Args:
        model: The model to use
        feature: Input features (batch, time)
        chunk_size: Size of each chunk in samples (not used in non-streaming mode)
        attention_sink_size: Number of frames for attention sink (not used in non-streaming mode)
        left_context_chunks: Number of left context chunks (not used in non-streaming mode)
        device: Device to run inference on
    
    Returns:
        Encoder outputs processed in non-streaming mode
    """
    # Check if feature is 2D (batch, time) or 3D (batch, time, dim)
    if feature.dim() == 3 and feature.size(2) > 1:
        # Already processed features, just return them
        logging.info(f"Input is already processed features with shape {feature.shape}")
        return feature
    
    # Create feature lengths for all sequences
    batch_size = feature.shape[0]
    feature_lens = torch.tensor([feature.shape[1]] * batch_size, device=device)
    
    # Use encoder directly if model is wrapped in DDP
    if hasattr(model, 'module'):
        encoder = model.module.encoder
    else:
        encoder = model.encoder
    
    # Process in non-streaming mode (since model is in pre-training phase)
    logging.info("Using non-streaming mode for decoding (pre-training phase)")
    encoder_out, _ = encoder(
        x=feature,
        x_lens=feature_lens
    )
    
    logging.info(f"Encoder output shape: {encoder_out.shape}")
    
    return encoder_out

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
    logging.info(f"Model loaded successfully with parameters: {params}")
    
    # Load and normalize audio
    logging.info(f"Processing audio file: {args.audio_file}")
    audio, sample_rate = torchaudio.load(args.audio_file)
    audio = normalize_audio(audio, sample_rate)
    logging.info(f"Audio shape after normalization: {audio.shape}, sample rate: {sample_rate}")
    
    # Process in non-streaming mode (since model is in pre-training phase)
    with torch.no_grad():
        # Move audio to device
        audio = audio.to(device)
        
        # Create feature lengths tensor for batch processing
        feature_lens = torch.tensor([audio.size(1)], device=device)
        
        # Process directly through encoder in non-streaming mode
        logging.info("Using non-streaming mode for decoding")
        
        # Get encoder directly
        if hasattr(model, 'module'):
            encoder = model.module.encoder
        else:
            encoder = model.encoder
        
        # Log encoder type
        logging.info(f"Encoder type: {type(encoder).__name__}")
        
        # Get encoder output directly (no chunking)
        encoder_out, encoder_out_lens = encoder(
            x=audio,
            x_lens=feature_lens
        )
        
        logging.info(f"Encoder output shape: {encoder_out.shape}, lengths: {encoder_out_lens}")
        
        # Apply encoder projection if available
        if hasattr(model, 'module') and hasattr(model.module, 'encoder_proj'):
            logging.info("Applying encoder projection from module")
            encoder_out = model.module.encoder_proj(encoder_out)
        elif hasattr(model, 'encoder_proj'):
            logging.info("Applying encoder projection")
            encoder_out = model.encoder_proj(encoder_out)
        
        logging.info(f"Encoder output after projection: {encoder_out.shape}")
        
        # Get decoder
        if hasattr(model, 'module'):
            decoder = model.module.decoder
        else:
            decoder = model.decoder
        
        logging.info(f"Decoder type: {type(decoder).__name__}, blank_id: {decoder.blank_id}, context_size: {decoder.context_size}")
        
        # Get joiner
        if hasattr(model, 'module'):
            joiner = model.module.joiner
        else:
            joiner = model.joiner
        
        logging.info(f"Joiner type: {type(joiner).__name__}")
        
        # Decode with beam search
        logging.info(f"Decoding with beam size: {args.beam_size}, blank penalty: {args.blank_penalty}")
        
        # Import the modified_beam_search function
        from beam_search import modified_beam_search
        
        # Set a higher blank penalty to discourage repetitions
        effective_blank_penalty = args.blank_penalty + 0.5
        logging.info(f"Using effective blank penalty: {effective_blank_penalty}")
        
        # Run beam search with increased blank penalty
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=args.beam_size,
            temperature=args.temperature,
            blank_penalty=effective_blank_penalty
        )
        
        # Log the raw token IDs for debugging
        if isinstance(hyp_tokens[0], torch.Tensor):
            token_list = hyp_tokens[0].tolist()
        else:
            token_list = hyp_tokens[0]
        
        logging.info(f"Raw token IDs: {token_list}")
        
        # Convert tokens to text
        hyps = []
        for h in hyp_tokens:
            # Check if h is a tensor or a list
            if isinstance(h, torch.Tensor):
                h_list = h.tolist()
            else:
                h_list = h
            
            # Decode to text
            text = sp.decode(h_list)
            hyps.append(text)
        
        # Print the transcription
        print(f"\nTranscription: {hyps[0]}")
        logging.info(f"Transcription: {hyps[0]}")

if __name__ == "__main__":
    main() 