#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

# Import necessary modules
from icefall.utils import AttributeDict, setup_logger
import sentencepiece as spm

# Import train functions except for get_transducer_model, get_encoder_model, and load_checkpoint_if_available
from train import (
    add_model_arguments,
    get_params,
    process_streaming_chunks,
    get_joiner_model,
    get_decoder_model
)

# Import beam search functions
from beam_search import modified_beam_search

# Import XLSR encoder and other components
from encoder_interface import EncoderInterface
from xlsr_encoder import XLSREncoder
from model import Transducer

# Override get_encoder_model to create XLSR encoder directly
def get_encoder_model(params: AttributeDict):
    """Create an XLSR encoder model.
    
    Args:
        params: Configuration parameters.
        
    Returns:
        An instance of XLSREncoder.
    """
    from xlsr_encoder import XLSREncoder
    
    logging.info(f"Creating XLSR encoder with model name: {params.xlsr_model_name}")
    
    # Set default values to ensure they match the XLSREncoder constructor
    decode_chunk_size = getattr(params, 'decode_chunk_size', 8000)
    use_attention_sink = getattr(params, 'use_attention_sink', True)
    attention_sink_size = getattr(params, 'attention_sink_size', 4)
    
    encoder = XLSREncoder(
        model_name=params.xlsr_model_name,
        decode_chunk_size=decode_chunk_size,
        use_attention_sink=use_attention_sink,
        attention_sink_size=attention_sink_size
    )
    
    return encoder

# Override get_transducer_model to use our custom encoder
def get_transducer_model(params: AttributeDict) -> torch.nn.Module:
    """Build a transducer model with XLSR encoder.
    
    Args:
        params: Model configuration parameters.
        
    Returns:
        A Transducer model.
    """
    from model import Transducer
    
    logging.info("Creating transducer model with XLSR encoder")
    
    # Create encoder
    encoder = get_encoder_model(params)
    
    # Get the decoder model
    try:
        # First try direct import from train
        from pruned_transducer_stateless7_streaming.train import get_decoder_model, get_joiner_model
        decoder = get_decoder_model(params)
        joiner = get_joiner_model(params)
    except (ImportError, AttributeError):
        # Fallback to direct implementation if train import fails
        from decoder import Decoder
        from joiner import Joiner
        
        decoder = Decoder(
            vocab_size=params.vocab_size,
            decoder_dim=params.decoder_dim,
            blank_id=params.blank_id,
            context_size=2,
        )
        
        joiner = Joiner(
            encoder_dim=params.encoder_dim,
            decoder_dim=params.decoder_dim,
            joiner_dim=params.joiner_dim,
            vocab_size=params.vocab_size,
        )
        
    # Create the transducer model
    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=params.encoder_dim,
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    
    return model

def get_parser():
    """Get argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add model arguments from train.py
    add_model_arguments(parser)
    
    # Checkpoint and model file arguments
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint file"
    )
    
    parser.add_argument(
        "--bpe-model", 
        type=str, 
        required=True, 
        help="Path to the sentencepiece model"
    )
    
    # Audio file argument
    parser.add_argument(
        "--wav-file", 
        type=str, 
        required=True, 
        help="Path to the input WAV file"
    )
    
    # Beam search parameters
    parser.add_argument(
        "--beam-size", 
        type=int, 
        default=4, 
        help="Beam size for beam search"
    )
    
    # Streaming parameters
    parser.add_argument(
        "--streaming", 
        action="store_true", 
        help="Whether to use streaming mode"
    )
    
    parser.add_argument(
        "--chunk-size", 
        type=str, 
        default="1280ms", 
        choices=["320ms", "640ms", "1280ms", "2560ms"], 
        help="Chunk size for streaming inference"
    )
    
    parser.add_argument(
        "--attention-sink-size", 
        type=int, 
        default=16, 
        help="Size of attention sink (number of frames)"
    )
    
    parser.add_argument(
        "--left-context-chunks", 
        type=int, 
        default=1, 
        help="Number of left context chunks"
    )
    
    # Audio preprocessing parameters
    parser.add_argument(
        "--normalization", 
        type=str, 
        default="peak", 
        choices=["none", "peak", "rms"], 
        help="Audio normalization method"
    )
    
    parser.add_argument(
        "--max-duration", 
        type=float, 
        default=20.0, 
        help="Maximum audio duration in seconds"
    )
    
    # Experimental options
    parser.add_argument(
        "--try-penalties", 
        action="store_true", 
        help="Try different blank penalties and show all results"
    )
    
    return parser


def load_audio(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and convert to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Audio waveform as torch tensor
    """
    logging.info(f"Loading audio from {file_path}")
    
    # Handle different possible path formats (especially for Windows paths)
    if file_path.startswith('/C/'):
        file_path = 'C:' + file_path[2:]
    
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
        
    try:
        # Load audio with torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sample_rate != target_sr:
            logging.info(f"Resampling from {sample_rate} to {target_sr}")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sr
            )
            waveform = resampler(waveform)
            
        return waveform
        
    except Exception as e:
        logging.error(f"Error loading audio: {e}")
        raise

def normalize_audio(waveform: torch.Tensor, normalization_type: str = "peak") -> torch.Tensor:
    """Normalize audio waveform.
    
    Args:
        waveform: Audio waveform
        normalization_type: Type of normalization ("peak", "rms", or "none")
        
    Returns:
        Normalized waveform
    """
    if normalization_type == "none":
        return waveform
        
    elif normalization_type == "peak":
        # Peak normalization to range [-1, 1]
        peak = torch.max(torch.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
            
    elif normalization_type == "rms":
        # RMS normalization
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms > 0:
            waveform = waveform / (rms * 2)  # Scale to roughly [-1, 1]
            
    # Ensure we don't exceed bounds
    waveform = torch.clamp(waveform, min=-1.0, max=1.0)
    
    return waveform

def prepare_audio_features(wav_path: str, 
                          min_duration: float = 0.5,
                          max_duration: float = 20.0,
                          normalization: str = "peak") -> Dict[str, torch.Tensor]:
    """Prepare audio features for inference.
    
    Args:
        wav_path: Path to WAV file
        min_duration: Minimum audio duration
        max_duration: Maximum audio duration (increased to 20s)
        normalization: Type of normalization to apply ("peak", "rms", or "none")
        
    Returns:
        Dict with preprocessed inputs and lengths
    """
    # Load audio file
    waveform = load_audio(wav_path)
    
    # Apply normalization
    waveform = normalize_audio(waveform, normalization_type=normalization)
    
    # Verify audio length is within bounds
    waveform = verify_audio_length(waveform, 
                                  sample_rate=16000, 
                                  min_duration=min_duration, 
                                  max_duration=max_duration)
    
    # Create inputs tensor
    inputs = waveform.squeeze(0).unsqueeze(0)  # [1, num_samples]
    
    # Get input length in samples
    input_lens = torch.tensor([inputs.size(1)], dtype=torch.long)
    
    logging.info(f"Preprocessed audio: shape={inputs.shape}, duration={inputs.size(1)/16000:.2f}s")
    logging.info(f"Audio stats - min: {inputs.min().item():.2f}, max: {inputs.max().item():.2f}, mean: {inputs.mean().item():.2f}")
    
    # Return features dict
    return {
        "inputs": inputs,
        "input_lens": input_lens
    }


def verify_audio_length(waveform: torch.Tensor, sample_rate: int, 
                  min_duration: float = 0.5, max_duration: float = 20.0) -> torch.Tensor:
    """Verify the audio is within the acceptable duration range.
    
    Args:
        waveform: Audio tensor
        sample_rate: Sample rate of the audio
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds (increased to 20s)
        
    Returns:
        Verified and possibly trimmed waveform
    """
    # Get audio duration in seconds
    duration = waveform.size(1) / sample_rate
    
    # Check if duration is below minimum
    if duration < min_duration:
        logging.warning(f"Audio duration ({duration:.2f}s) is below min_duration ({min_duration}s)")
        # Pad with zeros to reach minimum duration
        padding_length = int((min_duration - duration) * sample_rate)
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))
        
    # Check if duration exceeds maximum
    if duration > max_duration:
        logging.warning(f"Audio duration ({duration:.2f}s) exceeds max_duration ({max_duration}s), trimming")
        # Trim to maximum duration
        max_samples = int(max_duration * sample_rate)
        waveform = waveform[:, :max_samples]
        
    return waveform


def transcribe_wav(wav_path: str, 
                  model: torch.nn.Module, 
                  sp: spm.SentencePieceProcessor, 
                  params: AttributeDict, 
                  device: torch.device) -> str:
    """Transcribe a WAV file using the XLSR-Transducer model.
    
    Args:
        wav_path: Path to WAV file
        model: Loaded XLSR-Transducer model
        sp: SentencePiece processor
        params: Model parameters
        device: Device to run inference on
        
    Returns:
        Transcription text
    """
    logging.info(f"Transcribing: {wav_path}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Load and prepare audio features with improved preprocessing
    features = prepare_audio_features(
        wav_path,
        min_duration=0.5,
        max_duration=20.0,
        normalization="peak"  # Use peak normalization as default
    )
    
    # Log detailed information about features
    logging.info(f"Feature shapes - inputs: {features['inputs'].shape}, input_lens: {features['input_lens']}")
    
    # Move to device
    feature = features["inputs"].to(device)
    feature_lens = features["input_lens"].to(device)
    
    # Process according to streaming/non-streaming mode
    if params.streaming:
        logging.info(f"Using streaming mode with chunk size {params.chunk_sizes[params.chunk_size]}")
        
        # Get chunk size in samples
        chunk_size = params.chunk_sizes[params.chunk_size]
        
        # Process in streaming chunks with attention sink
        encoder_out, encoder_out_lens = process_streaming_chunks(
            model=model,
            feature=feature,
            chunk_size=chunk_size,
            attention_sink_size=params.attention_sink_size,
            left_context_chunks=params.left_context_chunks,
            is_pre_training=False
        )
    else:
        logging.info(f"Using non-streaming mode")
        # Process with full context
        logging.info("Passing features to encoder...")
        start_encode = time.time()
        encoder_out, encoder_out_lens = model.encoder(
            x=feature,
            x_lens=feature_lens
        )
        encode_time = time.time() - start_encode
        logging.info(f"Encoder processing took {encode_time:.3f} seconds")
        logging.info(f"Encoder output shape: {encoder_out.shape}, output_lens: {encoder_out_lens}")
    
    # Project encoder output if the model has a projection layer
    if hasattr(model, 'encoder_proj'):
        logging.info("Applying encoder projection...")
        encoder_out = model.encoder_proj(encoder_out)
        logging.info(f"After projection shape: {encoder_out.shape}")
    
    # Log tensor stats to help diagnose issues
    logging.info(f"Encoder output stats - min: {encoder_out.min().item():.3f}, "
                f"max: {encoder_out.max().item():.3f}, "
                f"mean: {encoder_out.mean().item():.3f}, "
                f"std: {encoder_out.std().item():.3f}")
    
    # Decode using beam search
    logging.info(f"Starting beam search with beam size {params.beam_size}, "
                f"blank penalty {params.blank_penalty}, "
                f"temperature {params.temperature}")
    start_time = time.time()
    
    # Call modified_beam_search with the parameters it accepts
    hyps = modified_beam_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=params.beam_size,
        blank_penalty=getattr(params, 'blank_penalty', 0.0),
        temperature=getattr(params, 'temperature', 1.0),
        return_timestamps=False,
    )
    
    end_time = time.time()
    latency = end_time - start_time
    logging.info(f"Beam search took {latency:.3f} seconds")
    
    # Log hypotheses for diagnosis
    logging.info(f"Number of hypotheses returned: {len(hyps)}")
    
    # Get the best hypothesis
    best_hyp = hyps[0][0]
    logging.info(f"Best hypothesis tokens: {best_hyp}")
    
    # Convert tokens to text
    if isinstance(best_hyp, int):
        tokens = [best_hyp]
    else:
        tokens = best_hyp
    
    # Log individual tokens for diagnosis    
    if len(tokens) <= 10:
        token_texts = []
        for token in tokens:
            piece = sp.id_to_piece(token)
            token_texts.append(f"{token} → '{piece}'")
        logging.info(f"Token details: {', '.join(token_texts)}")
    else:
        logging.info(f"First 10 tokens: {tokens[:10]}")
        
    text = sp.decode(tokens)
    
    logging.info(f"Transcription: {text}")
    
    return text


def load_checkpoint_if_available(params: AttributeDict, model: torch.nn.Module):
    """Simplified function to load checkpoint for inference only.
    
    Args:
        params: Configuration parameters.
        model: The model to load checkpoint for.
    """
    checkpoint_path = params.checkpoint
    
    if checkpoint_path is None:
        return None
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        logging.info("Loading model parameters from checkpoint['model']")
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        logging.info("Loading model parameters directly")
        model.load_state_dict(checkpoint, strict=False)
    
    return None


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    log_dir = Path("pruned_transducer_stateless7_streaming/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger with proper file path
    log_filename = log_dir / f"transcribe_{Path(args.wav_file).stem}.log"
    setup_logger(log_filename)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Get basic model parameters
    params = get_params()
    
    # Update with command line args
    params.update(vars(args))
    
    # Define chunk sizes
    params.chunk_sizes = {
        "320ms": 5120,    # 320ms at 16kHz
        "640ms": 10240,   # 640ms at 16kHz
        "1280ms": 20480,  # 1.28s at 16kHz
        "2560ms": 40960,  # 2.56s at 16kHz
    }
    
    # Set essential XLSR parameters
    params.encoder_type = "XLSR"
    params.encoder_dim = 1024  # XLSR output dimension
    params.decoder_dim = 512
    params.joiner_dim = 512
    params.vocab_size = 2500  # BPE vocabulary size
    params.use_encoder_proj = True
    
    # Set XLSR specific streaming parameters
    params.is_streaming = params.streaming  # Convert boolean to attribute
    params.use_attention_sink = True
    params.attention_sink_size = args.attention_sink_size if hasattr(args, 'attention_sink_size') else 16
    params.left_context_chunks = args.left_context_chunks if hasattr(args, 'left_context_chunks') else 1
    
    # Set beam search parameters
    params.beam_size = args.beam_size if hasattr(args, 'beam_size') else 4
    params.blank_penalty = 0.5  # Increased from 0.0 to encourage more tokens
    params.temperature = 1.0
    
    # Set pretrained encoder configs
    params.pretrained_encoder = True
    params.xlsr_model_name = "facebook/wav2vec2-large-xlsr-53"
    
    # Load sentencepiece model
    logging.info(f"Loading SentencePiece model from {params.bpe_model}")
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    
    # Set blank_id and update vocab_size from the loaded model
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()
    logging.info(f"Vocabulary size: {params.vocab_size}, blank_id: {params.blank_id}")
    
    # Create model
    logging.info("Creating model with XLSR encoder")
    model = get_transducer_model(params)
    model.to(device)
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {params.checkpoint}")
    load_checkpoint_if_available(params=params, model=model)
    
    # Transcribe with different blank penalty values if requested
    if hasattr(args, 'try_penalties') and args.try_penalties:
        for penalty in [0.0, 0.3, 0.5, 0.8, 1.0]:
            logging.info(f"\n======= Testing with blank_penalty={penalty} =======")
            params.blank_penalty = penalty
            transcript = transcribe_wav(
                wav_path=args.wav_file,
                model=model,
                sp=sp,
                params=params,
                device=device
            )
            print(f"\nBlank penalty {penalty}: {transcript}")
    else:
        # Single transcription with current parameters
        transcript = transcribe_wav(
            wav_path=args.wav_file,
            model=model,
            sp=sp,
            params=params,
            device=device
        )
        
        # Print the final result
        print(f"\nFinal Transcript: {transcript}")


if __name__ == "__main__":
    main() 