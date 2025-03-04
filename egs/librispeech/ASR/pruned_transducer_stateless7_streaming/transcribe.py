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

# Import train functions except for get_transducer_model and get_encoder_model
from train import (
    add_model_arguments,
    get_params,
    load_checkpoint_if_available,
    process_streaming_chunks,
    get_joiner_model,
    get_decoder_model
)

# Import XLSR encoder and other components
from encoder_interface import EncoderInterface
from xlsr_encoder import XLSREncoder
from model import Transducer
from beam_search import modified_beam_search

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
        params: Configuration parameters
        
    Returns:
        A Transducer model
    """
    logging.info("Creating transducer model with XLSR encoder")
    
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)
    
    return Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
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
        "--bpe-model",
        type=str,
        required=True,
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--wav-file",
        type=str,
        required=True,
        help="Path to the WAV file to transcribe",
    )

    parser.add_argument(
        "--streaming",
        type=bool,
        default=True,
        help="Whether to use streaming mode for transcription",
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size for beam search decoding",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=32,
        help="Max states for beam search",
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=4,
        help="Max contexts for beam search",
    )

    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.9,
        help="Penalty applied to blank symbol during decoding",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="320ms",
        choices=["320ms", "640ms", "1280ms", "2560ms"],
        help="Chunk size for streaming inference",
    )

    parser.add_argument(
        "--attention-sink-size",
        type=int,
        default=16,
        help="Number of frames for attention sink",
    )

    parser.add_argument(
        "--left-context-chunks",
        type=int,
        default=1,
        help="Number of left context chunks for streaming inference",
    )

    return parser


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file and convert to the correct sample rate.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate (default: 16000)
        
    Returns:
        torch.Tensor: Waveform tensor
    """
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono if needed
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != sample_rate:
        logging.info(f"Resampling from {sr} to {sample_rate}")
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    return waveform


def verify_audio_length(waveform: torch.Tensor, sample_rate: int, 
                      min_duration: float = 1.0, max_duration: float = 10.0) -> torch.Tensor:
    """Verify and adjust audio length if needed.
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        
    Returns:
        torch.Tensor: Adjusted waveform
    """
    # Get duration
    duration = waveform.size(1) / sample_rate
    
    # Check if audio exceeds max duration
    if duration > max_duration:
        logging.warning(f"Audio duration ({duration:.2f}s) exceeds max_duration ({max_duration}s), trimming")
        max_samples = int(max_duration * sample_rate)
        waveform = waveform[:, :max_samples]
    
    # Check if audio is below min duration
    if duration < min_duration:
        logging.warning(f"Audio duration ({duration:.2f}s) below min_duration ({min_duration}s)")
        padding = int((min_duration - duration) * sample_rate)
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    return waveform


def prepare_audio_features(audio_path: str, sample_rate: int = 16000) -> Dict[str, torch.Tensor]:
    """Load and prepare audio features similar to EstonianDataset.__getitem__.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Dict containing audio features in batch format
    """
    # Load and process audio
    waveform = load_audio(audio_path, sample_rate)
    
    # Verify and adjust audio length
    waveform = verify_audio_length(waveform, sample_rate)
    
    # Create batch-like structure
    features = {
        "inputs": waveform,  # Shape: [1, time]
        "input_lens": torch.tensor([waveform.size(1)]),
        "supervisions": {
            "text": [""],  # Empty text as we don't have reference
            "audio_path": audio_path
        }
    }
    
    return features


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
    
    # Load and prepare audio features
    features = prepare_audio_features(wav_path)
    
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
        encoder_out, encoder_out_lens = model.encoder(
            x=feature,
            x_lens=feature_lens
        )
    
    # Project encoder output if the model has a projection layer
    if hasattr(model, 'encoder_proj'):
        encoder_out = model.encoder_proj(encoder_out)
    
    # Decode using beam search
    start_time = time.time()
    
    hyps = modified_beam_search(
        model=model,
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        beam=params.beam_size,
        max_states=params.max_states,
        max_contexts=params.max_contexts,
        blank_penalty=params.blank_penalty,
    )
    
    end_time = time.time()
    latency = end_time - start_time
    logging.info(f"Beam search took {latency:.3f} seconds")
    
    # Convert predictions to text
    hyps_texts = [sp.decode(hyp["tokens"]) for hyp in hyps]
    
    # Return best hypothesis (first one)
    transcript = hyps_texts[0]
    logging.info(f"Transcript: {transcript}")
    
    return transcript


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
    params.attention_sink_size = args.attention_sink_size
    params.left_context_chunks = args.left_context_chunks
    
    # Set pretrained encoder configs
    params.pretrained_encoder = True
    params.xlsr_model_name = "facebook/wav2vec2-large-xlsr-53"
    
    # Load sentencepiece model
    logging.info(f"Loading SentencePiece model from {params.bpe_model}")
    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)
    
    # Create model
    logging.info("Creating model with XLSR encoder")
    model = get_transducer_model(params)
    model.to(device)
    
    # Load checkpoint
    logging.info(f"Loading checkpoint from {params.checkpoint}")
    load_checkpoint_if_available(params=params, model=model)
    
    # Transcribe
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