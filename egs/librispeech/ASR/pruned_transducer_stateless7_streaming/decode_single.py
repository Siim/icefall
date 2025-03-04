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
    
    # Log initial shape
    logging.info(f"Initial audio shape: {audio.shape}")
    
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
    
    # Verify audio meets requirements (similar to EstonianDataset._verify_audio)
    min_val, max_val = audio.min().item(), audio.max().item()
    if min_val < -1.01 or max_val > 1.01:
        logging.info(f"Audio values outside range [-1, 1]: min={min_val:.4f}, max={max_val:.4f}. Normalizing.")
        # Normalize to [-1, 1]
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()
    
    # Check for silent or constant audio
    if audio.std() < 1e-6:
        logging.warning(f"Audio has very low variance (possibly silent or DC): std={audio.std():.8f}")
    
    # Ensure shape is (batch=1, time) without using transpose
    # At this point, audio should be (channels=1, time)
    audio = audio.reshape(1, -1)  # Safely reshape to (1, time) regardless of input shape
    
    logging.info(f"Normalized audio shape: {audio.shape}, range: [{audio.min().item():.2f}, {audio.max().item():.2f}], std: {audio.std().item():.6f}")
    
    return audio

def process_streaming_chunks(
    model: torch.nn.Module,
    feature: torch.Tensor,
    chunk_size: int,
    attention_sink_size: int,
    left_context_chunks: int,
    device: torch.device,
) -> torch.Tensor:
    """Process audio in streaming mode with chunks.
    
    Args:
        model: The model to use
        feature: Input features (batch, time)
        chunk_size: Size of each chunk in samples
        attention_sink_size: Number of frames for attention sink
        left_context_chunks: Number of left context chunks
        device: Device to run inference on
    
    Returns:
        Encoder outputs processed in streaming mode
    """
    # Ensure feature is on the correct device
    feature = feature.to(device)
    
    # Get dimensions
    batch_size, seq_len = feature.shape
    
    # Use encoder directly if model is wrapped
    if hasattr(model, 'module'):
        encoder = model.module.encoder
    else:
        encoder = model.encoder
    
    # Calculate the actual context size in samples
    left_context_size = left_context_chunks * chunk_size
    
    # Initialize the attention sink if needed
    if attention_sink_size > 0:
        # This will be initialized with actual frames later
        attention_sink = None
    else:
        attention_sink = None
    
    # Calculate number of chunks and log
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    logging.info(f"Processing audio in {num_chunks} chunks of size {chunk_size}")
    
    # Process the audio in chunks
    outputs = []
    cached_left_context = None
    
    for i in range(num_chunks):
        # Determine current chunk boundaries
        chunk_start = i * chunk_size
        chunk_end = min(chunk_start + chunk_size, seq_len)
        current_chunk = feature[:, chunk_start:chunk_end]
        
        logging.info(f"Processing chunk {i+1}/{num_chunks}: frames {chunk_start}:{chunk_end}, shape {current_chunk.shape}")
        
        # Add left context if available
        if cached_left_context is not None:
            context_size = min(left_context_size, cached_left_context.size(1))
            with_context = torch.cat([
                cached_left_context[:, -context_size:], 
                current_chunk
            ], dim=1)
        else:
            # For the first chunk, use available left context if any
            context_size = min(left_context_size, chunk_start)
            if context_size > 0:
                left_pad = feature[:, chunk_start-context_size:chunk_start]
                with_context = torch.cat([left_pad, current_chunk], dim=1)
            else:
                with_context = current_chunk
        
        # Create lengths tensor for this chunk
        chunk_lens = torch.tensor([with_context.size(1)] * batch_size, device=device)
        
        # Process through encoder
        with torch.no_grad():
            chunk_out, _ = encoder(x=with_context, x_lens=chunk_lens)
        
        # Apply attention sink mechanism if enabled
        if attention_sink_size > 0:
            if attention_sink is not None:
                # Prepend attention sink tokens to the chunk output
                chunk_out_with_sink = torch.cat([attention_sink, chunk_out], dim=1)
                
                # Keep only the output excluding the sink portion
                chunk_result = chunk_out_with_sink[:, attention_sink.size(1):]
                
                # Update sink with last n frames
                attention_sink = chunk_out[:, -min(attention_sink_size, chunk_out.size(1)):]
            else:
                # For first chunk, initialize attention sink and use full output
                chunk_result = chunk_out
                attention_sink = chunk_out[:, -min(attention_sink_size, chunk_out.size(1)):]
                logging.info(f"Added attention sink of size {attention_sink.size(1)}")
        else:
            # If not using attention sink, still need to trim left context
            downsample_factor = 4  # Assuming 4x downsampling in encoder
            left_context_frames = context_size // downsample_factor
            
            if left_context_frames < chunk_out.size(1):
                chunk_result = chunk_out[:, left_context_frames:]
            else:
                logging.warning(f"Left context frames {left_context_frames} exceeds chunk output size {chunk_out.size(1)}")
                chunk_result = chunk_out[:, -1:]
        
        # Store result
        outputs.append(chunk_result)
        
        # Update cached context
        cached_left_context = current_chunk
    
    # Concatenate all chunks
    if outputs:
        encoder_out = torch.cat(outputs, dim=1)
        logging.info(f"Combined encoder output shape: {encoder_out.shape}")
        return encoder_out
    else:
        logging.warning("No outputs generated during streaming processing")
        # Return empty tensor with correct dimensions
        output_dim = encoder.output_dim if hasattr(encoder, 'output_dim') else 1024
        return torch.zeros((batch_size, 0, output_dim), device=device)

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
    
    # Process with no gradient calculation
    with torch.no_grad():
        # Load model
        logging.info(f"Loading model from checkpoint")
        params, model = load_checkpoint(Path(args.checkpoint), device)
        logging.info(f"Model loaded successfully with parameters: {params}")
        
        # Load and normalize audio
        logging.info(f"Processing audio file: {args.audio_file}")
        waveform, sample_rate = torchaudio.load(args.audio_file)
        
        # Verify sample rate
        if sample_rate != 16000:
            logging.info(f"Resampling audio from {sample_rate}Hz to 16000Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Check that audio is mono
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            logging.warning(f"Audio should be mono, got {waveform.shape[0]} channels. Converting to mono.")
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Check normalization range (similar to EstonianDataset._verify_audio)
        min_val, max_val = waveform.min().item(), waveform.max().item()
        if min_val < -1.01 or max_val > 1.01:
            logging.info(f"Audio values outside range [-1, 1]: min={min_val:.4f}, max={max_val:.4f}. Normalizing.")
            # Normalize to [-1, 1]
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
        
        # Log audio stats
        logging.info(f"Normalized audio shape: {waveform.shape}, range: [{waveform.min().item():.2f}, {waveform.max().item():.2f}], std: {waveform.std().item():.6f}")
        
        # Move audio to device
        waveform = waveform.to(device)
        
        # Get encoder directly
        if hasattr(model, 'module'):
            encoder = model.module.encoder
        else:
            encoder = model.encoder
        
        # Log encoder type
        logging.info(f"Encoder type: {type(encoder).__name__}")
        
        # Determine if we should use streaming mode based on chunk size
        use_streaming = args.chunk_size > 0 and args.left_context_chunks > 0
        
        if use_streaming:
            logging.info(f"Using streaming mode with chunk size: {args.chunk_size}, " 
                        f"left context chunks: {args.left_context_chunks}, "
                        f"attention sink size: {args.attention_sink_size}")
            
            # Process in streaming chunks
            encoder_out = process_streaming_chunks(
                model=model,
                feature=waveform,
                chunk_size=args.chunk_size,
                attention_sink_size=args.attention_sink_size,
                left_context_chunks=args.left_context_chunks,
                device=device
            )
            encoder_out_lens = torch.tensor([encoder_out.size(1)], device=device)
        else:
            # Get encoder output directly (no chunking)
            logging.info("Processing full audio without chunking")
            feature_lens = torch.tensor([waveform.size(1)], device=device)
            encoder_out, encoder_out_lens = encoder(
                x=waveform,
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
        
        # Use a moderate blank penalty - 0.5 is used in training
        effective_blank_penalty = args.blank_penalty + 0.5  # Use same as training
        logging.info(f"Using effective blank penalty: {effective_blank_penalty} (base: {args.blank_penalty})")
        
        # Use the exact same beam search parameters as in validation
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=args.beam_size,
            temperature=1.0,
            blank_penalty=effective_blank_penalty
        )
        
        # Log the raw token IDs for debugging
        if len(hyp_tokens) > 0:
            logging.info(f"Raw token IDs: {hyp_tokens[0]}")
            logging.info(f"Number of tokens: {len(hyp_tokens[0])}")
        
        # Convert tokens to text
        hyps = []
        for h in hyp_tokens:
            # Check if h is a tensor or a list
            if isinstance(h, torch.Tensor):
                h_list = h.tolist()
            else:
                h_list = h
            
            text = sp.decode(h_list)
            hyps.append(text)
        
        # Print the transcription
        print(f"\nTranscription: {hyps[0]}")
        logging.info(f"Transcription: {hyps[0]}")

if __name__ == "__main__":
    main() 