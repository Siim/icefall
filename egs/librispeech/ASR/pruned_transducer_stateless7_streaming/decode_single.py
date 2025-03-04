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
    logging.info(vars(args))
    logging.info("device: {}".format(device))

    with torch.no_grad():
        logging.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model = checkpoint["model"]
        model.to(device)
        model.eval()
        model.device = device

        logging.info(f"Loading BPE model from {args.bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(args.bpe_model)

        logging.info(f"Reading sound file: {args.audio_file}")
        wave = load_audio(args.audio_file)
        sample_rate = wave.sample_rate
        samples = wave.samples.to(device)

        logging.info(f"Original sample rate: {sample_rate}")
        if sample_rate != 16000:
            logging.info(f"Resampling to 16000 Hz")
            samples = torchaudio.functional.resample(
                samples, sample_rate, 16000
            )
            sample_rate = 16000

        # Convert to mono if needed
        if len(samples.shape) > 1 and samples.shape[0] > 1:
            logging.info(f"Converting multi-channel audio to mono")
            samples = torch.mean(samples, dim=0, keepdim=True)

        # Normalize audio if values are outside [-1, 1]
        max_value = torch.max(torch.abs(samples))
        if max_value > 1.0:
            logging.info(f"Normalizing audio (max value: {max_value:.3f})")
            samples = samples / max_value

        logging.info(f"Normalized audio shape: {samples.shape}, range: [{samples.min():.3f}, {samples.max():.3f}], std: {samples.std():.3f}")
        
        # Move samples to the correct device
        samples = samples.to(device)

        logging.info("Encoder type: {}".format(model.encoder.__class__.__name__))
        logging.info("Decoder type: {}".format(model.decoder.__class__.__name__))
        logging.info("Joiner type: {}".format(model.joiner.__class__.__name__))

        use_streaming = args.chunk_size > 0 and args.left_context_chunks > 0
        if use_streaming:
            logging.info(f"Using streaming mode with chunk size: {args.chunk_size}")
            encoder_out = process_streaming_chunks(
                model=model,
                features=samples,
                chunk_size=args.chunk_size,
                left_context_chunks=args.left_context_chunks,
                attention_sink_size=args.attention_sink_size,
                device=device,
            )
        else:
            logging.info("Using non-streaming mode (processing full audio without chunking)")
            encoder_out = model.encode_audio(samples.unsqueeze(0))

        logging.info(f"Encoder output shape: {encoder_out.shape}")
        
        # Use a moderate blank penalty consistent with training
        blank_penalty = 0.5
        
        # Add a repetition penalty function to the beam search
        def modified_beam_search_with_repetition_penalty(
            model, 
            encoder_out, 
            encoder_out_lens, 
            beam=4, 
            temperature=1.0, 
            blank_penalty=0.5
        ):
            import copy
            import functools
            from beam_search import modified_beam_search
            
            # Make a copy of the model for our modified version
            model_copy = copy.deepcopy(model)
            
            # Override the joiner forward method to add repetition penalty
            original_joiner = model_copy.joiner
            
            @functools.wraps(original_joiner.__call__)
            def joiner_with_penalty(encoder_out, decoder_out, project_input=True):
                # Get original logits
                logits = original_joiner(encoder_out, decoder_out, project_input)
                
                # In shape (batch, decoder_dim1, decoder_dim2, vocab_size)
                if logits.dim() == 4 and decoder_out is not None:
                    batch_size = logits.size(0)
                    
                    # For each item in the batch
                    for i in range(batch_size):
                        # Get the decoder input (context)
                        # This contains the previous tokens
                        if hasattr(decoder_out, 'decoder_input') and decoder_out.decoder_input is not None:
                            prev_tokens = decoder_out.decoder_input[i]
                            
                            # If we have at least 3 previous tokens
                            if prev_tokens.size(0) >= 3:
                                # Check if the last 3 tokens are the same
                                last_3 = prev_tokens[-3:]
                                if torch.all(last_3[0] == last_3[1]) and torch.all(last_3[0] == last_3[2]):
                                    # Apply a penalty to that token
                                    repeated_token = last_3[0].item()
                                    logits[i, :, :, repeated_token] -= 8.0  # Apply a significant penalty
                
                return logits
            
            # Patch the model's joiner
            try:
                model_copy.joiner.__call__ = joiner_with_penalty
                
                # Call the original modified_beam_search with our patched model
                return modified_beam_search(
                    model=model_copy,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    beam=beam,
                    temperature=temperature,
                    blank_penalty=blank_penalty
                )
            finally:
                # Clean up to avoid memory leaks
                del model_copy
        
        # Define a simple function to count consecutive repetitions
        def count_repetitions(token_ids):
            if not token_ids:
                return 0
                
            max_reps = 0
            current_reps = 1
            prev_token = token_ids[0]
            
            for token in token_ids[1:]:
                if token == prev_token:
                    current_reps += 1
                else:
                    max_reps = max(max_reps, current_reps)
                    current_reps = 1
                prev_token = token
                
            return max(max_reps, current_reps)
            
        # Use a length penalty to prevent excessive repetition
        max_output_length = 50  # Set a reasonable max length
        
        logging.info(f"Using blank penalty: {blank_penalty}")
        
        beam_size = args.beam_size
        logging.info(f"Using beam size: {beam_size}")
        
        encoder_out_lens = torch.tensor([encoder_out.shape[1]], device=device)
        
        try:
            # First try with our custom beam search with repetition penalty
            hyps = beam_search.modified_beam_search(
                model=model,
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=beam_size,
                temperature=1.0,
                blank_penalty=blank_penalty,
            )
            
            # Log the raw token IDs for debugging
            logging.info(f"Raw token IDs: {hyps[0]}")
            
            # Check for excessive repetition
            rep_count = count_repetitions(hyps[0])
            
            # If excessive repetition detected, try increasingly aggressive measures
            if rep_count > 5:
                logging.warning(f"Excessive repetition detected ({rep_count} repeats), trying higher blank penalty")
                hyps = beam_search.modified_beam_search(
                    model=model,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    beam=beam_size,
                    temperature=1.0,
                    blank_penalty=2.0,  # Increase blank penalty
                )
                
                rep_count = count_repetitions(hyps[0])
                
                # If still repetitive, limit output length
                if rep_count > 5:
                    logging.warning(f"Still excessive repetition ({rep_count} repeats), limiting output length")
                    hyps = [hyps[0][:max_output_length]]
            
            # Convert token IDs to text
            text = sp.decode(hyps[0])
            
        except Exception as e:
            logging.error(f"Error during decoding: {e}")
            text = ""
            
        logging.info(f"Transcription: {text}")
        print(f"Transcription: {text}")

if __name__ == "__main__":
    main() 