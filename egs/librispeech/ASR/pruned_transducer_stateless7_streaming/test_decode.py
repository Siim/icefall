#!/usr/bin/env python3

import argparse
import logging
import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple
import numpy as np

from xlsr_encoder import XLSREncoder
from estonian_decoder import create_estonian_token_table, create_estonian_decoding_graph

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-file",
        type=str,
        required=True,
        help="Path to the audio file to test",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        required=True,
        help="Path to the vocabulary file",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8000,  # 0.5s at 16kHz
        help="Chunk size for streaming test",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot chunk outputs for visualization",
    )
    return parser.parse_args()

def load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Load and preprocess audio file."""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Ensure shape is (batch, time)
    waveform = waveform.transpose(0, 1)  # (time, channel) -> (channel, time)
    waveform = waveform.unsqueeze(0)  # Add batch dimension
    waveform = waveform.squeeze(1)  # Remove channel dimension
    
    return waveform, 16000

def plot_outputs(streaming_out: torch.Tensor, non_streaming_out: torch.Tensor, save_path: str = "chunk_comparison.png"):
    """Plot streaming vs non-streaming outputs for visualization"""
    import matplotlib.pyplot as plt
    
    # Take mean across feature dimension
    streaming_mean = streaming_out[0].mean(dim=1).cpu().numpy()
    non_streaming_mean = non_streaming_out[0].mean(dim=1).cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    plt.plot(streaming_mean, label='Streaming', alpha=0.7)
    plt.plot(non_streaming_mean, label='Non-streaming', alpha=0.7)
    plt.title('Streaming vs Non-streaming Output Comparison')
    plt.xlabel('Time frames')
    plt.ylabel('Mean activation')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    args = get_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    # Initialize XLSR encoder
    encoder = XLSREncoder(
        model_name="facebook/wav2vec2-xls-r-300m",
        decode_chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_size // 2,
        use_attention_sink=True
    )
    
    # Create token table and decoding graph
    token_table = create_estonian_token_table(args.vocab_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoding_graph = create_estonian_decoding_graph(
        token_table=token_table,
        num_tokens=len(token_table),
        device=device
    )
    
    # Move encoder to device
    encoder = encoder.to(device)
    encoder.eval()
    
    # Load and preprocess audio
    audio, sample_rate = load_audio(args.audio_file)
    audio = audio.to(device)
    audio_len = torch.tensor([audio.shape[1]], dtype=torch.int32, device=device)
    
    logging.info(f"Testing streaming decoding with chunk size {args.chunk_size} samples ({args.chunk_size/16000:.3f}s)")
    logging.info(f"Audio duration: {audio.shape[1]/16000:.2f}s")
    
    # Test streaming decoding
    with torch.no_grad():
        # First get non-streaming output for reference
        non_streaming_out, non_streaming_lens = encoder(audio, audio_len)
        logging.info(f"Non-streaming output shape: {non_streaming_out.shape}")
        
        # Initialize streaming state
        states = encoder.get_init_state(device)
        
        # Process audio in chunks
        current = 0
        encoder_out_chunks = []
        chunk_overlap = args.chunk_size // 2
        
        # Calculate expected number of chunks
        total_samples = audio.shape[1]
        effective_chunk_size = args.chunk_size - chunk_overlap
        num_chunks = int(np.ceil(total_samples / effective_chunk_size))
        logging.info(f"Expected number of chunks: {num_chunks}")
        
        while current < audio.shape[1]:
            end = min(current + args.chunk_size, audio.shape[1])
            chunk = audio[:, current:end]
            chunk_len = torch.tensor([chunk.shape[1]], dtype=torch.int32, device=device)
            
            # Process chunk
            chunk_out, chunk_lens, states = encoder.streaming_forward(chunk, chunk_len, states)
            
            # For all chunks except the last, remove overlap frames from the end
            if end < audio.shape[1]:
                overlap_frames = chunk_overlap // encoder.downsample_factor
                chunk_out = chunk_out[:, :-overlap_frames]
            
            encoder_out_chunks.append(chunk_out)
            logging.info(f"Processed chunk {len(encoder_out_chunks)}/{num_chunks}: {chunk.shape[1]/16000:.3f}s")
            
            # Move to next chunk, considering overlap
            if end == audio.shape[1]:  # Last chunk
                break
            current = end - chunk_overlap
        
        # Concatenate chunks
        encoder_out = torch.cat(encoder_out_chunks, dim=1)
        logging.info(f"Final encoder output shape: {encoder_out.shape}")
        
        # Verify output dimensions
        expected_frames = ((audio_len.float() / encoder.downsample_factor).floor() - 1).to(torch.int64)
        logging.info(f"Expected output frames: {expected_frames.item()}")
        logging.info(f"Got output frames: {encoder_out.shape[1]}")
        
        # Compare outputs
        min_len = min(encoder_out.shape[1], non_streaming_out.shape[1])
        streaming_out = encoder_out[:, :min_len]
        non_streaming_out = non_streaming_out[:, :min_len]
        
        # Calculate differences
        abs_diff = (streaming_out - non_streaming_out).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Calculate per-feature differences
        feature_diffs = abs_diff.mean(dim=(0, 1))  # Average across batch and time
        worst_features = torch.topk(feature_diffs, k=5)
        
        logging.info(f"Max difference between streaming and non-streaming: {max_diff:.6f}")
        logging.info(f"Mean difference between streaming and non-streaming: {mean_diff:.6f}")
        logging.info("Top 5 most different features:")
        for i, (idx, diff) in enumerate(zip(worst_features.indices, worst_features.values)):
            logging.info(f"  Feature {idx}: {diff:.6f}")
        
        # Plot outputs if requested
        if args.plot:
            plot_outputs(streaming_out, non_streaming_out)
            logging.info("Saved output visualization to chunk_comparison.png")

if __name__ == "__main__":
    main() 