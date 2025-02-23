#!/usr/bin/env python3

import argparse
import logging
import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from icefall.utils import str2bool
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

from xlsr_encoder import XLSREncoder

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="TalTechNLP/xls-r-300m-et",
        help="Model name or path",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to audio file",
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
        default=5120,  # 320ms at 16kHz
        help="Chunk size for streaming inference",
    )
    parser.add_argument(
        "--use-attention-sink",
        type=str2bool,
        default=True,
        help="Whether to use attention sink",
    )
    parser.add_argument(
        "--plot",
        type=str2bool,
        default=True,
        help="Whether to plot output comparison",
    )
    parser.add_argument(
        "--show-progressive",
        type=str2bool,
        default=False,
        help="Show decoded text for each chunk",
    )
    return parser.parse_args()

def load_audio(audio_path: str) -> Tuple[torch.Tensor, int]:
    """Load and preprocess audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, sample_rate

def create_processor(vocab_file: str) -> Wav2Vec2Processor:
    """Create Wav2Vec2 processor from TalTechNLP's Estonian model.
    
    Args:
        vocab_file: Path to vocabulary file (not used, kept for compatibility)
        
    Returns:
        Wav2Vec2Processor instance
    """
    # Load TalTechNLP's Estonian processor
    processor = Wav2Vec2Processor.from_pretrained("TalTechNLP/xls-r-300m-et")
    return processor

def create_model(model_name: str) -> Wav2Vec2ForCTC:
    """Create Wav2Vec2ForCTC model with CTC head.
    
    Args:
        model_name: Name or path of the model to load
        
    Returns:
        Wav2Vec2ForCTC model instance
    """
    # Load model with CTC head
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    # Disable feature extraction updates
    model.freeze_feature_encoder()
    
    return model

def decode_ctc(logits: torch.Tensor, processor: Wav2Vec2Processor) -> str:
    """Decode CTC logits to text using the processor's built-in decoding.
    
    Args:
        logits: Output logits from model, shape (batch, time, vocab_size)
        processor: Wav2Vec2Processor instance
        
    Returns:
        Decoded text string
    """
    # Get predicted ids directly from logits
    pred_ids = torch.argmax(logits, dim=-1)
    
    # Use processor's built-in decoding
    transcription = processor.decode(pred_ids[0])  # Take first batch item
    
    return transcription

def plot_outputs(streaming_out: torch.Tensor, non_streaming_out: torch.Tensor, save_path: str = "chunk_comparison.png"):
    """Plot streaming vs non-streaming outputs for visualization"""
    import matplotlib.pyplot as plt
    
    # Take mean across feature dimension and detach before converting to numpy
    streaming_mean = streaming_out[0].mean(dim=1).detach().cpu().numpy()
    non_streaming_mean = non_streaming_out[0].mean(dim=1).detach().cpu().numpy()
    
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
    
    # Load and preprocess audio
    waveform, sample_rate = load_audio(args.audio_path)
    
    # Create processor and model
    processor = create_processor(args.vocab_file)
    model = create_model(args.model_name)
    
    # Process audio input properly
    input_values = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_values
    
    # Create encoder for streaming
    encoder = XLSREncoder(
        model_name=args.model_name,
        decode_chunk_size=args.chunk_size,
        use_attention_sink=args.use_attention_sink
    )
    
    # Set to eval mode
    model.eval()
    encoder.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    encoder = encoder.to(device)
    input_values = input_values.to(device)
    
    # Process audio
    with torch.no_grad():
        # First get non-streaming output for reference
        non_streaming_out, non_streaming_lens = encoder(input_values, torch.tensor([input_values.shape[1]], device=device))
        
        # Get CTC logits from model
        non_streaming_logits = model.forward(input_values).logits
        
        logging.info(f"Non-streaming output shape: {non_streaming_out.shape}")
        
        # Decode non-streaming output
        logging.info("\nNon-streaming decoding:")
        non_streaming_text = decode_ctc(non_streaming_logits, processor)
        logging.info("\nNon-streaming decoded text:")
        logging.info("-" * 50)
        logging.info(f"{non_streaming_text}")
        logging.info("-" * 50)
        
        # Initialize streaming with warmup
        states = encoder.get_init_state(device)
        warmup_size = args.chunk_size  # Use one full chunk for warmup
        
        # Process audio in chunks
        current_pos = 0
        streaming_chunks = []
        streaming_logits = []
        chunk_size = args.chunk_size
        chunk_overlap = chunk_size // 2
        effective_chunk_size = chunk_size - chunk_overlap
        
        # Calculate number of chunks
        num_chunks = int(np.ceil((input_values.shape[1] - warmup_size) / effective_chunk_size))
        logging.info(f"\nProcessing {num_chunks} chunks with warmup...")
        
        # Warmup phase
        if warmup_size > 0:
            warmup_chunk = input_values[:, :warmup_size]
            warmup_len = torch.tensor([warmup_chunk.shape[1]], device=device)
            _, _, states = encoder.streaming_forward(warmup_chunk, warmup_len, states)
            logging.info("Completed warmup phase")
            current_pos = warmup_size - chunk_overlap  # Start with overlap from warmup
        
        for i in range(num_chunks):
            # Get chunk boundaries
            chunk_start = current_pos
            chunk_end = min(chunk_start + chunk_size, input_values.shape[1])
            
            # Extract chunk
            chunk = input_values[:, chunk_start:chunk_end]
            chunk_len = torch.tensor([chunk.shape[1]], device=device)
            
            # Process chunk through encoder and model
            chunk_out, chunk_lens, states = encoder.streaming_forward(chunk, chunk_len, states)
            chunk_logits = model.forward(chunk).logits
            
            # Only append after warmup
            if i > 0 or warmup_size == 0:
                streaming_chunks.append(chunk_out)
                streaming_logits.append(chunk_logits)
            
            # Show progressive output if requested
            if args.show_progressive:
                logging.info(f"\nChunk {i+1}/{num_chunks} decoding:")
                chunk_text = decode_ctc(chunk_logits, processor)
                logging.info(f"Chunk text: {chunk_text}")
            else:
                logging.info(f"Processed chunk {i+1}/{num_chunks}: {chunk_end/sample_rate:.3f}s")
            
            # Move to next chunk position, considering overlap
            current_pos = chunk_start + effective_chunk_size
            
            # Clear GPU cache periodically
            if i % 10 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate streaming outputs
        streaming_out = torch.cat(streaming_chunks, dim=1)
        streaming_logits = torch.cat(streaming_logits, dim=1)
        logging.info(f"\nFinal encoder output shape: {streaming_out.shape}")
        
        # Decode streaming output
        logging.info("\nStreaming decoding:")
        streaming_text = decode_ctc(streaming_logits, processor)
        logging.info("\nStreaming decoded text:")
        logging.info("-" * 50)
        logging.info(f"{streaming_text}")
        logging.info("-" * 50)

        # Compare outputs
        expected_frames = non_streaming_out.shape[1]
        actual_frames = streaming_out.shape[1]
        logging.info(f"Expected output frames: {expected_frames}")
        logging.info(f"Got output frames: {actual_frames}")

        # Calculate differences
        min_len = min(streaming_out.shape[1], non_streaming_out.shape[1])
        streaming_out = streaming_out[:, :min_len, :]
        non_streaming_out = non_streaming_out[:, :min_len, :]
        
        abs_diff = (streaming_out - non_streaming_out).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        
        # Calculate per-feature differences
        feature_diffs = abs_diff.mean(dim=(0, 1))  # Average across batch and time
        worst_features = torch.topk(feature_diffs, k=5)
        
        logging.info(f"\nOutput Differences:")
        logging.info("-" * 50)
        logging.info(f"Max difference between streaming and non-streaming: {max_diff:.6f}")
        logging.info(f"Mean difference between streaming and non-streaming: {mean_diff:.6f}")
        logging.info("Top 5 most different features:")
        for i, (idx, diff) in enumerate(zip(worst_features.indices, worst_features.values)):
            logging.info(f"  Feature {idx}: {diff:.6f}")
        
        # Plot outputs if requested
        if args.plot:
            plot_outputs(streaming_out, non_streaming_out)
            logging.info("\nSaved output visualization to chunk_comparison.png")

        # Clean up
        del streaming_chunks
        del streaming_logits
        del streaming_out
        del non_streaming_out
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 