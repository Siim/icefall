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
    parser.add_argument(
        "--cpu-only",
        type=str2bool,
        default=False,
        help="Run entirely on CPU to avoid GPU memory issues",
    )
    parser.add_argument(
        "--skip-non-streaming",
        type=str2bool,
        default=False,
        help="Skip non-streaming inference to save memory",
    )
    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Use half precision to reduce memory usage",
    )
    parser.add_argument(
        "--plot-only-streaming",
        type=str2bool,
        default=False,
        help="Plot only streaming output (no comparison)",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=0,  # 0 means use full audio
        help="Maximum audio length in seconds (0 = use full audio)",
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

def plot_outputs(streaming_out, non_streaming_out, save_path="chunk_comparison.png"):
    """Plot streaming vs non-streaming outputs for visualization"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Log what we're plotting
    logging.info(f"Plotting comparison: streaming shape={streaming_out.shape}, non-streaming shape={non_streaming_out.shape}")
    
    # Take mean across feature dimension and detach before converting to numpy
    streaming_mean = streaming_out[0].mean(dim=1).detach().cpu().numpy()
    non_streaming_mean = non_streaming_out[0].mean(dim=1).detach().cpu().numpy()
    
    # Create x-axis values that represent the same time scale
    x_streaming = np.linspace(0, 1, streaming_mean.shape[0])
    x_non_streaming = np.linspace(0, 1, non_streaming_mean.shape[0])
    
    logging.info(f"Mean values - streaming: min={streaming_mean.min():.4f}, max={streaming_mean.max():.4f}")
    logging.info(f"Mean values - non-streaming: min={non_streaming_mean.min():.4f}, max={non_streaming_mean.max():.4f}")
    
    plt.figure(figsize=(15, 5))
    plt.plot(x_streaming, streaming_mean, label='Streaming', alpha=0.7)
    plt.plot(x_non_streaming, non_streaming_mean, label='Non-streaming', alpha=0.7)
    plt.title('Streaming vs Non-streaming Output Comparison')
    plt.xlabel('Normalized time')
    plt.ylabel('Mean activation')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    logging.info(f"Plot saved to {save_path}")
    plt.close()

def plot_streaming_only(streaming_out, save_path="streaming_output.png"):
    """Plot only streaming outputs without comparison."""
    import matplotlib.pyplot as plt
    
    # Log what we're plotting
    logging.info(f"Plotting streaming output only: shape={streaming_out.shape}")
    
    # Take mean across feature dimension and detach before converting to numpy
    streaming_mean = streaming_out[0].mean(dim=1).detach().cpu().numpy()
    
    logging.info(f"Mean values - streaming: min={streaming_mean.min():.4f}, max={streaming_mean.max():.4f}")
    
    plt.figure(figsize=(15, 5))
    plt.plot(streaming_mean, label='Streaming', alpha=0.9)
    plt.title('Streaming Output Visualization')
    plt.xlabel('Time frames')
    plt.ylabel('Mean activation')
    plt.grid(True)
    plt.savefig(save_path)
    logging.info(f"Plot saved to {save_path}")
    plt.close()

def main():
    args = get_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    # Load and preprocess audio
    waveform, sample_rate = load_audio(args.audio_path)
    
    # Optionally trim audio to reduce memory requirements
    if args.max_audio_seconds > 0:
        max_samples = int(args.max_audio_seconds * sample_rate)
        if waveform.shape[1] > max_samples:
            logging.info(f"Trimming audio from {waveform.shape[1]/sample_rate:.2f}s to {args.max_audio_seconds:.2f}s")
            waveform = waveform[:, :max_samples]
    
    # Create processor and model
    processor = create_processor(args.vocab_file)
    model = create_model(args.model_name)
    
    # Process audio input properly
    input_values = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_values
    
    # Log audio length
    audio_seconds = input_values.shape[1] / sample_rate
    logging.info(f"Processing audio of length: {audio_seconds:.2f} seconds ({input_values.shape[1]} samples)")
    
    # Create encoder for streaming
    encoder = XLSREncoder(
        model_name=args.model_name,
        decode_chunk_size=args.chunk_size,
        use_attention_sink=args.use_attention_sink
    )
    
    # Set to eval mode
    model.eval()
    encoder.eval()
    
    # Choose device based on arguments and availability
    if args.cpu_only:
        device = torch.device("cpu")
        logging.info("Running on CPU as requested")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logging.info(f"Using device: {device}")
    
    # Move to device and optionally convert to half precision
    if args.use_fp16 and device.type != "cpu":
        model = model.half()
        encoder = encoder.half()
        input_values = input_values.half()
        logging.info("Using FP16 precision to reduce memory usage")
    
    model = model.to(device)
    encoder = encoder.to(device)
    input_values = input_values.to(device)
    
    # Add memory tracking and management
    if device.type == "cuda":
        logging.info(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Process audio
    with torch.no_grad():
        # Free up memory before starting
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        non_streaming_out = None
        non_streaming_text = "Non-streaming inference skipped to save memory"
        
        # Run non-streaming inference only if not skipped
        if not args.skip_non_streaming:
            try:
                # First get non-streaming output for reference
                non_streaming_out, non_streaming_lens = encoder(input_values, torch.tensor([input_values.shape[1]], device=device))
                if device.type == "cuda":
                    logging.info(f"GPU memory after non-streaming: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
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
                
                # Free memory after non-streaming inference
                del non_streaming_logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    logging.info(f"GPU memory after clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            except RuntimeError as e:
                logging.error(f"Error during non-streaming inference: {e}")
                logging.info("Continuing with streaming-only inference")
                non_streaming_out = None
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            logging.info("Skipping non-streaming inference as requested")
        
        # Initialize streaming with warmup
        states = encoder.get_init_state(device)
        warmup_size = args.chunk_size  # Use one full chunk for warmup
        
        # Process audio in chunks
        current_pos = 0
        all_streaming_logits = []
        streaming_outputs = []  # For text output
        streaming_encodings = []  # For encoder outputs (needed for plotting)
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
            
            # Skip if we've reached the end
            if chunk_start >= input_values.shape[1]:
                logging.info(f"Skipping chunk {i+1} as we've reached the end of input")
                continue
            
            # Extract chunk
            chunk = input_values[:, chunk_start:chunk_end]
            chunk_len = torch.tensor([chunk.shape[1]], device=device)
            
            try:
                # Process chunk through encoder and model
                chunk_out, chunk_lens, states = encoder.streaming_forward(chunk, chunk_len, states)
                chunk_logits = model.forward(chunk).logits
                
                # Save all logits and encoder outputs for later use
                all_streaming_logits.append(chunk_logits.cpu())  # For full decoding
                streaming_encodings.append(chunk_out.cpu())  # For plotting
                
                # Show progressive output if requested
                if args.show_progressive:
                    logging.info(f"\nChunk {i+1}/{num_chunks} decoding:")
                    chunk_text = decode_ctc(chunk_logits, processor)
                    streaming_outputs.append(chunk_text)  # Only text
                    logging.info(f"Chunk text: {chunk_text}")
                else:
                    logging.info(f"Processed chunk {i+1}/{num_chunks}: {chunk_end/sample_rate:.3f}s")
                
                # Move to next chunk position with overlap
                current_pos = chunk_start + effective_chunk_size
                
                # Clear memory after each chunk
                del chunk_out
                del chunk_logits
                del chunk
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                
                if i % 5 == 0 and device.type == "cuda":
                    logging.info(f"Current GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    
            except RuntimeError as e:
                logging.error(f"Error processing chunk {i+1}: {e}")
                logging.error("Skipping to next chunk")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        
        # After all chunks are processed, concatenate all logits
        if all_streaming_logits:
            logging.info("\nDecoding combined streaming logits...")
            combined_logits = torch.cat(all_streaming_logits, dim=1)
            # Move back to device for decoding
            combined_logits = combined_logits.to(device)
            full_streaming_text = decode_ctc(combined_logits, processor)
            logging.info("\nFull streaming transcript from combined chunks:")
            logging.info("-" * 50)
            logging.info(f"{full_streaming_text}")
            logging.info("-" * 50)
        
        # Compare outputs if both are available
        if non_streaming_out is not None:
            # Move to CPU for analysis to save GPU memory
            non_streaming_out = non_streaming_out.cpu()
            
            # Basic comparison - use streaming_encodings instead of streaming_outputs
            if streaming_encodings:
                # Concatenate streaming encodings for comparison
                streaming_concat = torch.cat(streaming_encodings, dim=1).cpu()
                expected_frames = non_streaming_out.shape[1]
                actual_frames = streaming_concat.shape[1]
                logging.info(f"Expected output frames: {expected_frames}")
                logging.info(f"Got output frames: {actual_frames}")
                
                # Calculate differences on small sections to avoid memory issues
                try:
                    # Calculate differences on downsized samples if needed
                    sample_rate = 10  # Take every 10th frame
                    
                    # Ensure we're comparing the same number of frames
                    min_len = min(actual_frames, expected_frames)
                    samples = min(min_len // sample_rate, 100)  # Cap at 100 sampled frames for memory
                    
                    if min_len > sample_rate:
                        # Sample both tensors at the same rate and limit to the same number of frames
                        s_sample = streaming_concat[:, ::sample_rate, :][:, :samples, :]
                        ns_sample = non_streaming_out[:, ::sample_rate, :][:, :samples, :]
                        
                        # Verify shapes match before comparison
                        logging.info(f"Sampled shapes - streaming: {s_sample.shape}, non-streaming: {ns_sample.shape}")
                        
                        if s_sample.shape[1] == ns_sample.shape[1]:
                            abs_diff = (s_sample - ns_sample).abs()
                            max_diff = abs_diff.max().item()
                            mean_diff = abs_diff.mean().item()
                            
                            # Calculate per-feature differences
                            feature_diffs = abs_diff.mean(dim=(0, 1))  # Average across batch and time
                            worst_features = torch.topk(feature_diffs, k=min(5, feature_diffs.shape[0]))
                            
                            logging.info(f"\nOutput Differences (sampled):")
                            logging.info("-" * 50)
                            logging.info(f"Max difference between streaming and non-streaming: {max_diff:.6f}")
                            logging.info(f"Mean difference between streaming and non-streaming: {mean_diff:.6f}")
                            logging.info("Top most different features:")
                            for i, (idx, diff) in enumerate(zip(worst_features.indices, worst_features.values)):
                                logging.info(f"  Feature {idx}: {diff:.6f}")
                        else:
                            logging.warning(f"Cannot compare tensors with different shapes after sampling")
                    
                    # Plot outputs if requested and we have non-streaming outputs to compare
                    if args.plot and not args.plot_only_streaming:
                        try:
                            # Move to CPU if not already there
                            non_streaming_cpu = non_streaming_out
                            streaming_cpu = streaming_concat
                            
                            # Limit data for plotting to avoid memory issues
                            max_frames = 1000
                            
                            # Check for empty streaming frames
                            if streaming_cpu.shape[1] == 0:
                                logging.warning("Empty streaming output detected. Creating minimal placeholder for plotting.")
                                # Create a placeholder tensor with the same feature dimension but 1 frame
                                streaming_cpu = torch.zeros((streaming_cpu.shape[0], 1, streaming_cpu.shape[2]), device=streaming_cpu.device)
                                
                            # Resample both to the same number of frames for plotting
                            ns_sample_rate = max(1, non_streaming_cpu.shape[1] // max_frames)
                            s_sample_rate = max(1, streaming_cpu.shape[1] // max_frames)
                            
                            ns_plot = non_streaming_cpu[:, ::ns_sample_rate, :]
                            s_plot = streaming_cpu[:, ::s_sample_rate, :]
                            
                            # Interpolate to make them the same length
                            logging.info(f"Resampling for plot - streaming: {s_plot.shape[1]} frames, non-streaming: {ns_plot.shape[1]} frames")
                            
                            # Use the smaller of the two as the target size
                            target_frames = min(s_plot.shape[1], ns_plot.shape[1])
                            
                            # Resample if needed
                            if s_plot.shape[1] != target_frames:
                                s_plot = s_plot[:, :target_frames, :]
                            if ns_plot.shape[1] != target_frames:
                                ns_plot = ns_plot[:, :target_frames, :]
                            
                            logging.info(f"Final plot shapes - streaming: {s_plot.shape}, non-streaming: {ns_plot.shape}")
                            
                            # Plot the data
                            plot_outputs(s_plot, ns_plot)
                            
                            # Clean up
                            del s_plot, ns_plot
                            
                        except Exception as e:
                            logging.error(f"Error during plotting: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                    
                    # Clean up
                    del streaming_concat
                    
                except Exception as e:
                    logging.error(f"Error during comparison: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
        
        # Final cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
            logging.info(f"Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Plot streaming-only output if requested
    if args.plot and (args.plot_only_streaming or args.skip_non_streaming) and streaming_encodings:
        try:
            logging.info("Generating streaming-only plot...")
            streaming_concat = torch.cat(streaming_encodings, dim=1).cpu()
            plot_streaming_only(streaming_concat)
        except Exception as e:
            logging.error(f"Error during streaming-only plotting: {e}")
            import traceback
            logging.error(traceback.format_exc())

if __name__ == "__main__":
    main() 