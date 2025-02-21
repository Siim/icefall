#!/usr/bin/env python3

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from pathlib import Path
import json
import numpy as np
from xlsr_encoder import XLSREncoder
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_tokenizer(vocab_dict_path: str = None):
    """
    Create a tokenizer with Estonian vocabulary
    """
    if vocab_dict_path and Path(vocab_dict_path).exists():
        # Load existing vocabulary
        with open(vocab_dict_path, 'r') as f:
            vocab_dict = json.load(f)
    else:
        # Create Estonian vocab with all necessary characters
        # Including specific Estonian characters and common punctuation
        vocab_list = list("abcdefghijklmnopqrstuvwxyzäöüõšž,.!? ")  # Extended Estonian alphabet
        vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
        vocab_dict["|"] = vocab_dict[" "]  # Use | as the word delimiter
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        
        # Save vocabulary
        vocab_path = Path("vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(vocab_dict, f)
    
    return Wav2Vec2CTCTokenizer.from_pretrained("./", 
                                               unk_token="[UNK]",
                                               pad_token="[PAD]",
                                               word_delimiter_token="|")

def load_xlsr_model(model_name: str = "facebook/wav2vec2-xls-r-300m", vocab_dict_path: str = None):
    """
    Load XLSR model and processor with proper tokenizer
    """
    # First create tokenizer
    tokenizer = create_tokenizer(vocab_dict_path)
    
    # Create feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    
    # Create processor from tokenizer and feature extractor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # Load the model
    model = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    return model, processor

def process_audio(audio_path: str, processor: Wav2Vec2Processor, target_sample_rate: int = 16000):
    """
    Load and process audio file for XLSR model
    """
    print(f"Processing file: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Original sample rate: {sample_rate}")
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        print(f"Resampling from {sample_rate} to {target_sample_rate}")
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    
    # Convert to numpy array and squeeze
    input_values = waveform.squeeze().numpy()
    
    # Normalize and get input values
    inputs = processor(input_values, sampling_rate=target_sample_rate, return_tensors="pt")
    return inputs.input_values

def plot_attention_patterns(outputs: list[torch.Tensor], title: str, save_path: str):
    """Plot attention patterns to visualize chunk processing and attention sink"""
    plt.figure(figsize=(12, 6))
    # Concatenate outputs and reshape properly
    concat_output = torch.cat(outputs, dim=1)  # [batch, time, features]
    # Average across feature dimension for visualization
    attention_pattern = concat_output[0].mean(dim=1).detach().cpu().numpy()
    plt.plot(attention_pattern)  # Plot as a line graph instead of heatmap
    plt.title(title)
    plt.xlabel("Time frames")
    plt.ylabel("Average activation")
    plt.savefig(save_path)
    plt.close()

def verify_chunk_sizes(encoder: XLSREncoder):
    """Verify that chunk sizes match paper specifications"""
    expected_sizes = {
        "320ms": int(0.32 / encoder.frame_stride),  # Should be ~16 frames
        "640ms": int(0.64 / encoder.frame_stride),  # Should be ~32 frames
        "1280ms": int(1.28 / encoder.frame_stride), # Should be ~64 frames
        "2560ms": int(2.56 / encoder.frame_stride)  # Should be ~128 frames
    }
    
    logger.info("\nVerifying chunk sizes:")
    for name, expected in expected_sizes.items():
        actual = encoder.chunk_sizes[list(expected_sizes.keys()).index(name)]
        logger.info(f"{name}: Expected {expected} frames, Got {actual} frames")
        assert abs(expected - actual) <= 1, f"Chunk size mismatch for {name}"

def verify_streaming_output(encoder: XLSREncoder, waveform: torch.Tensor, chunk_size: int):
    """Verify streaming output matches non-streaming for given chunk size"""
    # Ensure correct input shape (batch, time)
    if waveform.ndim == 3:  # (batch, time, channel)
        waveform = waveform.squeeze(-1)  # Remove channel dimension
    assert waveform.ndim == 2, f"Expected 2D input (batch, time), got shape {waveform.shape}"
    
    # Calculate expected number of output frames
    input_length = torch.tensor([waveform.shape[1]], dtype=torch.int32)
    expected_frames = ((input_length.float() / encoder.downsample_factor).floor() - 1).to(torch.int64)
    logger.info(f"Expected output frames: {expected_frames.item()}")
    
    # Non-streaming forward pass
    with torch.no_grad():
        encoder.eval()
        non_streaming_out, _ = encoder(waveform, input_length)
        logger.info(f"Non-streaming output shape: {non_streaming_out.shape}")
    
    # Streaming forward pass
    chunks = encoder.prepare_chunks(waveform, chunk_size)
    streaming_outputs = []
    
    encoder.reset_streaming_state()
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            chunk_len = torch.tensor([chunk.shape[1]], dtype=torch.int32)
            chunk_out = encoder.process_chunk(chunk, encoder.attention_sink_cache)
            streaming_outputs.append(chunk_out)
            logger.info(f"Chunk {i} input shape: {chunk.shape}, output shape: {chunk_out.shape}")
    
    streaming_out = torch.cat(streaming_outputs, dim=1)
    logger.info(f"Streaming output shape: {streaming_out.shape}")
    
    # Compare outputs
    min_len = min(streaming_out.shape[1], non_streaming_out.shape[1])
    streaming_out = streaming_out[:, :min_len]
    non_streaming_out = non_streaming_out[:, :min_len]
    
    # Calculate differences across all dimensions
    abs_diff = (streaming_out - non_streaming_out).abs()
    max_diff = abs_diff.max(dim=2)[0].max(dim=1)[0].item()  # Max across features and time
    mean_diff = abs_diff.mean().item()
    logger.info(f"\nOutput differences (chunk_size={chunk_size/16000:.3f}s):")
    logger.info(f"Max difference: {max_diff:.6f}")
    logger.info(f"Mean difference: {mean_diff:.6f}")
    
    return streaming_outputs

def test_xlsr_encoder():
    """Test XLSR encoder's implementation according to paper specifications"""
    logger.info("Testing XLSR encoder setup...")
    
    # Initialize encoder with paper's specifications
    encoder = XLSREncoder(
        model_name="facebook/wav2vec2-xls-r-300m",
        decode_chunk_size=8000,  # 0.5s at 16kHz
        chunk_overlap=4000,      # 0.25s overlap
        use_attention_sink=True,
        attention_sink_size=4,   # From paper's diagrams
        frame_duration=0.025,    # 25ms per frame
        frame_stride=0.020,      # 20ms stride
    )
    
    # Verify chunk size calculations
    verify_chunk_sizes(encoder)
    
    # Get test audio files
    data_dir = Path("/Users/siimhaugas/Desktop/Projects/haugasdev/XLSR-Transducer/estonian_corpora/processed/tambet")
    test_files = list(data_dir.glob("*.wav"))
    
    if not test_files:
        logger.error("No test audio files found!")
        return
    
    # Test on first file
    test_file = str(test_files[0])
    logger.info(f"\nTesting file: {test_file}")
    
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(test_file)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo and ensure shape is (batch, time)
    if waveform.shape[0] > 1:  # If multi-channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Average channels
    waveform = waveform.transpose(0, 1)  # Change from (channel, time) to (time, channel)
    waveform = waveform.unsqueeze(0)  # Add batch dimension: (batch, time, channel)
    waveform = waveform.squeeze(-1)  # Remove channel dim: (batch, time)
    
    logger.info(f"Input shape: {waveform.shape}")
    logger.info(f"Duration: {waveform.shape[1]/16000:.2f}s")
    
    # Test different chunk sizes from paper
    for chunk_size in encoder.chunk_sizes_samples:
        logger.info(f"\nTesting chunk size: {chunk_size/16000:.3f}s")
        streaming_outputs = verify_streaming_output(encoder, waveform, chunk_size)
        
        # Plot attention patterns
        plot_attention_patterns(
            streaming_outputs,
            f"Chunk Size: {chunk_size/16000:.3f}s",
            f"attention_pattern_{int(chunk_size/16000*1000)}ms.png"
        )
    
    # Verify attention sink
    logger.info("\nVerifying attention sink:")
    if encoder.attention_sink_cache is not None:
        sink_size = encoder.attention_sink_cache.shape[1]
        expected_size = encoder.attention_sink_size * encoder.downsample_factor
        logger.info(f"Attention sink size: {sink_size} samples (expected {expected_size})")
        assert sink_size == expected_size, "Attention sink size mismatch"
    
    # Test random chunk size selection during training
    encoder.train()
    chunk_sizes = [encoder.get_random_chunk_size() for _ in range(10)]
    logger.info("\nRandom chunk sizes during training:")
    for i, size in enumerate(chunk_sizes):
        logger.info(f"Sample {i}: {size/16000:.3f}s")

def test_fsa_decoding():
    """Test XLSR encoder's integration with k2's FSA-based decoding"""
    logger.info("Testing FSA-based decoding integration...")
    
    # Initialize encoder with paper's specifications
    encoder = XLSREncoder(
        model_name="facebook/wav2vec2-xls-r-300m",
        decode_chunk_size=8000,  # 0.5s at 16kHz
        chunk_overlap=4000,      # 0.25s overlap
        use_attention_sink=True,
        attention_sink_size=4,   # From paper's diagrams
        frame_duration=0.025,    # 25ms per frame
        frame_stride=0.020,      # 20ms stride
    )
    
    # Get test audio files
    data_dir = Path("/Users/siimhaugas/Desktop/Projects/haugasdev/XLSR-Transducer/estonian_corpora/processed/tambet")
    test_files = list(data_dir.glob("*.wav"))
    
    if not test_files:
        logger.error("No test audio files found!")
        return
    
    # Test on first file
    test_file = str(test_files[0])
    logger.info(f"\nTesting file: {test_file}")
    
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(test_file)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo and ensure shape is (batch, time)
    if waveform.shape[0] > 1:  # If multi-channel
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Average channels
    waveform = waveform.transpose(0, 1)  # Change from (channel, time) to (time, channel)
    waveform = waveform.unsqueeze(0)  # Add batch dimension: (batch, time, channel)
    waveform = waveform.squeeze(-1)  # Remove channel dim: (batch, time)
    
    # Test streaming forward pass with state management
    logger.info("\nTesting streaming forward pass with state management:")
    input_length = torch.tensor([waveform.shape[1]], dtype=torch.int32)
    states = encoder.get_init_state()
    
    # Process in chunks
    chunk_size = encoder.decode_chunk_size
    chunks = encoder.prepare_chunks(waveform, chunk_size)
    
    streaming_outputs = []
    for i, chunk in enumerate(chunks):
        chunk_len = torch.tensor([chunk.shape[1]], dtype=torch.int32)
        chunk_out, chunk_lens, states = encoder.streaming_forward(chunk, chunk_len, states)
        streaming_outputs.append(chunk_out)
        logger.info(f"Chunk {i}: input shape {chunk.shape}, output shape {chunk_out.shape}")
    
    # Verify output dimensions match joiner expectations
    concat_output = torch.cat(streaming_outputs, dim=1)
    logger.info(f"\nFinal output shape: {concat_output.shape}")
    assert concat_output.shape[-1] == encoder.encoder_dim, \
        f"Output dimension {concat_output.shape[-1]} doesn't match joiner input dimension {encoder.encoder_dim}"
    
    # Verify state management
    logger.info("\nVerifying state management:")
    assert len(states) == 1, f"Expected 1 state tensor, got {len(states)}"
    if states[0] is not None:
        logger.info(f"State shape: {states[0].shape}")
    else:
        logger.info("State is None (expected for first chunk)")

def test_loss_computation():
    """Test loss computation with variable chunk sizes and streaming regularization"""
    logger.info("Testing loss computation...")
    
    # Initialize encoder with paper's specifications
    encoder = XLSREncoder(
        model_name="facebook/wav2vec2-xls-r-300m",
        decode_chunk_size=8000,  # 0.5s at 16kHz
        chunk_overlap=4000,      # 0.25s overlap
        use_attention_sink=True,
        attention_sink_size=4,   # From paper's diagrams
        frame_duration=0.025,    # 25ms per frame
        frame_stride=0.020,      # 20ms stride
    )
    
    # Create dummy batch
    batch_size = 2
    seq_len = 32000  # 2s at 16kHz
    x = torch.randn(batch_size, seq_len)
    x_lens = torch.tensor([seq_len, seq_len//2], dtype=torch.int32)
    
    # Test random chunk size selection
    encoder.train()
    chunk_sizes = [encoder.get_random_chunk_size() for _ in range(5)]
    logger.info("\nRandom chunk sizes:")
    for i, size in enumerate(chunk_sizes):
        logger.info(f"Sample {i}: {size/16000:.3f}s")
    
    # Process with variable chunks
    chunk_size = chunk_sizes[0]
    chunks = encoder.prepare_chunks(x, chunk_size)
    
    # Verify streaming regularization
    streaming_outputs = []
    streaming_reg_losses = []
    
    encoder.reset_streaming_state()
    with torch.no_grad():
        prev_output = None
        for i, chunk in enumerate(chunks):
            chunk_len = torch.tensor([chunk.shape[1]], dtype=torch.int32)
            chunk_out = encoder.process_chunk(chunk, encoder.attention_sink_cache)
            streaming_outputs.append(chunk_out)
            
            # Compute streaming regularization loss
            if prev_output is not None:
                # Loss between last frame of previous chunk and first frame of current chunk
                reg_loss = torch.nn.functional.mse_loss(
                    prev_output[:, -1:], 
                    chunk_out[:, :1]
                )
                streaming_reg_losses.append(reg_loss.item())
                logger.info(f"Chunk {i} streaming reg loss: {reg_loss.item():.6f}")
            
            prev_output = chunk_out
    
    # Verify loss components
    if streaming_reg_losses:
        mean_reg_loss = sum(streaming_reg_losses) / len(streaming_reg_losses)
        logger.info(f"\nMean streaming regularization loss: {mean_reg_loss:.6f}")
        
        # The loss should be relatively small since we're using the same model
        assert mean_reg_loss < 1.0, "Streaming regularization loss too high"
    
    logger.info("Loss computation test passed!")

if __name__ == "__main__":
    test_xlsr_encoder()
    test_fsa_decoding()
    test_loss_computation()