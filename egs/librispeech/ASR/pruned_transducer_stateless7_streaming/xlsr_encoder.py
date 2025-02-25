#!/usr/bin/env python3

import torch
import torch.nn as nn
import time
import logging
import random
import numpy as np
from typing import Optional, List, Tuple
from icefall.utils import make_pad_mask
import math

class EncoderInterface(nn.Module):
    """Interface for encoders used in transducer models"""
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 0  # Must be set by implementing class
        
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, time) or (batch, time, 1)
            x_lens: Length of each sequence in the batch
        Returns:
            (output, output_lens)
            output: (batch, time', output_dim)
            output_lens: (batch,)
        """
        raise NotImplementedError

    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[List[List[Optional[torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[Optional[torch.Tensor]]]]:
        """
        Streaming forward pass with proper context and attention sink handling
        Args:
            x: Input tensor (batch, time) or (batch, time, 1)
            x_lens: Length of each sequence in batch
            states: Optional cached states from previous chunk
            
        Returns:
            (encoder_out, encoder_out_lens, next_states)
        """
        # Ensure input is float and in correct shape
        x = x.float()
        if x.ndim == 3:
            x = x.squeeze(-1)
        
        # Get batch size
        batch_size = x.size(0)
        device = x.device
        
        # Calculate expected output lengths
        feature_downsampling = self.downsample_factor  # 320 for XLS-R
        expected_frames = ((x_lens.float() / feature_downsampling).ceil()).to(torch.int64)
        expected_frames = torch.maximum(expected_frames, torch.ones_like(expected_frames))
        
        # Process each batch item separately to handle different states
        all_outputs = []
        new_states = []
        
        for b in range(batch_size):
            # Get single sequence
            single_x = x[b:b+1]
            single_len = x_lens[b]
            
            # Get states for this sequence
            if states is not None and b < len(states) and states[b] is not None:
                # Unpack states
                past_hidden_states = states[b][0]  # Previous hidden states for left context
                attention_sink_cache = states[b][1]  # Attention sink frames
            else:
                past_hidden_states = None
                attention_sink_cache = None
            
            # Create attention mask for current chunk
            attention_mask = torch.ones_like(single_x, dtype=torch.long)
            if single_len < single_x.size(1):
                attention_mask[0, single_len:] = 0
            
            # Forward through XLSR model
            outputs = self.model(
                single_x,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
            
            # Save current output states
            current_frames = min(hidden_states.size(1), self.frames_per_chunk)
            
            # Prepare for next step
            # 1. Left context: Last frames from current chunk
            new_left_context = hidden_states[:, -self.chunk_overlap//self.downsample_factor:]
            
            # 2. Attention sink: First few frames if enabled
            if self.use_attention_sink:
                new_attention_sink = hidden_states[:, :self.attention_sink_size]
            else:
                new_attention_sink = None
            
            # Save states for this batch item
            new_states.append([new_left_context, new_attention_sink])
            
            # CRITICAL FIX: When processing chunks, we need to properly handle overlap regions
            # to avoid frame duplication. The paper carefully removes overlapping regions.
            
            # If we have previous states with left context frames, need to handle overlaps
            if states is not None and b < len(states) and states[b] is not None and states[b][0] is not None:
                # Adjust the attention mask to account for overlap with previous chunk
                overlap_frames = self.chunk_overlap // self.downsample_factor
                
                # Only use the non-overlapping portion of the output
                # This is crucial to prevent output frame duplication
                if hidden_states.size(1) > overlap_frames:
                    hidden_states = hidden_states[:, overlap_frames:]
            
            # Add to outputs
            all_outputs.append(hidden_states)
        
        # Combine all outputs - pad to max length
        max_len = max(output.size(1) for output in all_outputs)
        padded_outputs = []
        
        for output in all_outputs:
            if output.size(1) < max_len:
                padding = torch.zeros(
                    1, max_len - output.size(1), output.size(2),
                    device=device, dtype=output.dtype
                )
                padded_outputs.append(torch.cat([output, padding], dim=1))
            else:
                padded_outputs.append(output)
        
        encoder_out = torch.cat(padded_outputs, dim=0)
        encoder_out_lens = torch.tensor([min(out.size(1), max_len) for out in all_outputs], device=device)
        
        # After processing all batch items, before returning:
        # Verify frame counts match expected lengths
        expected_total = sum(expected_frames.tolist())
        actual_total = sum(encoder_out_lens.tolist())

        if actual_total > expected_total * 1.1:  # >10% more frames than expected
            self.logger.warning(
                f"Output frame count mismatch: expected ~{expected_total}, got {actual_total}. "
                f"This may cause transcript duplication. Check overlap handling."
            )
        
        return encoder_out, encoder_out_lens, new_states

    def get_init_state(self, device: Optional[torch.device] = None) -> List[Optional[torch.Tensor]]:
        """Get initial states for streaming inference"""
        if device is None:
            device = next(self.parameters()).device
        return [None]  # Initial state is None since we'll build it from first chunk

    def reset_streaming_state(self):
        """Reset all streaming state variables to ensure clean inference"""
        self.cached_features = None
        self.cached_len = 0
        self.current_chunk_size = self.decode_chunk_size
        self.last_chunk_latency = 0
        self.streaming_state = None
        self.attention_sink_cache = None
        self.context_cache = None
        self.last_chunk_output = None
        self.left_context_buffer = []
        self.past_context = None
        self.is_streaming = False
        
        # Also clear model's cache if supported
        if hasattr(self.model, "clear_cache"):
            self.model.clear_cache()

    def prepare_chunk_with_context(self, chunk: torch.Tensor, left_context: torch.Tensor = None, right_context: torch.Tensor = None) -> torch.Tensor:
        """Prepare chunk with left and right context for better boundary handling"""
        context_size = self.frames_per_chunk
        
        # Add left context if available
        if left_context is not None:
            chunk = torch.cat([left_context[:, -context_size:], chunk], dim=1)
        else:
            # Pad with zeros if no left context
            chunk = torch.nn.functional.pad(chunk, (context_size, 0))
            
        # Add right context if available
        if right_context is not None:
            chunk = torch.cat([chunk, right_context[:, :context_size]], dim=1)
        else:
            # Pad with zeros if no right context
            chunk = torch.nn.functional.pad(chunk, (0, context_size))
            
        return chunk

    def smooth_transition(self, current_output: torch.Tensor, previous_output: torch.Tensor = None) -> torch.Tensor:
        """Apply smooth transition between chunks using sinusoidal interpolation
        
        Args:
            current_output: Current chunk output
            previous_output: Previous chunk output
            
        Returns:
            Smoothed output tensor
        """
        if previous_output is None:
            return current_output
            
        # Calculate actual overlap size based on available frames
        overlap_size = min(
            previous_output.size(1),  # Previous chunk size
            current_output.size(1),   # Current chunk size
            self.frames_per_chunk // 4  # Use 25% of frame size for transition
        )
        
        if overlap_size <= 0:
            return current_output
            
        # Get transition regions
        prev_trans = previous_output[:, -overlap_size:]
        curr_trans = current_output[:, :overlap_size]
        
        # Create transition weights using sin for smoother ramp-up
        weights = torch.sin(torch.linspace(0.0, float(torch.pi/2), steps=overlap_size, device=current_output.device))
        weights = weights.view(1, -1, 1)  # Shape for broadcasting
        
        # Interpolate
        transition = weights * curr_trans + (1 - weights) * prev_trans
        
        # Replace transition region in current output
        current_output = current_output.clone()
        current_output[:, :overlap_size] = transition
        
        return current_output

    def prepare_attention_sink(self, chunk: torch.Tensor, sink_cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare attention sink following paper's approach.
        Args:
            chunk: Input tensor (batch, time)
            sink_cache: Optional cached sink from previous chunk
        Returns:
            (chunk_with_sink, new_sink_cache, attention_mask)
        """
        if not self.use_attention_sink:
            return chunk, None, None
            
        # Calculate sink size in samples (16 frames as per paper)
        sink_size = self.attention_sink_size * self.downsample_factor
        
        if sink_cache is None:
            # For first chunk, use beginning of chunk as sink
            new_sink_cache = chunk[:, :sink_size]
            chunk_with_sink = chunk
        else:
            # Concatenate sink with current chunk
            chunk_with_sink = torch.cat([sink_cache, chunk], dim=1)
            # Update sink cache with end of current chunk
            new_sink_cache = chunk[:, -sink_size:]
            
        # Add masking for attention constraints
        seq_len = chunk_with_sink.size(1)
        mask = torch.zeros(seq_len, device=chunk.device)
        
        # Allow attention to sink and current chunk only
        mask[:sink_size] = 1  # Attention sink
        mask[sink_size:sink_size+self.decode_chunk_size] = 1  # Current chunk
        
        return chunk_with_sink, new_sink_cache, mask

    def prepare_left_context(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Prepare left context following paper's approach.
        Args:
            chunk: Current input chunk
        Returns:
            Chunk with left context prepended
        """
        if not self.left_context_chunks or not self.left_context_buffer:
            return chunk
            
        # Take up to left_context_chunks previous chunks (1 as per paper)
        context = self.left_context_buffer[-self.left_context_chunks:]
        context_tensor = torch.cat(context, dim=1)
        
        return torch.cat([context_tensor, chunk], dim=1)

    def process_streaming_chunks(self, model, feature, chunk_size, attention_sink_size, left_context_chunks):
        """Process input in streaming chunks following paper's approach
        
        Args:
            model: The XLSR model
            feature: Input feature tensor (batch, time)
            chunk_size: Size of chunks in samples
            attention_sink_size: Size of attention sink in frames
            left_context_chunks: Number of left context chunks to use
            
        Returns:
            Processed encoder output
        """
        batch_size = feature.size(0)
        device = feature.device
        
        # Create storage for outputs
        all_encoder_outputs = []
        
        # Process each sequence separately
        for b in range(batch_size):
            seq = feature[b:b+1]  # Keep batch dimension
            seq_len = seq.size(1)
            
            # Process in chunks
            pos = 0
            encoder_chunks = []
            
            # Initialize states for streaming
            states = None
            
            while pos < seq_len:
                # Get current chunk with proper bounds checking
                end_pos = min(pos + chunk_size, seq_len)
                chunk = seq[:, pos:end_pos]
                
                # Process chunk with streaming forward
                # Note how we're initializing states as None for first chunk, then using returned states
                encoder_out, _, states = self.streaming_forward(
                    chunk, 
                    torch.tensor([chunk.size(1)], device=device),
                    states=[states] if states is not None else None
                )
                
                # Add to output chunks
                encoder_chunks.append(encoder_out)
                
                # Move position forward (with overlap as in paper)
                pos = end_pos - (chunk_size // 2)
                pos = max(pos, end_pos - chunk_size // 2)  # Ensure we make forward progress
            
            # Concatenate chunks along time dimension
            if encoder_chunks:
                # Simple concatenation for now - could use better chunk merging in the future
                sequence_output = torch.cat(encoder_chunks, dim=1)
                all_encoder_outputs.append(sequence_output)
            else:
                # Fallback if no chunks were processed (shouldn't happen)
                logging.warning(f"No chunks processed for batch item {b}")
                dummy_output = torch.zeros((1, 1, self.output_dim), device=device)
                all_encoder_outputs.append(dummy_output)
        
        # Combine all batch outputs - pad to max length
        max_len = max(out.size(1) for out in all_encoder_outputs)
        padded_outputs = []
        
        for output in all_encoder_outputs:
            if output.size(1) < max_len:
                padding = torch.zeros(
                    1, max_len - output.size(1), output.size(2),
                    device=device, dtype=output.dtype
                )
                padded_outputs.append(torch.cat([output, padding], dim=1))
            else:
                padded_outputs.append(output)
        
        # Return combined output
        return torch.cat(padded_outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        is_pre_training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Non-streaming forward pass with option for pre-training mode
        
        Args:
            x: Input tensor (batch, time) or (batch, time, 1)
            x_lens: Length of each sequence in batch
            is_pre_training: Whether we're in pre-training mode (no streaming/chunks)
            
        Returns:
            (encoder_out, encoder_out_lens)
        """
        # Ensure input is float and in correct shape
        x = x.float()
        if x.ndim == 3:
            x = x.squeeze(-1)
        
        batch_size = x.size(0)
        
        # Create attention mask for padding
        attention_mask = torch.ones_like(x, dtype=torch.long)
        for i in range(batch_size):
            if i < x_lens.size(0) and x_lens[i] < x.size(1):
                attention_mask[i, x_lens[i]:] = 0
        
        # Calculate output lengths right away to ensure they're always defined
        encoder_out_lens = ((x_lens.float() / self.downsample_factor).floor()).to(torch.int64)
        encoder_out_lens = torch.maximum(encoder_out_lens, torch.ones_like(encoder_out_lens))
        
        if is_pre_training:
            # During pre-training, use full attention without chunking
            outputs = self.model(
                x,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state
        else:
            # Use chunked attention with masks for streaming simulation during training
            # This is crucial for streaming performance
            
            # Apply chunked attention mask
            streaming_mask = self.create_streaming_mask(
                x, attention_mask, self.decode_chunk_size, self.left_context_chunks
            )
            
            # Forward with streaming mask
            outputs = self.model(
                x,
                attention_mask=streaming_mask,
                output_hidden_states=False,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state
        
        return hidden_states, encoder_out_lens

    def set_chunk_size(self, chunk_size: int):
        """Set the chunk size for streaming inference
        
        Args:
            chunk_size: Size of chunk in samples
        """
        if chunk_size not in self.chunk_sizes.values():
            self.logger.warning(
                f"Chunk size {chunk_size} not in paper's configurations: "
                f"{list(self.chunk_sizes.values())}. Using anyway, but consider changing."
            )
        
        self.decode_chunk_size = chunk_size
        self.chunk_overlap = int(chunk_size * 0.4)  # 40% overlap as per paper
        
        # Update frames per chunk based on new chunk size
        self.frames_per_chunk = chunk_size // self.samples_per_frame
        
        self.logger.info(f"Set chunk size to {chunk_size} samples ({chunk_size/16000*1000:.1f} ms), "
                         f"{self.frames_per_chunk} frames per chunk, "
                         f"overlap: {self.chunk_overlap} samples ({self.chunk_overlap/16000*1000:.1f} ms)")

    def apply_attention_sink_mask(self, x, attention_mask=None):
        """Apply attention sink masking as described in the paper
        
        Args:
            x: Input features (batch, time, dim)
            attention_mask: Optional existing mask
            
        Returns:
            Modified attention mask allowing attention to sink frames
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), 
                                        device=x.device, 
                                        dtype=torch.long)
        
        if not self.use_attention_sink:
            return attention_mask
        
        # Create custom attention mask for streaming with attention sink
        # This is the key implementation from the paper
        extended_mask = torch.zeros((batch_size, seq_len, seq_len), 
                                   device=x.device, 
                                   dtype=torch.bool)
        
        for b in range(batch_size):
            # Allow each position to attend to itself and previous positions in chunk
            for i in range(seq_len):
                # Find start of current chunk
                chunk_size = self.frames_per_chunk
                chunk_start = max(0, i - i % chunk_size)
                
                # Allow attention to current chunk
                extended_mask[b, i, chunk_start:i+1] = True
                
                # Allow attention to sink frames (first N frames)
                if self.attention_sink_size > 0:
                    sink_end = min(self.attention_sink_size, seq_len)
                    extended_mask[b, i, :sink_end] = True
        
        # Combine with original mask to respect padding
        final_mask = extended_mask & attention_mask.unsqueeze(-1)
        
        return final_mask

    def create_streaming_mask(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor, 
        chunk_size: int,
        left_context_chunks: int = 1
    ) -> torch.Tensor:
        """Create streaming attention mask with optional attention sink
        
        Args:
            x: Input tensor (batch, time)
            attention_mask: Base mask for padding
            chunk_size: Size of chunk in samples
            left_context_chunks: Number of chunks to use as left context
            
        Returns:
            Extended attention mask for streaming
        """
        batch_size, seq_len = x.shape
        frame_chunk_size = chunk_size // self.samples_per_frame
        
        # Create mask that limits attention to current and previous chunks
        streaming_mask = torch.zeros(
            (batch_size, seq_len, seq_len), 
            dtype=torch.bool, 
            device=x.device
        )
        
        for b in range(batch_size):
            for i in range(seq_len):
                current_chunk_idx = i // frame_chunk_size
                
                # Start of visible context
                context_start = max(0, (current_chunk_idx - left_context_chunks) * frame_chunk_size)
                
                # Allow attention to current position and all positions back to context start
                streaming_mask[b, i, context_start:i+1] = True
                
                # Add attention sink if enabled
                if self.use_attention_sink and self.attention_sink_size > 0:
                    sink_end = min(self.attention_sink_size, seq_len)
                    streaming_mask[b, i, :sink_end] = True
        
        # Apply original attention mask for padding
        extended_mask = streaming_mask & attention_mask.unsqueeze(-1).bool()
        
        # Convert to transformers attention mask format (1 = attend, 0 = mask)
        transformers_mask = extended_mask.long()
        
        return transformers_mask

class XLSREncoder(EncoderInterface):
    def __init__(
        self, 
        model_name: str = "TalTechNLP/xls-r-300m-et",
        decode_chunk_size: int = 5120,  # 320ms at 16kHz
        chunk_overlap: int = None,
        use_attention_sink: bool = True,
        attention_sink_size: int = 16,  # Paper's optimal setting
        frame_duration: float = 0.025,  # 25ms per frame
        frame_stride: float = 0.020,  # 20ms stride
        min_chunk_size: int = 2560,  # 160ms at 16kHz
        max_chunk_size: int = 20480,  # 1280ms at 16kHz
        left_context_chunks: int = 1,  # Paper's optimal setting
    ) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load model with masking disabled for inference
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        
        config = Wav2Vec2Config.from_pretrained(model_name)
        config.mask_time_prob = 0.0
        config.mask_time_length = 1
        config.mask_feature_prob = 0.0
        config.mask_feature_length = 1
        self.model = Wav2Vec2Model.from_pretrained(model_name, config=config)
        
        # The downsample factor for wav2vec2/XLSR models
        self.downsample_factor = 320
        
        # Conversion factors between samples and frames
        self.sample_rate = 16000
        self.frame_duration = frame_duration  # 25ms
        self.frame_stride = frame_stride      # 20ms
        
        # Samples per frame (320 samples at 16kHz with 20ms stride)
        self.samples_per_frame = int(self.sample_rate * self.frame_stride)
        
        # Frames per chunk calculation
        self.frames_per_chunk = decode_chunk_size // self.samples_per_frame
        self.logger.info(f"Chunk size: {decode_chunk_size} samples = {decode_chunk_size/16000*1000:.1f}ms = {self.frames_per_chunk} frames")
        
        # Streaming parameters
        self.decode_chunk_size = decode_chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else int(decode_chunk_size * 0.4)
        self.logger.info(f"Using chunk overlap of {self.chunk_overlap} samples ({self.chunk_overlap/16000*1000:.1f}ms)")
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.left_context_chunks = left_context_chunks
        
        # Attention sink parameters
        self.use_attention_sink = use_attention_sink
        self.attention_sink_size = attention_sink_size
        
        # Initialize streaming state
        self.reset_streaming_state()
        
        # Output dimension
        self.output_dim = 1024  # Fixed for XLS-R 300M
        
        # Standard chunk sizes from paper (in samples)
        self.chunk_sizes = {
            "320ms": 5120,   # 16 frames
            "640ms": 10240,  # 32 frames
            "1280ms": 20480, # 64 frames
            "2560ms": 40960  # 128 frames
        }

        # Validate chunk size is one of paper's configurations
        if decode_chunk_size not in self.chunk_sizes.values():
            self.logger.warning(
                f"Chunk size {decode_chunk_size} not in paper's configurations: "
                f"{list(self.chunk_sizes.values())}. Using anyway, but consider changing."
            ) 