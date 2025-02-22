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

# Import EncoderInterface from icefall
from encoder_interface import EncoderInterface

class XLSREncoder(EncoderInterface):
    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        decode_chunk_size: int = 5120,  # 320ms at 16kHz (paper's best performing)
        chunk_overlap: int = None,  # Will be set to decode_chunk_size // 2
        use_attention_sink: bool = True,
        attention_sink_size: int = 16,  # Paper's optimal setting
        frame_duration: float = 0.025,  # 25ms per frame (from paper)
        frame_stride: float = 0.020,  # 20ms stride (from paper)
        min_chunk_size: int = 2560,  # 160ms at 16kHz (16 frames)
        max_chunk_size: int = 20480,  # 1280ms at 16kHz (128 frames)
        left_context_chunks: int = 1,  # Paper's optimal setting
    ) -> None:
        super().__init__()
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        
        # Load model with masking disabled for inference
        config = Wav2Vec2Config.from_pretrained(model_name)
        config.mask_time_prob = 0.0
        config.mask_time_length = 1
        config.mask_feature_prob = 0.0
        config.mask_feature_length = 1
        self.model = Wav2Vec2Model.from_pretrained(model_name, config=config)
        
        # The downsample factor is 320 for wav2vec2/XLSR models
        self.downsample_factor = 320
        
        # Frame parameters (from paper)
        self.frame_duration = frame_duration
        self.frame_stride = frame_stride
        
        # Calculate frames per chunk
        self.frames_per_chunk = int(decode_chunk_size / self.downsample_factor)
        
        # Streaming parameters
        self.decode_chunk_size = decode_chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else decode_chunk_size // 2
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.left_context_chunks = left_context_chunks
        
        # Attention sink parameters (from paper)
        self.use_attention_sink = use_attention_sink
        self.attention_sink_size = attention_sink_size
        
        # Initialize streaming state
        self.reset_streaming_state()
        
        # Ensure output_dim matches joiner input
        self.output_dim = 1024  # For XLS-R 300M
        
        # Define standard chunk sizes from paper
        self.chunk_sizes = {
            "320ms": 5120,   # 16 frames
            "640ms": 10240,  # 32 frames
            "1280ms": 20480, # 64 frames
            "2560ms": 40960  # 128 frames
        }
        
        # Validate chunk size is one of paper's configurations
        assert decode_chunk_size in self.chunk_sizes.values(), \
            f"Chunk size {decode_chunk_size} not in paper's configurations: {list(self.chunk_sizes.values())}"

    def get_init_state(self, device: Optional[torch.device] = None) -> List[Optional[torch.Tensor]]:
        """Get initial states for streaming inference"""
        if device is None:
            device = next(self.parameters()).device
        return [None]  # Initial state is None since we'll build it from first chunk

    def reset_streaming_state(self):
        """Reset all streaming state variables"""
        self.cached_features = None
        self.cached_len = 0
        self.current_chunk_size = self.decode_chunk_size
        self.last_chunk_latency = 0
        self.streaming_state = None
        self.attention_sink_cache = None
        self.context_cache = None
        self.last_chunk_output = None
        self.left_context_buffer = []  # Store left context chunks

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
        """Apply smooth transition between chunks using sinusoidal interpolation"""
        if previous_output is None or self.frames_per_chunk <= 0:
            return current_output
            
        # Get transition regions
        prev_trans = previous_output[:, -self.frames_per_chunk:]
        curr_trans = current_output[:, :self.frames_per_chunk]
        
        # Create transition weights using sin for smoother ramp-up
        weights = torch.sin(torch.linspace(0.0, float(torch.pi/2), steps=self.frames_per_chunk, device=current_output.device))
        weights = weights.view(1, -1, 1)  # Shape for broadcasting
        
        # Interpolate
        transition = weights * curr_trans + (1 - weights) * prev_trans
        
        # Replace transition region in current output
        current_output = current_output.clone()
        current_output[:, :self.frames_per_chunk] = transition
        
        return current_output

    def prepare_attention_sink(self, chunk: torch.Tensor, sink_cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare attention sink following paper's approach.
        Args:
            chunk: Input tensor (batch, time)
            sink_cache: Optional cached sink from previous chunk
        Returns:
            (chunk_with_sink, new_sink_cache)
        """
        if not self.use_attention_sink:
            return chunk, None
            
        # Calculate sink size in samples
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
            
        return chunk_with_sink, new_sink_cache

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
            
        # Take up to left_context_chunks previous chunks
        context = self.left_context_buffer[-self.left_context_chunks:]
        context_tensor = torch.cat(context, dim=1)
        
        return torch.cat([context_tensor, chunk], dim=1)

    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
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
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        
        # Clamp values silently since inputs are already normalized
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Add left context if available
        x = self.prepare_left_context(x)
        
        # Add attention sink
        x, new_sink_cache = self.prepare_attention_sink(x, self.attention_sink_cache)
        
        # Process through XLSR model
        outputs = self.model(
            x,
            attention_mask=None,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False
        )[0]
        
        # Update left context buffer
        self.left_context_buffer.append(x.clone())
        if len(self.left_context_buffer) > self.left_context_chunks:
            self.left_context_buffer.pop(0)
        
        # Calculate output lengths
        output_lengths = ((x_lens.float() / self.downsample_factor).floor()).to(torch.int64)
        output_lengths = torch.maximum(output_lengths, torch.ones_like(output_lengths))
        
        # Update states for next chunk
        next_states = [x, new_sink_cache]
        
        return outputs, output_lengths, next_states

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Non-streaming forward pass"""
        # Ensure input is float and in correct shape
        x = x.float()
        
        if x.ndim == 3:  # (batch, time, channel)
            x = x.squeeze(-1)
        elif x.ndim == 1:  # (time,)
            x = x.unsqueeze(0)  # Add batch dimension
        
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        
        # Clamp values silently since inputs are already normalized
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Forward pass through model
        outputs = self.model(
            x,
            attention_mask=None,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False
        )[0]
        
        # Calculate output lengths
        output_lengths = ((x_lens.float() / self.downsample_factor).floor()).to(torch.int64)
        output_lengths = torch.maximum(output_lengths, torch.ones_like(output_lengths))
        
        return outputs, output_lengths

    def prepare_chunks(self, x: torch.Tensor, chunk_size: int) -> List[torch.Tensor]:
        """
        Prepare input into chunks for streaming processing.
        Args:
            x: Input tensor (batch, time) or (batch, time, 1)
            chunk_size: Size of each chunk in samples
        Returns:
            List of chunks, each of shape (batch, chunk_size)
        """
        # Ensure input is 2D
        if x.ndim == 3:
            x = x.squeeze(-1)
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        
        # Calculate overlap size
        overlap = self.chunk_overlap
        
        # Initialize chunks list
        chunks = []
        
        # Process in chunks with overlap
        current = 0
        while current < x.size(1):
            end = min(current + chunk_size, x.size(1))
            chunk = x[:, current:end]
            
            # Add padding if needed for last chunk
            if chunk.size(1) < chunk_size:
                pad_size = chunk_size - chunk.size(1)
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
            
            chunks.append(chunk)
            
            # Move to next chunk considering overlap
            if end == x.size(1):  # Last chunk
                break
            current = end - overlap
        
        return chunks 