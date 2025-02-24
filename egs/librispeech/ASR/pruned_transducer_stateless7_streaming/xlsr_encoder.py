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
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Streaming forward pass
        Args:
            x: Input tensor (batch, time) or (batch, time, 1)
            x_lens: Length of each sequence in batch
            states: Optional cached states from previous chunk
        Returns:
            (encoder_out, encoder_out_lens, next_states)
        """
        raise NotImplementedError

    def get_init_state(self, device: Optional[torch.device] = None) -> List[Optional[torch.Tensor]]:
        """Get initial states for streaming inference"""
        raise NotImplementedError

class XLSREncoder(EncoderInterface):
    """XLSR encoder with streaming support and attention sink."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model: Optional[nn.Module] = None,
        decode_chunk_size: int = 5120,  # 320ms at 16kHz
        chunk_overlap: int = 2560,      # Half of chunk size
        use_attention_sink: bool = True,
        attention_sink_size: int = 16,   # Paper's optimal setting
        frame_duration: float = 0.025,   # 25ms per frame
        frame_stride: float = 0.020,     # 20ms stride
        min_chunk_size: int = 2560,      # 160ms at 16kHz
        max_chunk_size: int = 20480,     # 1280ms at 16kHz
        left_context_chunks: int = 1     # Paper's optimal setting
    ):
        super().__init__()
        
        if model is not None:
            self.model = model
        elif model_name is not None:
            from transformers import Wav2Vec2Model
            try:
                self.model = Wav2Vec2Model.from_pretrained(model_name)
                logging.info(f"Successfully loaded XLSR model from {model_name}")
            except Exception as e:
                raise ValueError(f"Failed to load XLSR model from {model_name}: {str(e)}")
        else:
            raise ValueError("Either model_name or model must be provided")
        
        # Store configuration
        self.decode_chunk_size = decode_chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_attention_sink = use_attention_sink
        self.attention_sink_size = attention_sink_size
        self.frame_duration = frame_duration
        self.frame_stride = frame_stride
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.left_context_chunks = left_context_chunks
        
        # Initialize streaming state
        self.reset_streaming_state()
        
        # Freeze feature encoder
        if hasattr(self.model, "feature_extractor"):
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
        
        # Add attention sink if enabled
        if use_attention_sink:
            self.attention_sink = nn.Parameter(
                torch.randn(attention_sink_size, self.model.config.hidden_size)
            )
        else:
            self.attention_sink = None
        
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
        """Reset streaming state between utterances."""
        self.streaming_state = None
        self.past_context = None
        self.cached_features = None
        self.cached_len = 0
        self.current_chunk_size = self.decode_chunk_size
        self.last_chunk_latency = 0
        self.attention_sink_cache = None
        self.context_cache = None
        self.last_chunk_output = None
        self.left_context_buffer = []  # Store left context chunks

    def prepare_chunk_with_context(self, chunk: torch.Tensor, left_context: torch.Tensor = None, right_context: torch.Tensor = None) -> torch.Tensor:
        """Prepare chunk with left and right context for better boundary handling"""
        context_size = self.decode_chunk_size // 2
        
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
        if previous_output is None:
            return current_output
            
        # Calculate actual overlap size based on available frames
        overlap_size = min(
            previous_output.size(1),  # Previous chunk size
            current_output.size(1),   # Current chunk size
            self.decode_chunk_size     # Maximum overlap
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
            
        # Calculate sink size in samples (16 frames as per paper)
        sink_size = self.attention_sink_size * self.decode_chunk_size // 320
        
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
            
        # Take up to left_context_chunks previous chunks (1 as per paper)
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
        batch_size = x.size(0)
        
        if states is not None:
            # For subsequent chunks, adjust for overlap
            effective_input_len = x_lens - self.chunk_overlap
            # Add small correction factor for chunk boundary alignment
            boundary_correction = self.decode_chunk_size // 320  # 20 samples at 16kHz
            effective_input_len = effective_input_len + boundary_correction
        else:
            # For first chunk, use full length
            effective_input_len = x_lens
            
        # Calculate expected frames precisely using ceil instead of floor
        expected_frames = ((effective_input_len.float() / self.decode_chunk_size).ceil()).to(torch.int64)
        expected_frames = torch.maximum(expected_frames, torch.ones_like(expected_frames))
        
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
        
        # Remove context frames from output
        if self.use_attention_sink:
            outputs = outputs[:, self.attention_sink_size:]
        
        # Ensure outputs match expected length precisely
        max_len = expected_frames.max().item()
        if outputs.size(1) > max_len:
            # Calculate optimal start position to minimize edge effects
            extra = outputs.size(1) - max_len
            # Bias towards keeping more recent frames for streaming
            start = (extra * 2) // 3  # Take more frames from the end
            outputs = outputs[:, start:start + max_len]
        elif outputs.size(1) < max_len and outputs.size(1) > 0:  # Only pad if we have some frames
            # Calculate required padding
            pad_len = max_len - outputs.size(1)
            # Add padding at the end to maintain temporal order
            outputs = torch.nn.functional.pad(
                outputs,
                (0, 0, 0, pad_len),
                mode='replicate'  # Replicate last frame instead of zeros
            )
        elif outputs.size(1) == 0:  # Handle empty tensor case
            # Create a tensor of the expected size filled with zeros
            outputs = torch.zeros(
                (outputs.size(0), max_len, outputs.size(2)),
                device=outputs.device,
                dtype=outputs.dtype
            )
        
        # Apply smooth transition if we have previous output
        if self.last_chunk_output is not None:
            outputs = self.smooth_transition(outputs, self.last_chunk_output)
        
        # Cache current output for next chunk
        self.last_chunk_output = outputs.clone()
        
        # Update states for next chunk
        next_states = [x, new_sink_cache]
        
        return outputs, expected_frames, next_states

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        simulate_streaming: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (B, T)
            x_lens: Length of each sequence (B,)
            simulate_streaming: Whether to simulate streaming inference
            
        Returns:
            Tuple of (encoder_out, encoder_out_lens)
        """
        batch_size = x.size(0)
        
        if simulate_streaming:
            # Simulate streaming inference with chunks
            outputs = []
            current_lengths = []
            
            for i in range(batch_size):
                # Process each sequence in the batch
                seq_len = x_lens[i]
                sequence = x[i, :seq_len]
                
                # Process in chunks
                chunk_outputs = []
                pos = 0
                while pos < seq_len:
                    # Get current chunk
                    chunk_size = min(self.decode_chunk_size, seq_len - pos)
                    chunk = sequence[pos:pos + chunk_size]
                    
                    # Add left context if available
                    if self.past_context is not None:
                        chunk = torch.cat([self.past_context, chunk], dim=0)
                    
                    # Process chunk
                    if isinstance(self.model, Wav2Vec2ForCTC):
                        chunk_output = self.model(
                            chunk.unsqueeze(0),
                            attention_mask=torch.ones_like(chunk).unsqueeze(0),
                            output_hidden_states=True
                        ).hidden_states[-1]
                    else:
                        chunk_output = self.model(
                            chunk.unsqueeze(0),
                            attention_mask=torch.ones_like(chunk).unsqueeze(0)
                        ).last_hidden_state
                    
                    # Add attention sink if enabled
                    if self.use_attention_sink:
                        chunk_output = torch.cat([
                            self.attention_sink.unsqueeze(0).expand(1, -1, -1),
                            chunk_output
                        ], dim=1)
                    
                    chunk_outputs.append(chunk_output)
                    
                    # Update position and save context
                    pos += chunk_size - self.chunk_overlap
                    if pos < seq_len:
                        self.past_context = chunk[-self.chunk_overlap:]
                
                # Concatenate chunks
                sequence_output = torch.cat(chunk_outputs, dim=1)
                outputs.append(sequence_output)
                current_lengths.append(sequence_output.size(1))
            
            # Pad to max length
            max_len = max(current_lengths)
            padded_outputs = []
            for output, length in zip(outputs, current_lengths):
                if length < max_len:
                    padding = torch.zeros(
                        1,
                        max_len - length,
                        output.size(-1),
                        device=output.device,
                        dtype=output.dtype
                    )
                    padded_outputs.append(torch.cat([output, padding], dim=1))
                else:
                    padded_outputs.append(output)
            
            encoder_out = torch.cat(padded_outputs, dim=0)
            encoder_out_lens = torch.tensor(current_lengths, device=x.device)
            
        else:
            # Non-streaming forward pass
            if isinstance(self.model, Wav2Vec2ForCTC):
                encoder_out = self.model(
                    x,
                    attention_mask=torch.ones_like(x),
                    output_hidden_states=True
                ).hidden_states[-1]
            else:
                encoder_out = self.model(
                    x,
                    attention_mask=torch.ones_like(x)
                ).last_hidden_state
            
            # Add attention sink if enabled
            if self.use_attention_sink:
                encoder_out = torch.cat([
                    self.attention_sink.unsqueeze(0).expand(batch_size, -1, -1),
                    encoder_out
                ], dim=1)
            
            # Update lengths to account for attention sink
            encoder_out_lens = x_lens
            if self.use_attention_sink:
                encoder_out_lens = encoder_out_lens + self.attention_sink_size
        
        return encoder_out, encoder_out_lens 