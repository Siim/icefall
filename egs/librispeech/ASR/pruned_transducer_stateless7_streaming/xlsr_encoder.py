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

class XLSREncoder(EncoderInterface):
    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        decode_chunk_size: int = 20480,  # Default to 1280ms at 16kHz (paper's recommended size)
        chunk_overlap: int = None,  # Will be set to decode_chunk_size // 2
        use_attention_sink: bool = True,
        attention_sink_size: int = 4,  # Number of attention sink frames
        frame_duration: float = 0.025,  # 25ms per frame
        frame_stride: float = 0.020,  # 20ms stride
        min_chunk_size: int = 5120,  # 320ms at 16kHz (paper's min size)
        max_chunk_size: int = 40960,  # 2560ms at 16kHz (paper's max size)
        context_frames: int = None,  # Will be set based on chunk size
        transition_frames: int = None,  # Will be set based on chunk size
        sink_warmup_frames: int = None,  # Will be set based on attention sink size
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
        
        # Streaming parameters
        self.decode_chunk_size = decode_chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else decode_chunk_size // 2
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Context frames should be one full chunk (paper's approach)
        self.context_frames = context_frames if context_frames is not None else self.decode_chunk_size // self.downsample_factor
        
        # Transition frames for smooth chunk boundaries (paper suggests gradual transitions)
        self.transition_frames = transition_frames if transition_frames is not None else self.decode_chunk_size // (self.downsample_factor * 8)
        
        # Attention sink parameters (from paper)
        self.use_attention_sink = use_attention_sink
        self.attention_sink_size = attention_sink_size
        self.sink_warmup_frames = sink_warmup_frames if sink_warmup_frames is not None else self.attention_sink_size // 2
        
        # Initialize streaming state
        self.reset_streaming_state()
        
        # Ensure output_dim matches joiner input
        self.output_dim = 1024  # For XLS-R 300M

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

    def prepare_chunk_with_context(self, chunk: torch.Tensor, left_context: torch.Tensor = None, right_context: torch.Tensor = None) -> torch.Tensor:
        """Prepare chunk with left and right context for better boundary handling"""
        context_size = self.context_frames * self.downsample_factor
        
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
        if previous_output is None or self.transition_frames <= 0:
            return current_output
            
        # Get transition regions
        prev_trans = previous_output[:, -self.transition_frames:]
        curr_trans = current_output[:, :self.transition_frames]
        
        # Create transition weights using sin for smoother ramp-up
        weights = torch.sin(torch.linspace(0.0, float(torch.pi/2), steps=self.transition_frames, device=current_output.device))
        weights = weights.view(1, -1, 1)  # Shape for broadcasting
        
        # Interpolate
        transition = weights * curr_trans + (1 - weights) * prev_trans
        
        # Replace transition region in current output
        current_output = current_output.clone()
        current_output[:, :self.transition_frames] = transition
        
        return current_output

    def prepare_attention_sink(self, chunk: torch.Tensor, sink_cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare attention sink for streaming inference following paper's approach.
        Args:
            chunk: Input tensor (batch, time)
            sink_cache: Optional cached sink from previous chunk
        Returns:
            (chunk_with_sink, new_sink_cache)
        """
        if not self.use_attention_sink or sink_cache is None:
            return chunk, chunk[:, -self.attention_sink_size * self.downsample_factor:]
            
        # Calculate sink size in samples
        sink_size = self.attention_sink_size * self.downsample_factor
        
        # Warm up sink with more context if available
        warmup_size = self.sink_warmup_frames * self.downsample_factor
        if sink_cache.size(1) >= sink_size + warmup_size:
            sink = sink_cache[:, -(sink_size + warmup_size):]
        else:
            sink = sink_cache[:, -sink_size:]
            
        # Concatenate sink with current chunk
        chunk_with_sink = torch.cat([sink, chunk], dim=1)
        
        # Cache new sink for next chunk
        new_sink_cache = chunk[:, -sink_size:]
        
        return chunk_with_sink, new_sink_cache

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
        
        # Get context and sink from states
        left_context = states[0] if states is not None and states[0] is not None else None
        sink_cache = states[1] if states is not None and len(states) > 1 else None
        
        # Calculate effective context size
        context_size = self.context_frames * self.downsample_factor
        
        # Calculate expected output length for full sequence
        # Match paper's frame calculation without -1
        expected_frames = ((x_lens.float() / self.downsample_factor).floor()).to(torch.int64)
        expected_frames = torch.maximum(expected_frames, torch.ones_like(expected_frames))
        
        # Calculate chunk-specific output length to match expected frames
        chunk_samples = x.size(1)
        chunk_frames = ((torch.tensor([chunk_samples], dtype=torch.float32, device=x.device) / 
                      self.downsample_factor).floor()).to(torch.int64)
        
        # Prepare chunk with context
        if left_context is not None:
            # Use context from previous chunk
            chunk_with_context = torch.cat([left_context[:, -context_size:], x], dim=1)
        else:
            # Pad with zeros if no left context
            chunk_with_context = torch.nn.functional.pad(x, (context_size, 0))
        
        # Add right context padding
        chunk_with_context = torch.nn.functional.pad(chunk_with_context, (0, context_size))
        
        # Add attention sink if enabled
        if self.use_attention_sink:
            chunk_with_sink, new_sink_cache = self.prepare_attention_sink(chunk_with_context, sink_cache)
            
            # Create attention mask that allows attention to sink frames and local context
            attention_mask = None
            if sink_cache is not None:
                sink_size = self.attention_sink_size * self.downsample_factor
                # Create base mask
                attention_mask = torch.ones(chunk_with_sink.shape[:2], device=chunk_with_sink.device)
                
                # Allow attention to sink frames with gradual transition
                sink_weights = torch.linspace(start=0.0, end=1.0, steps=int(sink_size), device=chunk_with_sink.device)
                attention_mask[:, :sink_size] = sink_weights.view(1, -1)
                
                # Allow attention to local context with gradual transition
                if left_context is not None:
                    context_weights = torch.linspace(start=1.0, end=0.0, steps=int(context_size), device=chunk_with_sink.device)
                    attention_mask[:, sink_size:sink_size + context_size] = context_weights.view(1, -1)
        else:
            chunk_with_sink = chunk_with_context
            new_sink_cache = None
            attention_mask = None
        
        # Process chunk
        outputs = self.model(
            chunk_with_sink,
            attention_mask=attention_mask,
            mask_time_indices=None,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False
        )[0]
        
        # Remove sink frames from output if used
        if self.use_attention_sink and sink_cache is not None:
            sink_frames = self.attention_sink_size
            if self.sink_warmup_frames > 0:
                sink_frames += self.sink_warmup_frames
            outputs = outputs[:, sink_frames:]
        
        # Remove context frames from output
        outputs = outputs[:, self.context_frames:-self.context_frames]
        
        # Apply smooth transition if we have previous output
        if self.last_chunk_output is not None:
            outputs = self.smooth_transition(outputs, self.last_chunk_output)
        
        # Cache current output for next chunk
        self.last_chunk_output = outputs.clone()
        
        # Handle overlap between chunks with gradual transition
        if left_context is not None:
            overlap_frames = self.chunk_overlap // self.downsample_factor
            
            # Ensure we have enough frames for overlap
            if outputs.size(1) > overlap_frames:
                # Create transition weights using proper float values
                weights = torch.cos(torch.linspace(start=0.0, end=3.14159, steps=int(overlap_frames), device=outputs.device)) * 0.5 + 0.5
                weights = weights.view(1, -1, 1)  # Shape for broadcasting
                
                # Get overlapping regions
                overlap_region = outputs[:, :overlap_frames]
                main_region = outputs[:, overlap_frames:]
                
                # Apply weighted transition
                outputs = torch.cat([
                    overlap_region * weights,
                    main_region
                ], dim=1)
                
                # Update chunk frames to account for overlap
                chunk_frames = chunk_frames - overlap_frames
        
        # Ensure outputs match expected length
        if outputs.size(1) > chunk_frames:
            # Take center frames with smooth transition
            extra = outputs.size(1) - chunk_frames
            start = extra // 2
            end = start + chunk_frames
            
            # Add smooth transition at boundaries
            if start > 0:
                transition_size = min(self.transition_frames, start)
                weights = torch.cos(torch.linspace(start=0.0, end=3.14159, steps=int(transition_size), device=outputs.device)) * 0.5 + 0.5
                weights = weights.view(1, -1, 1)
                outputs[:, :transition_size] = outputs[:, :transition_size] * weights
            
            if end < outputs.size(1):
                transition_size = min(self.transition_frames, outputs.size(1) - end)
                weights = torch.cos(torch.linspace(start=0.0, end=3.14159, steps=int(transition_size), device=outputs.device)) * 0.5 + 0.5
                weights = weights.view(1, -1, 1)
                outputs[:, -transition_size:] = outputs[:, -transition_size:] * weights
            
            outputs = outputs[:, start:end]
        
        # Final length check to match non-streaming output
        if outputs.size(1) > chunk_frames:
            # Take center frames
            extra = outputs.size(1) - chunk_frames
            start = extra // 2
            outputs = outputs[:, start:start + chunk_frames]
        
        # Update states for next chunk
        next_states = [x, new_sink_cache] if self.use_attention_sink else [x, None]
            
        return outputs, expected_frames, next_states

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
        
        # Add context padding
        context_size = self.context_frames * self.downsample_factor
        x_padded = torch.nn.functional.pad(x, (context_size, context_size))
        
        # Forward pass through model
        outputs = self.model(
            x_padded,
            attention_mask=None,
            mask_time_indices=None,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False
        )[0]
        
        # Remove context frames from output
        outputs = outputs[:, self.context_frames:-self.context_frames]
        
        # Calculate output lengths
        output_lengths = ((x_lens.float() / self.downsample_factor).floor() - 1).to(torch.int64)
        output_lengths = torch.maximum(output_lengths, torch.ones_like(output_lengths))
        
        # Ensure outputs don't exceed calculated lengths
        max_len = output_lengths.max().item()
        if outputs.size(1) > max_len:
            outputs = outputs[:, :max_len, :]
            
        return outputs, output_lengths 