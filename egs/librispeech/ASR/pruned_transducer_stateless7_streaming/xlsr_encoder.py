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
    def __init__(
        self, 
        model_name: str = "TalTechNLP/xls-r-300m-et",
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
        self.past_context = None  # Add past context for streaming
        self.is_streaming = False  # Track streaming state

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
        if previous_output is None:
            return current_output
            
        # Calculate actual overlap size based on available frames
        overlap_size = min(
            previous_output.size(1),  # Previous chunk size
            current_output.size(1),   # Current chunk size
            self.frames_per_chunk     # Maximum overlap
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
        # Ensure input is float and in correct shape
        x = x.float()
        if x.ndim == 3:
            x = x.squeeze(-1)
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        
        # Clamp values silently since inputs are already normalized
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Get batch size from input and ensure x_lens has same batch size
        batch_size = x.size(0)
        if x_lens.size(0) != batch_size:
            logging.warning(f"Mismatch between batch_size ({batch_size}) and x_lens.size(0) ({x_lens.size(0)})")
            
            # Adjust x_lens to match batch size
            if x_lens.size(0) > batch_size:
                # If x_lens has more elements, truncate
                x_lens = x_lens[:batch_size]
            else:
                # If x_lens has fewer elements, expand with copies of first element
                if x_lens.size(0) > 0:
                    padding = x_lens[0].repeat(batch_size - x_lens.size(0))
                    x_lens = torch.cat([x_lens, padding])
                else:
                    # Create default lens based on input size
                    x_lens = torch.full((batch_size,), x.size(1), device=x.device, dtype=torch.long)
        
        # Calculate expected output length before adding context
        # Account for overlap and context frames precisely
        chunk_overlap = self.decode_chunk_size // 2
        context_frames = self.attention_sink_size if self.use_attention_sink else 0
        if self.left_context_chunks > 0:
            context_frames += self.left_context_chunks * (self.decode_chunk_size // self.downsample_factor)
            
        # Calculate effective input length considering overlap and context
        if states is not None:
            # For subsequent chunks, adjust for overlap
            effective_input_len = x_lens - chunk_overlap
            # Add small correction factor for chunk boundary alignment
            boundary_correction = self.downsample_factor // 16  # 20 samples at 16kHz
            effective_input_len = effective_input_len + boundary_correction
        else:
            # For first chunk, use full length
            effective_input_len = x_lens
            
        # Calculate expected frames precisely using ceil instead of floor
        expected_frames = ((effective_input_len.float() / self.downsample_factor).ceil()).to(torch.int64)
        expected_frames = torch.maximum(expected_frames, torch.ones_like(expected_frames))
        
        # Process each batch item separately to avoid dimension issues
        all_outputs = []
        all_sink_caches = []
        
        for b in range(batch_size):
            single_x = x[b:b+1]  # Keep batch dimension
            single_len = x_lens[b]
            
            # Get batch-specific states if available
            batch_states = None if states is None else [
                None if s is None else s[b:b+1] for s in states
            ]
            
            # Add left context if available for this batch item
            if batch_states is not None and len(batch_states) > 0 and batch_states[0] is not None:
                left_context = batch_states[0]
                single_x = torch.cat([left_context, single_x], dim=1)
            
            # Add attention sink if enabled and available for this batch item
            if self.use_attention_sink and batch_states is not None and len(batch_states) > 1 and batch_states[1] is not None:
                sink = batch_states[1]
                single_x = torch.cat([sink, single_x], dim=1)
            
            # Process through XLSR model
            attention_mask = torch.ones_like(single_x, dtype=torch.long)
            if single_len < single_x.size(1):
                attention_mask[0, single_len:] = 0
            
            outputs = self.model(
                single_x,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False
            )[0]
            
            # Create next states for this batch item
            # States[0] = left context for next chunk, States[1] = attention sink
            new_left_context = single_x[:, -self.chunk_overlap:] if self.left_context_chunks > 0 else None
            new_sink_cache = outputs[:, -self.attention_sink_size:] if self.use_attention_sink else None
            
            # Store individual outputs and states
            all_outputs.append(outputs)
            if new_sink_cache is not None:
                all_sink_caches.append(new_sink_cache)
        
        # Combine batch results - handle potential different sequence lengths
        max_seq_len = max(output.size(1) for output in all_outputs)
        padded_outputs = []
        
        for output in all_outputs:
            if output.size(1) < max_seq_len:
                padding = torch.zeros(
                    1, max_seq_len - output.size(1), output.size(2),
                    device=output.device, dtype=output.dtype
                )
                padded_outputs.append(torch.cat([output, padding], dim=1))
            else:
                padded_outputs.append(output)
        
        # Concatenate batch dimension
        combined_output = torch.cat(padded_outputs, dim=0)
        
        # Create combined sink cache
        combined_sink_cache = None
        if all_sink_caches:
            combined_sink_cache = torch.cat(all_sink_caches, dim=0)
        
        # Combine left context and sink cache into states
        next_states = [
            None if self.left_context_chunks == 0 else x[:, -self.chunk_overlap:],
            combined_sink_cache
        ]
        
        # Ensure output has consistent length
        if combined_output.size(1) > max(expected_frames):
            # Handle case where output is longer than expected
            combined_output = combined_output[:, :max(expected_frames)]
        elif combined_output.size(1) < max(expected_frames):
            # Handle case where output is shorter than expected
            padding = torch.zeros(
                batch_size, 
                max(expected_frames) - combined_output.size(1), 
                combined_output.size(2),
                device=combined_output.device, 
                dtype=combined_output.dtype
            )
            combined_output = torch.cat([combined_output, padding], dim=1)
        
        return combined_output, expected_frames, next_states

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
        batch_size = x.size(0)
        
        # Ensure input is float and in correct shape
        x = x.float()
        if x.ndim == 3:  # (batch, time, channel)
            x = x.squeeze(-1)
        elif x.ndim == 1:  # (time,)
            x = x.unsqueeze(0)  # Add batch dimension
        
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        
        # Clamp values silently since inputs are already normalized
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Create attention mask
        attention_mask = torch.ones_like(x, dtype=torch.long)
        for i in range(batch_size):
            if i < x_lens.size(0):  # Make sure we don't go out of bounds
                attention_mask[i, x_lens[i]:] = 0
            else:
                # Handle case where x_lens size doesn't match batch size
                logging.warning(f"Mismatch between batch_size ({batch_size}) and x_lens.size(0) ({x_lens.size(0)})")
                # Use the first length if we can't access the correct one
                if x_lens.size(0) > 0:
                    attention_mask[i, x_lens[0]:] = 0
        
        # Ensure x_lens has the same batch size as x
        if x_lens.size(0) != batch_size:
            logging.warning(f"Fixing x_lens batch size from {x_lens.size(0)} to {batch_size}")
            if x_lens.size(0) > batch_size:
                # Take only first batch_size elements
                x_lens = x_lens[:batch_size]
            else:
                # Pad with copies of the first length
                pad_lens = x_lens[0].expand(batch_size - x_lens.size(0))
                x_lens = torch.cat([x_lens, pad_lens])
        
        if is_pre_training:
            # During pre-training, use full sequence without chunking
            outputs = self.model(
                x,
                attention_mask=attention_mask,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False
            )[0]
            
            # Calculate output lengths
            encoder_out_lens = ((x_lens.float() / self.downsample_factor).floor()).to(torch.int64)
            encoder_out_lens = torch.maximum(encoder_out_lens, torch.ones_like(encoder_out_lens))
            
            return outputs, encoder_out_lens
        else:
            # Use streaming mode with chunks
            if self.training:
                # During training with streaming, simulate chunked processing
                outputs = []
                current_lengths = []
                
                for i in range(batch_size):
                    # Process each sequence in the batch
                    seq_len = x_lens[i]
                    sequence = x[i, :seq_len]
                    
                    # Process in chunks
                    chunk_outputs = []
                    pos = 0
                    self.past_context = None  # Reset past context for each sequence
                    
                    while pos < seq_len:
                        # Get current chunk
                        chunk_size = min(self.decode_chunk_size, seq_len - pos)
                        chunk = sequence[pos:pos + chunk_size]
                        
                        # Add left context if available
                        if self.past_context is not None:
                            chunk = torch.cat([self.past_context, chunk], dim=0)
                        
                        # Process chunk
                        chunk_output = self.model(
                            chunk.unsqueeze(0),
                            attention_mask=torch.ones_like(chunk).unsqueeze(0),
                            output_hidden_states=False,
                            output_attentions=False,
                            return_dict=False
                        )[0]
                        
                        # Add attention sink if enabled
                        if self.use_attention_sink:
                            chunk_output = torch.cat([
                                self.attention_sink_cache.unsqueeze(0).expand(1, -1, -1) if self.attention_sink_cache is not None else chunk_output[:, :self.attention_sink_size],
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
                # During inference, use streaming_forward but ensure batch size consistency
                encoder_out, encoder_out_lens, _ = self.streaming_forward(x, x_lens)
                
                # Ensure consistent batch size output
                if encoder_out.size(0) != batch_size:
                    logging.warning(f"Streaming forward returned inconsistent batch size: {encoder_out.size(0)}, expected {batch_size}")
                    if encoder_out.size(0) > batch_size:
                        # Trim extra batch items
                        encoder_out = encoder_out[:batch_size]
                        encoder_out_lens = encoder_out_lens[:batch_size]
                    else:
                        # Pad with zeros for missing batch items
                        padding = torch.zeros(
                            batch_size - encoder_out.size(0),
                            encoder_out.size(1),
                            encoder_out.size(2),
                            device=encoder_out.device,
                            dtype=encoder_out.dtype
                        )
                        encoder_out = torch.cat([encoder_out, padding], dim=0)
                        # Pad lengths with copies of first length
                        pad_lens = encoder_out_lens[0].repeat(batch_size - encoder_out_lens.size(0))
                        encoder_out_lens = torch.cat([encoder_out_lens, pad_lens])
        
        return encoder_out, encoder_out_lens 