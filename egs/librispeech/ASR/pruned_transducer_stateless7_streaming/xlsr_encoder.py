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
    """XLSR encoder for ASR.
    
    This encoder uses a pre-trained XLSR model from fairseq and adapts it
    for use in a transducer setup with streaming support.
    """
    
    def __init__(
        self,
        xlsr_model_name: str = "facebook/wav2vec2-large-xlsr-53",
        output_dim: int = 1024,
        layer_dropout: float = 0.1,
        finetune_last_n: int = -1,
        use_attention_sink: bool = True,
        attention_sink_size: int = 16,
    ):
        """
        Args:
            xlsr_model_name: Hugging Face model identifier for the XLSR model
            output_dim: Output dimension for the encoder
            layer_dropout: Dropout probability for the encoder layers
            finetune_last_n: Number of transformer layers to fine-tune (-1 for all)
            use_attention_sink: Whether to use attention sink for streaming
            attention_sink_size: Number of initial frames to use as attention sink
        """
        super().__init__()
        
        # Load the pre-trained XLSR model
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Config
            config = Wav2Vec2Config.from_pretrained(xlsr_model_name)
            config.layer_norm_first = True  # Important for stability
            self.xlsr = Wav2Vec2Model.from_pretrained(xlsr_model_name, config=config)
        except Exception as e:
            print(f"Error loading XLSR model {xlsr_model_name}: {e}")
            raise
        
        # Important parameters for streaming and processing
        self.encoder_embed_dim = self.xlsr.config.hidden_size
        self.encoder_embed_dim_out = output_dim
        
        # Define downsampling factor - XLSR uses 320 samples per step (20ms at 16kHz)
        self.feature_extractor_stride = 320
        self.downsample_factor = self.feature_extractor_stride
        
        # For dimensional compatibility with other parts of the code
        self.output_dim = output_dim
        
        # Add an output projection layer
        self.output_proj = nn.Linear(self.encoder_embed_dim, output_dim)
        self.layer_dropout = layer_dropout
        
        # Freeze/unfreeze parts of the model based on fine-tuning strategy
        self.finetune_last_n = finetune_last_n
        self._freeze_unused_layers()
        
        # Attention sink configuration for streaming
        self.use_attention_sink = use_attention_sink
        self.attention_sink_size = attention_sink_size
        
        # Initialize streaming state
        self.last_chunk_output = None
    
    def _freeze_unused_layers(self):
        """Freeze layers that won't be fine-tuned"""
        # Always freeze the feature extractor (as recommended in the paper)
        for param in self.xlsr.feature_extractor.parameters():
            param.requires_grad = False
        
        # Freeze transformer layers based on finetune_last_n
        if self.finetune_last_n >= 0:
            layers_to_freeze = self.xlsr.encoder.layers[:-self.finetune_last_n]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
    
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
        self.left_context_buffer = []  # Store left context chunks
        self.past_context = None  # Add past context for streaming
        self.is_streaming = False  # Track streaming state

    def initialize_streaming(self, device=None):
        """Initialize streaming state for inference
        
        Args:
            device: The device to initialize states on
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.reset_streaming_state()
        self.is_streaming = True
        
        # Initialize empty left context buffer
        self.left_context_buffer = []
        
        # Initialize empty attention sink cache
        self.attention_sink_cache = None
        
        # Initialize last chunk output as None
        self.last_chunk_output = None
        
        return self.get_init_state(device)
    
    def update_streaming_state(self, states: List[torch.Tensor]):
        """Update streaming state from the provided states
        
        Args:
            states: States returned from the previous streaming_forward call
        """
        if states is None or len(states) < 2:
            return
        
        # Update left context buffer if needed
        if states[0] is not None:
            if len(self.left_context_buffer) >= self.left_context_chunks:
                self.left_context_buffer.pop(0)
            self.left_context_buffer.append(states[0])
        
        # Update attention sink cache
        self.attention_sink_cache = states[1]

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
        sink_size = min(
            self.attention_sink_size * self.downsample_factor, 
            chunk.size(1)  # Ensure we don't exceed chunk size
        )
        
        if sink_cache is None:
            # For first chunk, use beginning of chunk as sink
            if chunk.size(1) >= sink_size:
                new_sink_cache = chunk[:, :sink_size].detach().clone()
            else:
                # If chunk is smaller than sink size, use the entire chunk
                new_sink_cache = chunk.detach().clone()
            # First time, don't add sink to input
            chunk_with_sink = chunk
        else:
            # Concatenate sink with current chunk
            chunk_with_sink = torch.cat([sink_cache, chunk], dim=1)
            # Update sink cache with end of current chunk
            if chunk.size(1) >= sink_size:
                new_sink_cache = chunk[:, -sink_size:].detach().clone()
            else:
                # If chunk is smaller than sink size, use the entire chunk
                new_sink_cache = chunk.detach().clone()
            
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
        context = self.left_context_buffer[-min(self.left_context_chunks, len(self.left_context_buffer)):]
        
        # Ensure all context tensors have the same batch size
        batch_size = chunk.size(0)
        valid_context = []
        
        for ctx in context:
            if ctx.size(0) == batch_size:
                valid_context.append(ctx)
            else:
                # If batch size doesn't match, we need to adjust
                if ctx.size(0) < batch_size:
                    # Expand by repeating
                    ctx = ctx.repeat(batch_size // ctx.size(0) + 1, 1)[:batch_size]
                else:
                    # Trim
                    ctx = ctx[:batch_size]
                valid_context.append(ctx)
        
        if not valid_context:
            return chunk
            
        context_tensor = torch.cat(valid_context, dim=1)
        
        return torch.cat([context_tensor, chunk], dim=1)

    def init_states(self, batch_size=1):
        """Initialize states for streaming inference.
        
        Args:
            batch_size: The batch size for streaming inference
            
        Returns:
            A dictionary containing the initialized states
        """
        states = {}
        states["last_chunk_output"] = None
        states["chunk_count"] = 0
        states["batch_size"] = batch_size
        return states

    def update_states(self, states, encoder_out):
        """Update the states for the next chunk.
        
        Args:
            states: The current states
            encoder_out: The encoder output from the current chunk
            
        Returns:
            Updated states for the next chunk
        """
        # Store last chunk output for smooth transitions
        states["last_chunk_output"] = encoder_out.detach().clone()
        states["chunk_count"] += 1
        states["batch_size"] = encoder_out.size(0)
        return states

    def streaming_forward(
        self, 
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states=None,
        chunk_size: int = None,
        simulate_streaming: bool = False,
    ) -> [torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: A 3-D tensor of shape (N, T, C)
            x_lens: A 1-D tensor of shape (N,)
            states: States from previous chunk for streaming ASR
            chunk_size: Size of each chunk for streaming processing
            simulate_streaming: Whether to simulate streaming by using chunk-wise processing
            
        Returns:
            A tuple containing:
            - encoder output of shape (N, T, encoder_dim)
            - encoder output lengths of shape (N,)
            - updated states for next chunk
        """
        if states is None:
            states = self.init_states(x.size(0))
        
        # Check if we need to reset or initialize states based on batch size change
        if states.get("batch_size", -1) != x.size(0):
            logging.debug(f"Batch size changed from {states.get('batch_size', -1)} to {x.size(0)}. Reinitializing states.")
            states = self.init_states(x.size(0))
            
        # Ensure input is float and in correct shape
        x = x.float()
        if x.ndim == 3:
            x = x.squeeze(-1)
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        
        # Make sure batch size is consistent
        batch_size = x.size(0)
        
        # Validate x_lens and ensure proper batch size
        if isinstance(x_lens, int):
            x_lens = torch.tensor([x_lens], device=x.device)
        elif not isinstance(x_lens, torch.Tensor):
            x_lens = torch.tensor(x_lens, device=x.device)
        
        # Ensure x_lens has the same batch dimension as x
        if x_lens.size(0) != batch_size:
            if x_lens.size(0) == 1:
                # Broadcast single length to match batch size
                x_lens = x_lens.expand(batch_size)
            else:
                # Trim or pad x_lens to match batch size
                if x_lens.size(0) > batch_size:
                    x_lens = x_lens[:batch_size]
                else:
                    padding = torch.ones(batch_size - x_lens.size(0), device=x_lens.device, 
                                         dtype=x_lens.dtype) * x_lens[-1]
                    x_lens = torch.cat([x_lens, padding])
        
        # Ensure x_lens is on the same device as x
        if x_lens.device != x.device:
            x_lens = x_lens.to(x.device)
        
        # Clamp values silently since inputs are already normalized
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Store original input for context updates
        original_x = x.clone()
        
        # Calculate expected output length for this chunk
        expected_frames = ((x_lens.float() / self.downsample_factor).floor()).to(torch.int64)
        expected_frames = torch.maximum(expected_frames, torch.ones_like(expected_frames))
        
        # Get audio sink from states if available
        audio_sink = None
        if states is not None and len(states) > 1 and states[1] is not None:
            audio_sink = states[1]  # Get audio sink from states
            
            # Check if audio_sink is on the same device
            if audio_sink.device != x.device:
                audio_sink = audio_sink.to(x.device)
                
            # Check data type compatibility
            if audio_sink.dtype != x.dtype:
                audio_sink = audio_sink.to(dtype=x.dtype)
                
            # Add audio sink to current chunk if available
            if audio_sink is not None and self.use_attention_sink:
                # Verify dimensions are compatible for concatenation
                if audio_sink.dim() != x.dim():
                    logging.warning(f"Dimension mismatch: audio_sink {audio_sink.shape}, x {x.shape}")
                    audio_sink = audio_sink.reshape(x.size(0), -1)
                    
                x = torch.cat([audio_sink, x], dim=1)
                # Adjust lens to account for the added audio sink
                x_lens = x_lens + audio_sink.size(1)
        
        # Process through XLSR model
        outputs = self.xlsr(
            x,
            attention_mask=None,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False
        )[0]
        
        # Calculate output lengths considering downsampling
        output_lengths = ((x_lens.float() / self.downsample_factor).floor()).to(torch.int64)
        output_lengths = torch.maximum(output_lengths, torch.ones_like(output_lengths))
        
        # Remove audio sink frames from output if they were added
        if audio_sink is not None and self.use_attention_sink:
            # Calculate how many encoder frames the audio sink corresponds to
            sink_frames = audio_sink.size(1) // self.downsample_factor
            if outputs.size(1) > sink_frames:
                outputs = outputs[:, sink_frames:]
                output_lengths = output_lengths - sink_frames
        
        # Prepare new audio sink from end of current chunk - use 40% of chunk
        new_audio_sink = None
        if self.use_attention_sink:
            # Calculate attention sink size in audio samples
            audio_sink_samples = self.attention_sink_size * self.downsample_factor
            if original_x.size(1) >= audio_sink_samples:
                new_audio_sink = original_x[:, -audio_sink_samples:].detach().clone()
            else:
                new_audio_sink = original_x.detach().clone()
        
        # Ensure outputs don't exceed calculated lengths
        max_len = expected_frames.max().item()
        if outputs.size(1) > max_len:
            # Log the trimming for diagnostic purposes
            logging.debug(f"Trimming encoder output from {outputs.size(1)} to {max_len} frames")
            outputs = outputs[:, :max_len]
        
        # Apply smooth transition if we have previous output
        if hasattr(self, 'last_chunk_output') and self.last_chunk_output is not None:
            # Check device and dtype compatibility
            if self.last_chunk_output.device != outputs.device:
                self.last_chunk_output = self.last_chunk_output.to(outputs.device)
            if self.last_chunk_output.dtype != outputs.dtype:
                self.last_chunk_output = self.last_chunk_output.to(dtype=outputs.dtype)
            
            # Handle batch size mismatch between cached and current outputs
            current_batch_size = outputs.size(0)
            cached_batch_size = self.last_chunk_output.size(0)
            
            if current_batch_size != cached_batch_size:
                # If batch sizes don't match, we need a different approach
                logging.debug(f"Batch size mismatch: current={current_batch_size}, cached={cached_batch_size}")
                # Skip blending for this chunk but update cache for next time
                self.last_chunk_output = outputs.detach().clone()
            else:
                # Simple crossfade at chunk boundaries with matching batch sizes
                # Calculate overlap based on chunk_overlap
                # For 40% overlap, we should blend approximately 40% of the frames
                overlap_frames = min(4, max(2, int(outputs.size(1) * 0.4)), self.last_chunk_output.size(1))
                
                if outputs.size(1) > overlap_frames and self.last_chunk_output.size(1) > overlap_frames:
                    # Create crossfade weights - more gradual transition
                    weights = torch.linspace(0.0, 1.0, steps=overlap_frames, device=outputs.device)
                    weights = weights.view(1, -1, 1)  # Shape for broadcasting
                    
                    # Get transition regions
                    prev_end = self.last_chunk_output[:, -overlap_frames:].clone()
                    curr_start = outputs[:, :overlap_frames].clone()
                    
                    # Blend transition
                    blended = weights * curr_start + (1 - weights) * prev_end
                    
                    # Apply transition
                    outputs = outputs.clone()
                    outputs[:, :overlap_frames] = blended
        
        # Cache current output for next chunk
        self.last_chunk_output = outputs.detach().clone()
        
        # Update states for next chunk
        states = self.update_states(states, outputs)
            
        return outputs, output_lengths, states

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
        if x.ndim == 3:  # (batch, time, channel)
            x = x.squeeze(-1)
        elif x.ndim == 1:  # (time,)
            x = x.unsqueeze(0)  # Add batch dimension
        
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        
        batch_size = x.size(0)
        
        # Ensure x_lens has the right shape
        if isinstance(x_lens, int):
            x_lens = torch.tensor([x_lens], device=x.device)
        elif not isinstance(x_lens, torch.Tensor):
            x_lens = torch.tensor(x_lens, device=x.device)
            
        # Ensure x_lens has the same batch dimension as x
        if x_lens.size(0) != batch_size:
            if x_lens.size(0) == 1:
                # Broadcast single length to match batch size
                x_lens = x_lens.expand(batch_size)
            else:
                # Trim or pad x_lens to match batch size
                if x_lens.size(0) > batch_size:
                    x_lens = x_lens[:batch_size]
                else:
                    padding = torch.ones(batch_size - x_lens.size(0), device=x_lens.device, 
                                         dtype=x_lens.dtype) * x_lens[-1]
                    x_lens = torch.cat([x_lens, padding])
        
        # Clamp values silently since inputs are already normalized
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Create attention mask
        attention_mask = torch.ones_like(x, dtype=torch.long)
        for i in range(batch_size):
            if i < x_lens.size(0):  # Safety check
                attention_mask[i, x_lens[i]:] = 0
        
        if is_pre_training:
            # During pre-training, use full sequence without chunking
            outputs = self.xlsr(
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
                        chunk_output = self.xlsr(
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
                # During inference, use streaming_forward
                encoder_out, encoder_out_lens, _ = self.streaming_forward(x, x_lens)
            
            return encoder_out, encoder_out_lens 