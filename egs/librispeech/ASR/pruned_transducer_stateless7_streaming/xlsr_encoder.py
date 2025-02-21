import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
import time
import logging
import random
import numpy as np
from typing import Optional, List, Tuple

# We use transformers library to load the XLSR model
# Make sure to install transformers: pip install transformers

class XLSREncoder(EncoderInterface):
    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-xls-r-300m",
        decode_chunk_size: int = 8000,  # Default to 0.5s at 16kHz
        chunk_overlap: int = None,  # Will be set to decode_chunk_size // 2
        use_attention_sink: bool = True,
        attention_sink_size: int = 4,  # Number of attention sink frames
        frame_duration: float = 0.025,  # 25ms per frame
        frame_stride: float = 0.020,  # 20ms stride
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
        
        # Chunk sizes in frames (320ms, 640ms, 1280ms, 2560ms)
        self.chunk_sizes = [
            int(0.32 / frame_stride),  # 16 frames
            int(0.64 / frame_stride),  # 32 frames
            int(1.28 / frame_stride),  # 64 frames
            int(2.56 / frame_stride)   # 128 frames
        ]
        
        # Convert chunk sizes to samples
        self.chunk_sizes_samples = [
            size * self.downsample_factor for size in self.chunk_sizes
        ]
        
        # Streaming parameters
        self.decode_chunk_size = decode_chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else decode_chunk_size // 2
        
        # Attention sink parameters (from paper)
        self.use_attention_sink = use_attention_sink
        self.attention_sink_size = attention_sink_size
        
        # Initialize streaming state
        self.reset_streaming_state()
        
        # Ensure output_dim matches joiner input
        self.output_dim = 1024  # For XLS-R 300M

    def reset_streaming_state(self):
        """Reset all streaming state variables"""
        self.cached_features = None
        self.cached_len = 0
        self.current_chunk_size = self.decode_chunk_size
        self.last_chunk_latency = 0
        self.streaming_state = None
        self.attention_sink_cache = None

    def get_random_chunk_size(self):
        """Get random chunk size during training"""
        return random.choice(self.chunk_sizes_samples)

    def prepare_chunks(self, x: torch.Tensor, chunk_size: int) -> list[torch.Tensor]:
        """Prepare chunks with optional attention sink"""
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        batch_size, seq_len = x.shape
        chunks = []
        current = 0
        
        while current < seq_len:
            end = min(current + chunk_size, seq_len)
            chunk = x[:, current:end]
            
            # Ensure minimum chunk size
            if chunk.shape[1] < 20:  # minimum size to avoid model issues
                break
                
            chunks.append(chunk)
            current += chunk_size - self.chunk_overlap
            
        return chunks

    def process_chunk(self, chunk: torch.Tensor, attention_sink: torch.Tensor = None) -> torch.Tensor:
        """Process a single chunk with optional attention sink"""
        assert chunk.ndim == 2, f"Expected 2D input (batch, time), got shape {chunk.shape}"
        
        if attention_sink is not None and self.use_attention_sink:
            # Concatenate attention sink at the start of the chunk
            chunk = torch.cat([attention_sink, chunk], dim=1)
        
        outputs = self.model(
            chunk,
            attention_mask=None,
            mask_time_indices=None,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False
        )[0]
        
        if attention_sink is not None and self.use_attention_sink:
            # Remove attention sink frames from output
            sink_frames = attention_sink.shape[1] // self.downsample_factor
            outputs = outputs[:, sink_frames:]
            
        return outputs

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, time) or (batch, time, 1)
            x_lens: Length of each sequence in the batch
        Returns:
            (output, output_lens)
        """
        # Ensure input is float and in correct shape
        x = x.float()
        
        # Handle various input shapes
        if x.ndim == 3:  # (batch, time, channel)
            x = x.squeeze(-1)
        elif x.ndim == 1:  # (time,)
            x = x.unsqueeze(0)  # Add batch dimension
        
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        
        # Clamp values silently since inputs are already normalized
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Calculate output sequence lengths
        output_lengths = ((x_lens.float() / self.downsample_factor).floor() - 1).to(torch.int64)
        output_lengths = torch.maximum(output_lengths, torch.ones_like(output_lengths))
        
        # For non-streaming forward pass, process directly
        outputs = self.model(
            x,
            attention_mask=None,
            mask_time_indices=None,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False
        )[0]
        
        # Ensure outputs don't exceed calculated lengths
        max_len = output_lengths.max().item()
        if outputs.size(1) > max_len:
            outputs = outputs[:, :max_len, :]
            
        return outputs, output_lengths 

    def streaming_forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        states: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Streaming forward pass that's compatible with k2's FSA-based decoding
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
        
        # Calculate output sequence lengths
        output_lengths = ((x_lens.float() / self.downsample_factor).floor() - 1).to(torch.int64)
        output_lengths = torch.maximum(output_lengths, torch.ones_like(output_lengths))
        
        # Process with attention sink if enabled
        if self.use_attention_sink and states is not None:
            # Use last chunk's output as attention sink
            sink_size = self.attention_sink_size * self.downsample_factor
            if states[0] is not None:
                x = torch.cat([states[0][:, -sink_size:], x], dim=1)
                x_lens = x_lens + sink_size
        
        # Forward pass through model
        outputs = self.model(
            x,
            attention_mask=None,
            mask_time_indices=None,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=False
        )[0]
        
        # Update states for next chunk
        next_states = [x] if self.use_attention_sink else [None]
        
        # Remove attention sink frames from output if used
        if self.use_attention_sink and states is not None and states[0] is not None:
            sink_frames = self.attention_sink_size
            outputs = outputs[:, sink_frames:]
            output_lengths = output_lengths - sink_frames
        
        # Ensure outputs don't exceed calculated lengths
        max_len = output_lengths.max().item()
        if outputs.size(1) > max_len:
            outputs = outputs[:, :max_len, :]
            
        return outputs, output_lengths, next_states

    def get_init_state(self, device: torch.device = None) -> List[torch.Tensor]:
        """Get initial states for streaming inference"""
        if device is None:
            device = next(self.parameters()).device
        return [None]  # Initial state is None since we'll build it from first chunk 

    @property
    def encoder_dim(self) -> int:
        """Return encoder output dimension for joiner network"""
        return self.output_dim  # 1024 for XLS-R 300M 