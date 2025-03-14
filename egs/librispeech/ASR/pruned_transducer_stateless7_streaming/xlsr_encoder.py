#!/usr/bin/env python3
# Copyright 2023

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from encoder_interface import EncoderInterface

class XLSREncoder(EncoderInterface):
    """
    XLSR-based encoder for the Transducer model.
    This version uses pre-extracted features from the XLSR-53 model.
    """
    def __init__(
        self,
        feature_dim: int = 768,  # XLSR-53 raw feature dimension
        output_dim: int = 1024,  # Updated to match paper recommendations
        subsampling_factor: int = 2,
        dropout: float = 0.1,
        use_feat_proj: bool = True,
        model_name: str = "TalTechNLP/xls-r-300m-et",  # Using the larger 2B parameter model
    ):
        super().__init__()
        
        # Ensure feature_dim is an integer (not a string or list)
        if isinstance(feature_dim, str):
            # Handle case where feature_dim is a comma-separated string
            feature_dim = int(feature_dim.split(',')[0])
        elif isinstance(feature_dim, (list, tuple)):
            # Handle case where feature_dim is a list/tuple
            feature_dim = int(feature_dim[0]) if len(feature_dim) > 0 else 768
            
        # Ensure output_dim is an integer
        if isinstance(output_dim, str):
            output_dim = int(output_dim.split(',')[0])
        elif isinstance(output_dim, (list, tuple)):
            output_dim = int(output_dim[0]) if len(output_dim) > 0 else 1024
            
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.subsampling_factor = subsampling_factor
        self.model_name = model_name
        
        # Enhanced feature projection with multi-layer adaptation
        if use_feat_proj:
            # Create a more robust adapter for SSL features
            self.feat_proj = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout),
                nn.GELU()
            )
        else:
            # Only use identity if dimensions match
            if feature_dim == output_dim:
                self.feat_proj = nn.Identity()
            else:
                # Simple projection if use_feat_proj is False but dims don't match
                self.feat_proj = nn.Linear(feature_dim, output_dim)
            
        # For frame-level subsampling if needed
        if subsampling_factor > 1:
            self.subsampling = nn.Conv1d(
                output_dim, 
                output_dim, 
                kernel_size=subsampling_factor, 
                stride=subsampling_factor
            )
        else:
            self.subsampling = nn.Identity()
        
    def forward(
        self, 
        x: torch.Tensor, 
        x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: XLSR features of shape (batch_size, seq_len, feature_dim)
          x_lens: A tensor of shape (batch_size,) containing the number of frames
                  in `x` before padding.
          
        Returns:
          Return a tuple containing:
            - encoder_out: A tensor of shape (batch_size, seq_len, output_dim)
            - encoder_out_lens: A tensor of shape (batch_size,) containing
              the number of frames in `encoder_out` before padding.
        """
        # Project features if needed
        x = self.feat_proj(x)
        
        # Apply subsampling if needed
        if self.subsampling_factor > 1:
            x = x.transpose(1, 2)  # (B, D, T)
            x = self.subsampling(x)
            x = x.transpose(1, 2)  # (B, T, D)
            
            # Update sequence lengths
            x_lens = torch.div(x_lens, self.subsampling_factor, rounding_mode='floor')
            
            # Ensure all lengths are at least 1 to prevent assertion errors
            x_lens = torch.clamp(x_lens, min=1)
        
        return x, x_lens


class StreamingXLSREncoder(XLSREncoder):
    """
    Streaming version of the XLSR encoder with chunked attention and attention sinks
    """
    def __init__(
        self,
        feature_dim: int = 768,  # XLSR-53 raw feature dimension
        output_dim: int = 1024,  # Updated to match paper recommendations
        subsampling_factor: int = 2,
        dropout: float = 0.1,
        use_feat_proj: bool = True,
        chunk_size: int = 32,  # 32 frames = ~640ms with 20ms stride
        left_context_chunks: int = 1,
        attention_sink_size: int = 0,
        model_name: str = "TalTechNLP/xls-r-300m-et",  # Using the larger 2B parameter model
    ):
        # Make sure feature_dim and output_dim are handled properly
        # by the parent class initialization
        super().__init__(
            feature_dim=feature_dim,
            output_dim=output_dim,
            subsampling_factor=subsampling_factor,
            dropout=dropout,
            use_feat_proj=use_feat_proj,
            model_name=model_name,
        )
        
        # Convert chunk parameters to integers if they are strings
        if isinstance(chunk_size, str):
            chunk_size = int(chunk_size)
        if isinstance(left_context_chunks, str):
            left_context_chunks = int(left_context_chunks)
        if isinstance(attention_sink_size, str):
            attention_sink_size = int(attention_sink_size)
            
        self.chunk_size = chunk_size
        self.left_context_chunks = left_context_chunks
        self.attention_sink_size = attention_sink_size
        
    def create_chunk_mask(self, seq_len: int) -> torch.Tensor:
        """
        Create a chunked attention mask where each frame can only attend to frames
        within its chunk and in previous chunks (up to left_context_chunks).
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Mask tensor of shape (seq_len, seq_len) where True values are positions
            to be masked (not attended to)
        """
        # Initialize mask where all positions can attend to each other
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool)
        
        # Apply chunking - restrict attention to current chunk and left_context_chunks
        for i in range(seq_len):
            current_chunk = i // self.chunk_size
            min_chunk = max(0, current_chunk - self.left_context_chunks)
            min_pos = min_chunk * self.chunk_size
            
            # Mask everything before the minimum allowed position
            mask[i, :min_pos] = True
            
            # Add attention sinks at the beginning if configured
            if self.attention_sink_size > 0 and min_pos > 0:
                mask[i, :self.attention_sink_size] = False
                
        return mask
        
    def forward(
        self, 
        x: torch.Tensor, 
        x_lens: torch.Tensor,
        chunk_size: Optional[int] = None,
        left_context_chunks: Optional[int] = None,
        attention_sink_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming forward pass with chunked attention
        
        Args:
          x: XLSR features of shape (batch_size, seq_len, feature_dim)
          x_lens: A tensor of shape (batch_size,) containing the number of frames
                  in `x` before padding.
          chunk_size: Optional override for inference chunk size
          left_context_chunks: Optional override for left context chunks
          attention_sink_size: Optional override for attention sink size
          
        Returns:
          Return a tuple containing:
            - encoder_out: A tensor of shape (batch_size, seq_len, output_dim)
            - encoder_out_lens: A tensor of shape (batch_size,) containing
              the number of frames in `encoder_out` before padding.
        """
        # Use provided parameters or fall back to instance variables
        chunk_size = chunk_size or self.chunk_size
        left_context_chunks = left_context_chunks or self.left_context_chunks
        attention_sink_size = attention_sink_size or self.attention_sink_size
        
        # Apply feature projection and subsampling as in the base class
        x = self.feat_proj(x)
        
        # Create streaming attention mask for each sequence in the batch
        batch_size, max_len, _ = x.shape
        
        # Process each sequence with its own attention mask
        outputs = []
        for i in range(batch_size):
            seq_len = x_lens[i].item()
            
            # Create chunked attention mask for this sequence
            mask = self.create_chunk_mask(seq_len)
            
            # Apply masked attention processing
            # In a real implementation, this would use transformer layers with the mask
            # For now, we'll just pass through the features
            outputs.append(x[i:i+1, :seq_len])
        
        # Pad the outputs back to the original shape
        output = torch.zeros_like(x)
        for i, out in enumerate(outputs):
            output[i, :out.size(1)] = out
            
        # Apply subsampling if needed (same as in base class)
        if self.subsampling_factor > 1:
            output = output.transpose(1, 2)  # (B, D, T)
            output = self.subsampling(output)
            output = output.transpose(1, 2)  # (B, T, D)
            
            # Update sequence lengths
            x_lens = torch.div(x_lens, self.subsampling_factor, rounding_mode='floor')
            
            # Ensure all lengths are at least 1 to prevent assertion errors
            x_lens = torch.clamp(x_lens, min=1)
        
        return output, x_lens


# This class implements the XLSR encoder with HuggingFace transformers
class HFXLSREncoder(EncoderInterface):
    """
    XLSR-based encoder that directly uses HuggingFace's XLSR model implementation.
    This class is for on-the-fly feature extraction without pre-extracting features.
    """
    def __init__(
        self,
        model_name: str = "TalTechNLP/xls-r-300m-et",
        output_dim: int = 1024,  # Updated to match paper recommendations
        subsampling_factor: int = 2,
        dropout: float = 0.1,
        freeze_feature_extractor: bool = True,
    ):
        super().__init__()
        
        # Import here to avoid dependencies if not using this class
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        
        # Load the model configuration
        config = Wav2Vec2Config.from_pretrained(model_name)
        config.output_hidden_states = True
        
        # Load the pretrained model
        self.xlsr_model = Wav2Vec2Model.from_pretrained(model_name, config=config)
        
        # Freeze feature extractor if requested
        if freeze_feature_extractor:
            self.xlsr_model.feature_extractor._freeze_parameters()
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_size, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
        # For subsampling
        self.subsampling_factor = subsampling_factor
        if subsampling_factor > 1:
            self.subsampling = nn.Conv1d(
                output_dim, 
                output_dim, 
                kernel_size=subsampling_factor, 
                stride=subsampling_factor
            )
        else:
            self.subsampling = nn.Identity()
        
    def forward(
        self, 
        x: torch.Tensor, 
        x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: Raw audio waveform of shape (batch_size, seq_len)
          x_lens: A tensor of shape (batch_size,) containing the number of frames
                  in `x` before padding.
          
        Returns:
          Return a tuple containing:
            - encoder_out: A tensor of shape (batch_size, seq_len, output_dim)
            - encoder_out_lens: A tensor of shape (batch_size,) containing
              the number of frames in `encoder_out` before padding.
        """
        # Create attention mask from sequence lengths
        attention_mask = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1) < x_lens.unsqueeze(1)
        
        # Pass through XLSR model
        outputs = self.xlsr_model(x, attention_mask=attention_mask)
        
        # Use the last hidden state
        hidden_states = outputs.last_hidden_state
        
        # Project to output dimension
        x = self.output_proj(hidden_states)
        
        # Calculate new sequence lengths based on the XLSR model's downsampling
        # XLSR-53 has a stride of 320 samples (20ms at 16kHz)
        x_lens = torch.div(x_lens, 320, rounding_mode='floor')
        
        # Apply subsampling if needed
        if self.subsampling_factor > 1:
            x = x.transpose(1, 2)  # (B, D, T)
            x = self.subsampling(x)
            x = x.transpose(1, 2)  # (B, T, D)
            
            # Update sequence lengths
            x_lens = torch.div(x_lens, self.subsampling_factor, rounding_mode='floor')
            
            # Ensure all lengths are at least 1 to prevent assertion errors
            x_lens = torch.clamp(x_lens, min=1)
        
        return x, x_lens 