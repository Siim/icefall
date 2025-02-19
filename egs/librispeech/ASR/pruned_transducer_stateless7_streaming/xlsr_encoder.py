import torch
import torch.nn as nn
from encoder_interface import EncoderInterface
import time
import logging

# We use transformers library to load the XLSR model
# Make sure to install transformers: pip install transformers

class XLSREncoder(EncoderInterface):
    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-xls-r-300m", 
        decode_chunk_size: int = 8000,  # Default to 8000 samples (0.5s at 16kHz)
        chunk_overlap: int = None,  # Will be set to decode_chunk_size // 2
        adaptive_chunk_size: bool = True,
        min_chunk_size: int = 4000,
        max_chunk_size: int = 12000,
        latency_tolerance: float = 0.1  # 100ms tolerance
    ) -> None:
        super().__init__()
        from transformers import Wav2Vec2Model, Wav2Vec2Config
        config = Wav2Vec2Config.from_pretrained(model_name)
        # Disable masking for inference
        config.mask_time_prob = 0.0
        config.mask_time_length = 1
        config.mask_feature_prob = 0.0
        config.mask_feature_length = 1
        self.model = Wav2Vec2Model.from_pretrained(model_name, config=config)
        # The downsample factor is 320 for wav2vec2/XLSR models
        self.downsample_factor = 320
        
        # Set chunk sizes
        self.decode_chunk_size = decode_chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else decode_chunk_size // 2
        self.adaptive_chunk_size = adaptive_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.latency_tolerance = latency_tolerance
        
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

    def adjust_chunk_size(self, latency: float):
        """Adaptively adjust chunk size based on processing latency"""
        if not self.adaptive_chunk_size:
            return

        target_latency = self.latency_tolerance
        latency_ratio = latency / target_latency

        if latency_ratio > 1.1:  # Latency too high
            # Decrease chunk size
            self.current_chunk_size = max(
                self.min_chunk_size,
                int(self.current_chunk_size * 0.8)
            )
        elif latency_ratio < 0.9:  # Latency has room to grow
            # Increase chunk size
            self.current_chunk_size = min(
                self.max_chunk_size,
                int(self.current_chunk_size * 1.2)
            )

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
        if x.ndim == 3:
            x = x.squeeze(-1)  # Remove last dim if (batch, time, 1)
        assert x.ndim == 2, f"Expected 2D input (batch, time), got shape {x.shape}"
        
        # Clamp values silently since we know inputs are already normalized
        x = torch.clamp(x, min=-1.0, max=1.0)
        
        # Calculate output sequence lengths
        output_lengths = ((x_lens.float() / self.downsample_factor).floor() - 1).to(torch.int64)
        # Ensure output lengths are at least 1
        output_lengths = torch.maximum(output_lengths, torch.ones_like(output_lengths))
        
        if self.decode_chunk_size is None:
            # During inference, explicitly disable masking
            outputs = self.model(
                x,
                attention_mask=None,
                mask_time_indices=None,
                output_hidden_states=False,
                output_attentions=False,
                return_dict=False
            )[0]  # Get only the last hidden states
            
            # Ensure outputs don't exceed calculated lengths
            max_len = output_lengths.max().item()
            if outputs.size(1) > max_len:
                outputs = outputs[:, :max_len, :]
        else:
            outputs_list = []
            # Process each sample separately in chunks
            for i in range(x.size(0)):
                sample = x[i:i+1]  # (1, time)
                sample_len = int(x_lens[i].item())
                sample_outputs = []
                current = 0
                first_chunk = True

                while current < sample_len:
                    chunk_start_time = time.time()
                    
                    # Use adaptive chunk size if enabled
                    chunk_size = self.current_chunk_size if self.adaptive_chunk_size else self.decode_chunk_size
                    
                    if first_chunk or self.chunk_overlap == 0:
                        start_idx = current
                    else:
                        start_idx = max(0, current - self.chunk_overlap)
                    
                    end_idx = min(current + chunk_size, sample_len)
                    chunk = sample[:, start_idx:end_idx]
                    
                    # Ensure minimum chunk size to avoid masking issues
                    if chunk.size(1) < 20:  # minimum size to avoid masking issues
                        if end_idx >= sample_len:  # last chunk
                            # Pad if needed
                            pad_size = 20 - chunk.size(1)
                            chunk = torch.nn.functional.pad(chunk, (0, pad_size))
                        else:
                            # Skip this small chunk and process with next iteration
                            continue
                    
                    # Process chunk - explicitly disable masking during inference
                    chunk_out = self.model(
                        chunk,
                        attention_mask=None,
                        mask_time_indices=None,
                        output_hidden_states=False,
                        output_attentions=False,
                        return_dict=False
                    )[0]  # Get only the last hidden states
                    
                    # If not the first chunk and overlap is used, drop the overlapping frames
                    if not first_chunk and self.chunk_overlap > 0:
                        overlap_frames = self.chunk_overlap // self.downsample_factor
                        chunk_out = chunk_out[:, overlap_frames:, :]
                    
                    sample_outputs.append(chunk_out)
                    first_chunk = False
                    current += chunk_size - (self.chunk_overlap if not first_chunk else 0)
                    
                    # Measure and adjust chunk size based on latency
                    chunk_latency = time.time() - chunk_start_time
                    self.last_chunk_latency = chunk_latency
                    self.adjust_chunk_size(chunk_latency)
                
                sample_out = torch.cat(sample_outputs, dim=1)  # concatenate along time
                # Trim to match calculated output length
                sample_out = sample_out[:, :output_lengths[i], :]
                outputs_list.append(sample_out)
            
            # Pad sequences to the same length
            max_len = max(out.size(1) for out in outputs_list)
            padded_outputs = []
            for out in outputs_list:
                if out.size(1) < max_len:
                    padding = torch.zeros((1, max_len - out.size(1), out.size(2)), 
                                       dtype=out.dtype, device=out.device)
                    out = torch.cat([out, padding], dim=1)
                padded_outputs.append(out)
            outputs = torch.cat(padded_outputs, dim=0)
        
        return outputs, output_lengths 