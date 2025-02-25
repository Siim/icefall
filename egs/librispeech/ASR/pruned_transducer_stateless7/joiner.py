# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class Joiner(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        super().__init__()

        self.encoder_proj = nn.Linear(encoder_dim, joiner_dim)
        self.decoder_proj = nn.Linear(decoder_dim, joiner_dim)
        self.output_linear = nn.Linear(joiner_dim, vocab_size)

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is typically:
            - (N, T, C) for 3D inputs (from XLSR encoder)
            - (N, T, s_range, C) for 4D inputs 
            - (N, C) for 2D inputs
          decoder_out:
            Output from the decoder. Should have matching dimensions:
            - (N, T, C) for 3D inputs
            - (N, T, s_range, C) for 4D inputs
            - (N, C) for 2D inputs
          project_input:
            If true, apply input projections encoder_proj and decoder_proj.
            If this is false, it is the user's responsibility to do this
            manually.
        Returns:
          Return a tensor matching the input dimensions (N, T, [s_range], C).
        """
        # Handle dimension matching for different encoder types (XLSR vs Zipformer)
        assert encoder_out.ndim in (2, 3, 4), f"Encoder output has unsupported dimension: {encoder_out.ndim}"
        assert decoder_out.ndim in (2, 3, 4), f"Decoder output has unsupported dimension: {decoder_out.ndim}"
        
        # When dimensions don't match, we need to reshape appropriately
        if encoder_out.ndim != decoder_out.ndim:
            # Case: encoder_out is 3D (N,T,C) and decoder_out is 2D (N,C)
            if encoder_out.ndim == 3 and decoder_out.ndim == 2:
                # Expand decoder_out to match encoder time dimension
                decoder_out = decoder_out.unsqueeze(1).expand(-1, encoder_out.size(1), -1)
            # Case: encoder_out is 2D (N,C) and decoder_out is 3D (N,T,C)
            elif encoder_out.ndim == 2 and decoder_out.ndim == 3:
                # Expand encoder_out to match decoder time dimension
                encoder_out = encoder_out.unsqueeze(1).expand(-1, decoder_out.size(1), -1)
            # Handle 4D cases by reshaping to 3D when necessary
            elif encoder_out.ndim == 4 and decoder_out.ndim == 3:
                # Reshape 4D encoder output to 3D by merging T and s_range dimensions
                batch_size, T, s_range, channels = encoder_out.size()
                encoder_out = encoder_out.reshape(batch_size, T * s_range, channels)
            elif encoder_out.ndim == 3 and decoder_out.ndim == 4:
                # Add missing s_range dimension to encoder output
                encoder_out = encoder_out.unsqueeze(2)
        
        # Debug logging for dimensions
        # import logging
        # logging.debug(f"Joiner shapes: encoder={encoder_out.shape}, decoder={decoder_out.shape}")
        
        if project_input:
            logit = self.encoder_proj(encoder_out) + self.decoder_proj(decoder_out)
        else:
            logit = encoder_out + decoder_out

        logit = self.output_linear(torch.tanh(logit))

        return logit
