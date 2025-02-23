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
            encoder_out: Output from the encoder, of shape
                (batch_size, T, encoder_dim) or (batch_size, T, 1, encoder_dim)
            decoder_out: Output from the decoder, of shape
                (batch_size, U, decoder_dim) or (batch_size, T, U, decoder_dim)
            project_input: If True, apply input projections encoder_proj and decoder_proj.
                If False, assume inputs are already projected.
        Returns:
            torch.Tensor: of shape (batch_size, T, U, vocab_size)
        """
        if encoder_out.ndim == 3:
            # Add U dimension: (B, T, C) -> (B, T, 1, C)
            encoder_out = encoder_out.unsqueeze(2)
        
        if decoder_out.ndim == 3:
            # Add T dimension: (B, U, C) -> (B, 1, U, C)
            decoder_out = decoder_out.unsqueeze(1)
            # Expand T dimension: (B, 1, U, C) -> (B, T, U, C)
            encoder_T = encoder_out.size(1)
            decoder_out = decoder_out.expand(-1, encoder_T, -1, -1)
        
        assert encoder_out.ndim == 4
        assert decoder_out.ndim == 4
        
        # Now both are 4D: (batch_size, T, U, dim)
        if project_input:
            encoder_out = self.encoder_proj(encoder_out)
            decoder_out = self.decoder_proj(decoder_out)
        
        # Add outputs together then apply final projection
        logits = encoder_out + decoder_out
        logits = self.output_linear(logits)
        
        return logits
