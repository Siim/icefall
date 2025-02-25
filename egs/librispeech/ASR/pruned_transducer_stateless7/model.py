# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang, Wei Kang)
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


import random

import k2
import torch
import torch.nn as nn
from torch import amp
from xlsr_encoder import XLSREncoder, EncoderInterface
from scaling import penalize_abs_values_gt

from icefall.utils import add_sos


class Transducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder: EncoderInterface,
        decoder: nn.Module,
        joiner: nn.Module,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dm) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        assert isinstance(encoder, EncoderInterface), type(encoder)
        assert hasattr(decoder, "blank_id")

        self.encoder = encoder
        self.decoder = decoder
        self.joiner = joiner

        # Add projection layer for XLSR encoder if needed
        if isinstance(encoder, XLSREncoder):
            self.encoder_proj = nn.Linear(1024, encoder_dim)
        else:
            self.encoder_proj = nn.Identity()

        self.simple_am_proj = nn.Linear(
            encoder_dim,
            vocab_size,
        )
        self.simple_lm_proj = nn.Linear(decoder_dim, vocab_size)

        self.is_streaming = False  # Flag for token deduplication during inference

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        encoder_outputs_provided: bool = True,
    ) -> torch.Tensor:
        """
        Args:
          x:
            Either encoder outputs with shape (N, T, encoder_dim) if encoder_outputs_provided=True
            or raw inputs with shape (N, T, C) if encoder_outputs_provided=False
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          encoder_outputs_provided:
            If True, x is treated as encoder outputs and doesn't need further processing by the encoder.
            If False, x is treated as raw inputs that need to be processed by the encoder.
        Returns:
          Return the transducer loss.
        """
        # Input validation
        assert x.ndim == 3, f"Expected x to have 3 dimensions, got shape {x.shape}"
        assert x_lens.ndim == 1, f"Expected x_lens to have 1 dimension, got shape {x_lens.shape}"
        assert y.num_axes == 2, f"Expected y to have 2 axes, got {y.num_axes}"
        assert x.size(0) == x_lens.size(0) == y.dim0, f"Batch size mismatch: x={x.size(0)}, x_lens={x_lens.size(0)}, y={y.dim0}"

        # Get encoder output
        if encoder_outputs_provided:
            # x is already encoder output
            encoder_out = x
        else:
            # Process x through the encoder
            encoder_out, x_lens = self.encoder(x, x_lens)
            # Project XLSR output if needed
            encoder_out = self.encoder_proj(encoder_out)
        
        # Ensure x_lens matches actual encoder output size
        x_lens = torch.minimum(x_lens, torch.tensor(encoder_out.size(1), device=x_lens.device))
        
        assert torch.all(x_lens > 0), f"All x_lens must be positive, got {x_lens}"
        assert torch.all(x_lens <= encoder_out.size(1)), f"x_lens must not exceed encoder output size, got x_lens={x_lens}, output_size={encoder_out.size(1)}"

        # Get label lengths
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        # Decoder preparation
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # Get decoder output
        decoder_out = self.decoder(sos_y_padded)

        # Prepare labels
        y_padded = y.pad(mode="constant", padding_value=0)
        y_padded = y_padded.to(torch.int64)

        # Create and validate boundary tensor
        batch_size = x.size(0)
        boundary = torch.zeros((batch_size, 4), dtype=torch.int64, device=x.device)
        
        # First validate lengths
        assert torch.all(y_lens >= 0), f"Label lengths must be non-negative, got {y_lens}"
        assert torch.all(x_lens >= 0), f"Frame lengths must be non-negative, got {x_lens}"
        
        # Set boundary values
        boundary[:, 0] = 0  # Start frame index
        boundary[:, 1] = 0  # Start label index  
        boundary[:, 2] = y_lens  # End label index
        boundary[:, 3] = x_lens  # End frame index
        
        # Validate boundary conditions
        assert torch.all(boundary[:, 2] <= y_padded.size(1)), \
            f"Label length exceeds padded size: max_len={boundary[:, 2].max()}, padded_size={y_padded.size(1)}"
        assert torch.all(boundary[:, 3] <= encoder_out.size(1)), \
            f"Frame length exceeds encoder output size: max_len={boundary[:, 3].max()}, output_size={encoder_out.size(1)}"

        # Project to vocabulary space
        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # Compute losses
        with torch.amp.autocast('cuda', enabled=False):
            try:
                simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                    lm=lm.float(),
                    am=am.float(),
                    symbols=y_padded,
                    termination_symbol=blank_id,
                    lm_only_scale=lm_scale,
                    am_only_scale=am_scale,
                    boundary=boundary,
                    reduction="sum",
                    return_grad=True,
                )
            except Exception as e:
                print(f"Error in rnnt_loss_smoothed: {str(e)}")
                print(f"lm shape: {lm.shape}, am shape: {am.shape}")
                print(f"symbols shape: {y_padded.shape}, boundary: {boundary}")
                raise

        # Get pruning ranges
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # Prune and get final logits
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        # Compute pruned loss
        with torch.amp.autocast('cuda', enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return (simple_loss, pruned_loss)
