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
from typing import Tuple

import k2
import torch
import torch.nn as nn
from torch import amp
from encoder_interface import EncoderInterface
from xlsr_encoder import XLSREncoder
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

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.joiner_dim = joiner_dim
        self.vocab_size = vocab_size

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
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
        Returns:
          Return the transducer loss.

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        # x.T_dim == max(x_len)
        assert x.size(1) == x_lens.max().item(), (x.shape, x_lens, x_lens.max())

        encoder_out, x_lens = self.encoder(x, x_lens)
        
        # Project XLSR output if needed
        encoder_out = self.encoder_proj(encoder_out)
        
        assert torch.all(x_lens > 0)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros((x.size(0), 4), dtype=torch.int64, device=x.device)
        boundary[:, 2] = y_lens
        boundary[:, 3] = x_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with amp.autocast('cuda', enabled=False):
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

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with amp.autocast('cuda', enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return (simple_loss, pruned_loss)

    def simple_loss(self, logits: torch.Tensor, y: k2.RaggedTensor) -> torch.Tensor:
        """
        Compute simple loss (un-pruned) following paper's approach.
        Args:
            logits: Output from joiner, shape (B, T, U, vocab_size)
            y: Target labels as k2.RaggedTensor
        Returns:
            A scalar tensor containing the simple loss
        """
        # Convert ragged tensor to dense padded tensor
        y_padded = y.pad(mode="constant", padding_value=0)
        
        # Get batch size and max sequence lengths
        B, T, U, V = logits.shape
        
        # Create target tensor with blank padding
        target = torch.full(
            (B, T, U),
            fill_value=0,  # 0 is blank_id
            device=logits.device,
            dtype=torch.long,
        )
        
        # Fill in the target values
        for b in range(B):
            cur_len = y.shape[b][0]
            target[b, :, :cur_len] = y_padded[b, :cur_len]
        
        # Compute loss using cross entropy
        logits = logits.reshape(-1, V)  # (B*T*U, V)
        target = target.reshape(-1)  # (B*T*U)
        
        loss = torch.nn.functional.cross_entropy(
            logits,
            target,
            reduction="sum",
            ignore_index=0,  # Ignore blank_id
        )
        
        return loss

    def pruned_loss(
        self,
        logits: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute pruned loss following paper's approach.
        Args:
            logits: Output from joiner, shape (B, T, U, vocab_size)
            y: Target labels as k2.RaggedTensor
            prune_range: Range for pruning
            am_scale: Scale for acoustic model scores
            lm_scale: Scale for language model scores
        Returns:
            A scalar tensor containing the pruned loss
        """
        B, T, U, V = logits.shape
        
        # Convert ragged tensor to dense padded tensor
        y_padded = y.pad(mode="constant", padding_value=0)
        
        # Create pruning mask
        mask = torch.zeros((B, T, U), dtype=torch.bool, device=logits.device)
        
        # Fill pruning mask based on prune_range
        for b in range(B):
            cur_len = y.shape[b][0]
            for t in range(T):
                start = max(0, t - prune_range)
                end = min(cur_len, t + prune_range)
                mask[b, t, start:end] = True
        
        # Apply mask to logits
        logits = logits.masked_select(mask.unsqueeze(-1)).reshape(-1, V)
        
        # Create target tensor
        target = []
        for b in range(B):
            cur_len = y.shape[b][0]
            for t in range(T):
                start = max(0, t - prune_range)
                end = min(cur_len, t + prune_range)
                target.extend(y_padded[b, start:end].tolist())
        
        target = torch.tensor(target, device=logits.device)
        
        # Apply scaling if provided
        if am_scale != 0.0:
            logits = logits * am_scale
        if lm_scale != 0.0:
            # Apply language model scaling
            lm_scores = torch.log_softmax(logits, dim=-1)
            logits = logits + lm_scale * lm_scores
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits,
            target,
            reduction="sum",
            ignore_index=0,  # Ignore blank_id
        )
        
        return loss
