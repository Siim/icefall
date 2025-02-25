#!/usr/bin/env python3
# Copyright 2024
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

import heapq
import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from icefall.utils import DecodingResults


@dataclass
class Hypothesis:
    """Hypothesis class for beam search."""
    # Sequence of token ids
    sequence: List[int]
    # Log probability of the sequence
    score: float
    # Decoder hidden states
    decoder_out: Optional[torch.Tensor] = None
    # Timestamps for token emission
    timestamps: Optional[List[int]] = None
    # All scores for analysis
    all_scores: Optional[List[float]] = None

    def __lt__(self, other):
        """Compare hypotheses based on score for heap operations."""
        # Higher score is better (hence the negative)
        return -self.score < -other.score


class XLSRBeamSearch:
    """Beam search implementation for XLSR-Transducer as described in the paper."""
    def __init__(
        self,
        model: nn.Module,
        beam_size: int = 4,  # Paper's recommended beam width
        blank_penalty: float = 0.0,
        temperature: float = 1.0,
        return_timestamps: bool = False,
    ):
        """Initialize beam search.
        
        Args:
            model: The transducer model
            beam_size: Width of beam for search
            blank_penalty: Penalty applied to blank token (higher reduces insertions)
            temperature: Softmax temperature for logits
            return_timestamps: Whether to return timestamps along with hypotheses
        """
        self.model = model
        self.beam_size = beam_size
        self.blank_penalty = blank_penalty
        self.temperature = temperature
        self.return_timestamps = return_timestamps
        
        self.blank_id = model.decoder.blank_id
        self.context_size = model.decoder.context_size
        self.unk_id = getattr(model, "unk_id", self.blank_id)
        
    def _create_initial_hypothesis(self, device: torch.device) -> Hypothesis:
        """Create the initial hypothesis with blank token."""
        # Initial context with blank token
        initial_sequence = [-1] * (self.context_size - 1) + [self.blank_id]
        
        # Get the decoder output for this sequence
        decoder_input = torch.tensor(
            [initial_sequence],
            device=device,
            dtype=torch.int64,
        )
        
        decoder_out = self.model.decoder(decoder_input, need_pad=False)
        decoder_out = self.model.joiner.decoder_proj(decoder_out)
        
        return Hypothesis(
            sequence=initial_sequence,
            score=0.0,  # Start with log probability of 0 (probability of 1)
            decoder_out=decoder_out,
            timestamps=[],
            all_scores=[]
        )
    
    def _extend_hypothesis(
        self,
        hyp: Hypothesis,
        encoder_frame: torch.Tensor,
        frame_idx: int,
        device: torch.device
    ) -> List[Hypothesis]:
        """Extend a hypothesis with the next frame.
        
        Args:
            hyp: Current hypothesis
            encoder_frame: Current encoder frame output (1, 1, encoder_dim)
            frame_idx: Current frame index
            device: Device to perform computations on
            
        Returns:
            List of extended hypotheses
        """
        # Get the decoder output
        decoder_out = hyp.decoder_out
        
        # Compute joiner output (logits)
        encoder_proj = encoder_frame.unsqueeze(1)  # (1, 1, 1, encoder_dim)
        logits = self.model.joiner(
            encoder_proj,
            decoder_out.unsqueeze(1),
            project_input=False
        )
        logits = logits.squeeze(1).squeeze(1)  # (1, vocab_size)
        
        # Apply blank penalty if needed
        if self.blank_penalty != 0:
            logits[0, self.blank_id] -= self.blank_penalty
        
        # Apply temperature to logits
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get top-k tokens and their log probs
        top_k_log_probs, top_k_indices = log_probs.topk(self.beam_size)
        
        # Create new hypotheses
        new_hyps = []
        
        for i in range(self.beam_size):
            token_id = top_k_indices[0, i].item()
            token_log_prob = top_k_log_probs[0, i].item()
            
            # Create a copy of the current hypothesis sequence
            new_sequence = hyp.sequence.copy()
            new_timestamps = hyp.timestamps.copy() if hyp.timestamps is not None else []
            new_scores = hyp.all_scores.copy() if hyp.all_scores is not None else []
            
            # Only add non-blank tokens to the sequence
            if token_id not in (self.blank_id, self.unk_id):
                new_sequence.append(token_id)
                new_timestamps.append(frame_idx)
                new_scores.append(token_log_prob)
                
                # Get decoder output for new sequence
                decoder_input = torch.tensor(
                    [new_sequence[-self.context_size:] if len(new_sequence) >= self.context_size 
                     else ([-1] * (self.context_size - len(new_sequence)) + new_sequence)],
                    device=device,
                    dtype=torch.int64,
                )
                new_decoder_out = self.model.decoder(decoder_input, need_pad=False)
                new_decoder_out = self.model.joiner.decoder_proj(new_decoder_out)
            else:
                # For blank token, keep the same decoder output
                new_decoder_out = decoder_out
            
            # Create new hypothesis
            new_hyp = Hypothesis(
                sequence=new_sequence,
                score=hyp.score + token_log_prob,  # Accumulate log probability
                decoder_out=new_decoder_out,
                timestamps=new_timestamps,
                all_scores=new_scores
            )
            new_hyps.append(new_hyp)
        
        return new_hyps
    
    def _post_process_hyps(self, hyps: List[Hypothesis]) -> List[List[int]]:
        """Post-process hypotheses to return final token sequences."""
        results = []
        for hyp in hyps:
            # Strip initial context padding and blank tokens
            seq = [token for token in hyp.sequence if token not in (-1, self.blank_id)]
            results.append(seq)
        return results

    def _post_process_with_timestamps(self, hyps: List[Hypothesis]) -> DecodingResults:
        """Post-process hypotheses with timestamps."""
        hyp_tokens = []
        timestamps = []
        
        for hyp in hyps:
            # Strip initial context padding and blank tokens
            seq = [token for token in hyp.sequence if token not in (-1, self.blank_id)]
            
            # Get corresponding timestamps
            ts = hyp.timestamps
            
            hyp_tokens.append(seq)
            timestamps.append(ts)
        
        return DecodingResults(
            hyps=hyp_tokens,
            timestamps=timestamps,
        )
    
    def search_batch(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> Union[List[List[int]], DecodingResults]:
        """Perform beam search decoding on a batch of encoder outputs.
        
        Args:
            encoder_out: Output from the encoder (N, T, C)
            encoder_out_lens: Lengths of encoder outputs (N,)
            
        Returns:
            List of decoded token sequences or DecodingResults object
        """
        assert encoder_out.ndim == 3, "encoder_out must be 3D tensor"
        assert encoder_out.size(0) >= 1, "Batch size must be at least 1"
        
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        
        # Pack padded sequence for efficient processing
        packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
            input=encoder_out,
            lengths=encoder_out_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        
        # Project encoder output
        encoder_out = self.model.joiner.encoder_proj(packed_encoder_out.data)
        
        # Initialize beam for each item in batch
        hypotheses = [
            [self._create_initial_hypothesis(device)]
            for _ in range(batch_size)
        ]
        
        # Process frames
        offset = 0
        batch_size_list = packed_encoder_out.batch_sizes.tolist()
        
        for t, current_batch_size in enumerate(batch_size_list):
            start = offset
            end = offset + current_batch_size
            current_encoder_out = encoder_out[start:end]
            current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
            offset = end
            
            # Process each sample in the current batch
            for i in range(current_batch_size):
                # Get encoder output for current frame of current sample
                sample_encoder_out = current_encoder_out[i:i+1]
                
                # Get current hypotheses for this sample
                current_hyps = hypotheses[i]
                new_hyps = []
                
                # Extend each hypothesis
                for hyp in current_hyps:
                    extended_hyps = self._extend_hypothesis(
                        hyp, sample_encoder_out, t, device
                    )
                    new_hyps.extend(extended_hyps)
                
                # Keep only the best beam_size hypotheses
                hypotheses[i] = heapq.nlargest(self.beam_size, new_hyps)
        
        # Get best hypothesis for each sample in batch
        best_hyps = [hyps[0] for hyps in hypotheses]
        
        # Return either token sequences or DecodingResults with timestamps
        if self.return_timestamps:
            return self._post_process_with_timestamps(best_hyps)
        else:
            return self._post_process_hyps(best_hyps)


def beam_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam_size: int = 4,  # Paper's recommended beam width
    blank_penalty: float = 0,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Beam search in batch mode as described in the XLSR-Transducer paper.
    
    Args:
      model: The transducer model.
      encoder_out: Output from the encoder. Its shape is (N, T, C), where N >= 1.
      encoder_out_lens: A 1-D tensor of shape (N,), containing number of valid frames in
        encoder_out before padding.
      beam_size: Beam width for search (paper recommends 4)
      blank_penalty: Penalty applied to blank token
      return_timestamps: Whether to return timestamps.
      
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    searcher = XLSRBeamSearch(
        model=model,
        beam_size=beam_size,
        blank_penalty=blank_penalty,
        return_timestamps=return_timestamps,
    )
    
    return searcher.search_batch(encoder_out, encoder_out_lens) 