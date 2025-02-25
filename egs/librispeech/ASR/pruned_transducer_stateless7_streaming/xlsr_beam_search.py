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

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from icefall.utils import DecodingResults


@dataclass
class Hypothesis:
    """Hypothesis class for beam search."""
    # Sequence of token ids (not including initial blanks or padding)
    tokens: List[int]
    # Log probability of the sequence
    score: float
    # Decoder hidden states for continuing the hypothesis
    decoder_out: Optional[torch.Tensor] = None
    # Timestamps in frames for token emission (for return_timestamps=True)
    timestamps: Optional[List[int]] = None


class XLSRTransducerBeamSearch:
    """Simple beam search implementation for XLSR-Transducer following the paper.
    
    From the paper: "We perform beam search with width 4 for all the decoding experiments."
    """
    def __init__(
        self,
        model: nn.Module,
        beam_size: int = 4,  # Paper specifies beam width 4
        blank_id: Optional[int] = None,
        blank_bias: float = 0.0,  # Bias for blank token (positive = favor blank)
        max_sym_per_step: int = 1,  # Max symbols per step from the paper
        max_states: int = 32,  # Maximum number of states to maintain
    ):
        """Initialize beam search.
        
        Args:
            model: The transducer model
            beam_size: Width of beam for search (paper uses 4)
            blank_id: ID of blank token (if None, use model.decoder.blank_id)
            blank_bias: Bias added to blank token logits (positive favors blanks)
            max_sym_per_step: Maximum symbols emitted per step
            max_states: Maximum number of states to maintain before pruning (memory control)
        """
        self.model = model
        self.beam_size = beam_size
        self.blank_id = blank_id if blank_id is not None else model.decoder.blank_id
        self.blank_bias = blank_bias
        self.max_sym_per_step = max_sym_per_step
        self.max_states = max_states
        self.context_size = getattr(model.decoder, "context_size", 1)
        
        # Set up logger for diagnostics
        self.logger = logging.getLogger("XLSRTransducerBeamSearch")
        
    def _get_initial_tokens(self, device) -> List[int]:
        """Get initial token sequence for decoder."""
        # Handle different context sizes for different decoders
        if self.context_size > 1:
            # For context_size > 1, we need padding + blank
            return [-1] * (self.context_size - 1) + [self.blank_id]
        else:
            # For context_size = 1, just the blank
            return [self.blank_id]
            
    def _get_decoder_output(self, tokens: List[int], device: torch.device) -> torch.Tensor:
        """Get decoder output for a token sequence."""
        # Ensure we have at least context_size tokens
        if len(tokens) < self.context_size:
            # Pad with -1 for missing context
            context = [-1] * (self.context_size - len(tokens)) + tokens
        else:
            # Take last context_size tokens
            context = tokens[-self.context_size:]
            
        # Create input tensor
        decoder_input = torch.tensor([context], device=device, dtype=torch.int64)
        
        # Get decoder output
        try:
            decoder_out = self.model.decoder(decoder_input, need_pad=False)
            # We don't apply decoder_proj here anymore - we'll do it in the beam search
            return decoder_out
        except Exception as e:
            self.logger.warning(f"Error computing decoder output: {e}")
            return None

    def beam_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        return_timestamps: bool = False,
        early_stopping: bool = True,
    ) -> Union[List[List[int]], DecodingResults]:
        """Perform beam search on a batch of encoder outputs.
        
        Args:
            encoder_out: Output from the encoder, shape (batch, time, feature_dim)
            encoder_out_lens: Length of each sequence, shape (batch)
            return_timestamps: Whether to return timestamps with hypotheses
            early_stopping: Whether to enable early stopping for efficiency
            
        Returns:
            Either List of token sequences or DecodingResults with timestamps
        """
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        
        # Ensure encoder_out is 3D (batch, time, feature_dim)
        if encoder_out.ndim == 2:
            # Single feature vector, add time dimension
            if encoder_out.size(0) == 1:
                encoder_out = encoder_out.unsqueeze(1)  # (1, 1, feature_dim)
            else:
                # Assume (time, feature_dim) for single item
                encoder_out = encoder_out.unsqueeze(0)  # (1, time, feature_dim)
        elif encoder_out.ndim == 4:
            # Handle (batch, time, s_range, feature_dim) from pruned RNN-T loss
            b, t, s, f = encoder_out.size()
            encoder_out = encoder_out.reshape(b, t * s, f)
            
        assert encoder_out.ndim == 3, f"Expected 3D encoder output, got shape {encoder_out.shape}"
            
        # Ensure encoder_out_lens matches batch size
        if encoder_out_lens.size(0) != batch_size:
            self.logger.warning(
                f"encoder_out_lens batch size {encoder_out_lens.size(0)} doesn't match encoder_out {batch_size}"
            )
            encoder_out_lens = encoder_out_lens[:batch_size] if encoder_out_lens.size(0) > batch_size else \
                               torch.cat([encoder_out_lens, 
                                         torch.full((batch_size - encoder_out_lens.size(0),), 
                                                   encoder_out.size(1), 
                                                   device=device,
                                                   dtype=encoder_out_lens.dtype)])
                                         
        # Clamp lengths to actual encoder output size
        encoder_out_lens = torch.clamp(encoder_out_lens, min=1, max=encoder_out.size(1))
        
        all_results = []
        
        # Process each item in batch separately
        for b in range(batch_size):
            length = encoder_out_lens[b].item()
            enc_out = encoder_out[b, :length]  # (time, feature_dim)
            
            # Initialize beam with single empty hypothesis
            initial_tokens = self._get_initial_tokens(device)
            initial_decoder_out = self._get_decoder_output(initial_tokens, device)
            
            beam = [Hypothesis(
                tokens=initial_tokens,
                score=0.0,
                decoder_out=initial_decoder_out,
                timestamps=[] if return_timestamps else None
            )]
            
            # Keep track of finished hypotheses
            finished_beam = []
            max_symbol_per_frame = self.max_sym_per_step
            
            # Process each encoder frame
            for t in range(length):
                # Limit symbols per frame to prevent explosion
                symbols_added_current_frame = 0
                
                # Get current encoder frame (1, feature_dim)
                frame = enc_out[t:t+1]
                
                # Create new beam for extending hypotheses
                new_beam = []
                
                # Process each hypothesis in the beam
                for hyp in beam:
                    # Get decoder output - already projected for joiner
                    if hyp.decoder_out is None:
                        hyp.decoder_out = self._get_decoder_output(hyp.tokens, device)
                        # Skip if we still can't get decoder output
                        if hyp.decoder_out is None:
                            continue
                    
                    decoder_out = hyp.decoder_out  # (1, decoder_dim)
                    
                    # Ensure correct dimensions for joiner
                    # Encoder frame: (1, feature_dim) -> (1, 1, feature_dim)
                    # Decoder out: (1, decoder_dim) -> (1, 1, decoder_dim)
                    frame_for_joiner = frame.unsqueeze(0)  # (1, 1, feature_dim)
                    decoder_out_for_joiner = decoder_out.unsqueeze(1)  # (1, 1, decoder_dim)
                    
                    # Apply projections before joining - THIS IS THE KEY FIX
                    # Project encoder output
                    projected_encoder = self.model.joiner.encoder_proj(frame_for_joiner)
                    # Project decoder output
                    projected_decoder = self.model.joiner.decoder_proj(decoder_out_for_joiner)
                    
                    # Compute joint output and log probabilities
                    try:
                        # Now use the projected outputs with project_input=False
                        logits = self.model.joiner(
                            projected_encoder, projected_decoder, project_input=False
                        ).squeeze(0).squeeze(0)  # (vocab_size,)
                        
                        # Apply blank bias if specified
                        if self.blank_bias != 0:
                            logits[self.blank_id] += self.blank_bias
                            
                        log_probs = torch.log_softmax(logits, dim=-1)
                        
                    except Exception as e:
                        self.logger.warning(f"Error computing joint: {e}")
                        continue  # Skip this hypothesis
                    
                    # Get topk tokens and their probabilities
                    k = min(self.beam_size, log_probs.size(0))
                    topk_logp, topk_indices = log_probs.topk(k)
                    
                    # Process each token
                    for i in range(k):
                        token = topk_indices[i].item()
                        token_logp = topk_logp[i].item()
                        
                        # For blank token: advance time step without new token
                        if token == self.blank_id:
                            # Create new hypothesis with same tokens but updated score
                            new_hyp = Hypothesis(
                                tokens=hyp.tokens.copy(),
                                score=hyp.score + token_logp,
                                decoder_out=hyp.decoder_out,  # Decoder state unchanged for blank
                                timestamps=hyp.timestamps.copy() if return_timestamps else None
                            )
                            new_beam.append(new_hyp)
                        
                        # For non-blank token: emit a new token if we haven't hit the limit
                        elif symbols_added_current_frame < max_symbol_per_frame:
                            # Create new list of tokens with the new token added
                            new_tokens = hyp.tokens.copy()
                            new_tokens.append(token)
                            
                            # Update timestamps if needed
                            new_timestamps = None
                            if return_timestamps:
                                new_timestamps = hyp.timestamps.copy()
                                new_timestamps.append(t)
                            
                            # Get decoder output for next step
                            new_decoder_out = self._get_decoder_output(new_tokens, device)
                            
                            # Create new hypothesis
                            new_hyp = Hypothesis(
                                tokens=new_tokens,
                                score=hyp.score + token_logp,
                                decoder_out=new_decoder_out,
                                timestamps=new_timestamps
                            )
                            new_beam.append(new_hyp)
                            
                            # Track symbol addition limit
                            symbols_added_current_frame += 1
                
                # Keep top beam_size hypotheses for next step
                if len(new_beam) > self.beam_size:
                    # Use length-normalized scores for better beam diversity
                    # Add 1 to avoid divide-by-zero for empty hypotheses
                    beam = sorted(
                        new_beam,
                        key=lambda x: x.score / (len([t for t in x.tokens if t != self.blank_id]) + 1),
                        reverse=True
                    )[:self.beam_size]
                else:
                    beam = new_beam
                
                # Control beam states memory
                if len(beam) > self.max_states:
                    beam = beam[:self.max_states]
                
                # Early stopping check: if all hypotheses end with blank, consider stopping
                if early_stopping and all(h.tokens[-1] == self.blank_id for h in beam):
                    break
            
            # Finished processing encoder sequence
            # Add all remaining hypotheses to finished_beam
            finished_beam.extend(beam)
            
            # Get best hypothesis
            if finished_beam:
                # Use length-normalized scores for final selection
                best_hyp = max(
                    finished_beam,
                    key=lambda x: x.score / max(1, len([t for t in x.tokens if t not in (-1, self.blank_id)]))
                )
                
                # Extract tokens (removing initial context and blanks)
                result_tokens = []
                for token in best_hyp.tokens:
                    if token not in (-1, self.blank_id):
                        result_tokens.append(token)
                
                if return_timestamps:
                    all_results.append((result_tokens, best_hyp.timestamps))
                else:
                    all_results.append(result_tokens)
            else:
                # No hypotheses found - return empty result
                if return_timestamps:
                    all_results.append(([], []))
                else:
                    all_results.append([])
        
        # Format results based on return_timestamps flag
        if return_timestamps:
            token_seqs = [tokens for tokens, _ in all_results]
            timestamp_seqs = [timestamps for _, timestamps in all_results]
            return DecodingResults(hyps=token_seqs, timestamps=timestamp_seqs)
        else:
            return all_results


def beam_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam_size: int = 4,  # Paper's recommended beam width
    blank_penalty: float = 0.0,  # Penalty for blank token (negative = penalty, positive = bias)
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Perform beam search decoding on a batch of encoder outputs.
    
    Args:
      model: The transducer model.
      encoder_out: Output from the encoder, shape (batch, time, feature_dim).
      encoder_out_lens: Length of each sequence, shape (batch).
      beam_size: Beam width for search (paper recommends 4).
      blank_penalty: Penalty for blank token (negative = penalty, positive = bias).
      return_timestamps: Whether to return timestamps.
      
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object containing
      decoded result and corresponding timestamps.
    """
    # Determine if we're early in training and adjust accordingly
    training_phase = getattr(model, "training_phase", "late")
    epoch = getattr(model, "cur_epoch", 100)
    
    # For very early training, use small blank bias to help convergence
    early_blank_bias = 1.0 if epoch <= 3 else 0.0
    
    # Initialize beam search
    search = XLSRTransducerBeamSearch(
        model=model,
        beam_size=beam_size,
        blank_bias=early_blank_bias - blank_penalty,  # Combine early training bias with user penalty
        max_sym_per_step=1,  # Standard transducer allows 1 symbol per step
    )
    
    # Perform beam search
    return search.beam_search(
        encoder_out=encoder_out,
        encoder_out_lens=encoder_out_lens,
        return_timestamps=return_timestamps,
    ) 