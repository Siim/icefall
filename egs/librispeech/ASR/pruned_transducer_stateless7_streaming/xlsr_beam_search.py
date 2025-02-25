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
import logging

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
            encoder_frame: Current encoder frame output (shape may vary)
            frame_idx: Current frame index
            device: Device to perform computations on
            
        Returns:
            List of extended hypotheses
        """
        # Validate inputs to help with debugging
        if encoder_frame is None:
            logging.error("encoder_frame is None in _extend_hypothesis")
            # Return current hypothesis if we can't extend it
            return [hyp]
        
        if not isinstance(hyp.decoder_out, torch.Tensor):
            logging.error(f"Invalid decoder_out type: {type(hyp.decoder_out)}")
            # Try to recover by creating a new decoder output
            try:
                decoder_input = torch.tensor(
                    [hyp.sequence[-self.context_size:] if len(hyp.sequence) >= self.context_size 
                     else ([-1] * (self.context_size - len(hyp.sequence)) + hyp.sequence)],
                    device=device,
                    dtype=torch.int64,
                )
                decoder_out = self.model.decoder(decoder_input, need_pad=False)
                decoder_out = self.model.joiner.decoder_proj(decoder_out)
                hyp.decoder_out = decoder_out
            except Exception as e:
                logging.error(f"Failed to recover decoder_out: {str(e)}")
                # Return empty list if we can't recover
                return []
        
        # Get the decoder output
        decoder_out = hyp.decoder_out
        
        # Reshape encoder frame to appropriate dimensions
        # We need to ensure it has shape compatible with the joiner
        try:
            # Normalize encoder frame dimensions
            if encoder_frame.ndim == 1:  # (features,)
                encoder_proj = encoder_frame.unsqueeze(0).unsqueeze(0)  # (1, 1, features)
            elif encoder_frame.ndim == 2:  # (1, features) or (time, features)
                if encoder_frame.size(0) == 1:
                    encoder_proj = encoder_frame.unsqueeze(0)  # (1, 1, features)
                else:
                    # Take just the first time frame
                    encoder_proj = encoder_frame[0:1].unsqueeze(0)  # (1, 1, features)
            elif encoder_frame.ndim == 3:  # (1, 1, features) or (batch, time, features)
                if encoder_frame.size(0) == 1 and encoder_frame.size(1) == 1:
                    encoder_proj = encoder_frame  # Already (1, 1, features)
                else:
                    # Take just the first batch and time frame
                    encoder_proj = encoder_frame[0:1, 0:1]  # (1, 1, features)
            elif encoder_frame.ndim == 4:  # (1, 1, 1, features)
                encoder_proj = encoder_frame.squeeze(2)  # (1, 1, features)
            else:
                # Unexpected dimension - try to reshape to (1, 1, -1)
                logging.warning(f"Unexpected encoder_frame dimension: {encoder_frame.shape}")
                encoder_proj = encoder_frame.reshape(1, 1, -1)
            
            # Ensure decoder has right dimensions - we need (1, 1, features)
            if decoder_out.ndim == 2:  # (1, features)
                decoder_proj = decoder_out.unsqueeze(1)  # (1, 1, features)
            elif decoder_out.ndim == 3:  # Already (1, 1, features)
                decoder_proj = decoder_out
            else:
                # Unexpected dimension - try to reshape
                logging.warning(f"Unexpected decoder_out dimension: {decoder_out.shape}")
                decoder_proj = decoder_out.reshape(1, 1, -1)
            
            # Compute joiner output (logits)
            logits = self.model.joiner(
                encoder_proj,
                decoder_proj,
                project_input=False
            )
            
            # After joiner, get to (1, vocab_size)
            if logits.ndim == 3:  # (1, 1, vocab_size)
                logits = logits.squeeze(1)
            elif logits.ndim == 4:  # (1, 1, 1, vocab_size)
                logits = logits.squeeze(1).squeeze(1)
            
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
                    try:
                        decoder_input = torch.tensor(
                            [new_sequence[-self.context_size:] if len(new_sequence) >= self.context_size 
                             else ([-1] * (self.context_size - len(new_sequence)) + new_sequence)],
                            device=device,
                            dtype=torch.int64,
                        )
                        new_decoder_out = self.model.decoder(decoder_input, need_pad=False)
                        new_decoder_out = self.model.joiner.decoder_proj(new_decoder_out)
                    except Exception as e:
                        logging.error(f"Failed to get decoder output: {str(e)}")
                        new_decoder_out = decoder_out  # Fall back to current decoder output
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
            
        except Exception as e:
            logging.error(f"Error in _extend_hypothesis: {str(e)}")
            # Return current hypothesis if extension failed
            return [hyp]
    
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
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        
        # Validate input dimensions
        if encoder_out.ndim not in (2, 3, 4):
            raise ValueError(f"Unsupported encoder_out dimension: {encoder_out.ndim}")
        
        # Handle different encoder dimensions - we need (batch, time, features)
        if encoder_out.ndim == 2:
            # Handle (batch, features) by adding time dimension
            encoder_out = encoder_out.unsqueeze(1)
        elif encoder_out.ndim == 4:
            # Handle (batch, time, s_range, features) by reshaping
            b, t, s, f = encoder_out.size()
            encoder_out = encoder_out.reshape(b, t * s, f)
        
        # Ensure encoder_out_lens has correct batch size
        if encoder_out_lens.size(0) != batch_size:
            logging.warning(f"encoder_out_lens batch size ({encoder_out_lens.size(0)}) doesn't match encoder_out ({batch_size})")
            if encoder_out_lens.size(0) > batch_size:
                encoder_out_lens = encoder_out_lens[:batch_size]
            else:
                # If not enough length values, pad with max length
                max_len = encoder_out.size(1)
                padding = torch.full(
                    (batch_size - encoder_out_lens.size(0),), 
                    max_len, 
                    device=device, 
                    dtype=encoder_out_lens.dtype
                )
                encoder_out_lens = torch.cat([encoder_out_lens, padding])
        
        # Cap sequence lengths to actual encoder output size
        encoder_out_lens = torch.minimum(encoder_out_lens, torch.tensor(encoder_out.size(1), device=device))
        
        # For safety, double-check no zero-length sequences
        encoder_out_lens = torch.maximum(encoder_out_lens, torch.ones_like(encoder_out_lens))
        
        # Initialize beam for each item in batch
        beam_list = [self._create_initial_hypothesis(device) for _ in range(batch_size)]
        
        # Process batches with packed sequence to handle variable lengths efficiently
        try:
            # Try to use pack_padded_sequence for efficient processing
            sorted_lens, indices = torch.sort(encoder_out_lens, descending=True)
            reorder_indices = torch.argsort(indices)
            sorted_encoder_out = encoder_out[indices]
            
            # Pack the encoder outputs
            packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_encoder_out,
                sorted_lens.cpu(),
                batch_first=True
            )
            
            # Project packed encoder output
            packed_projected = self.model.joiner.encoder_proj(packed_encoder_out.data)
            
            # Process each batch separately
            hypotheses = [None] * batch_size
            for b in range(batch_size):
                # Get the index in the sorted order
                idx = indices[b].item()
                current_len = encoder_out_lens[idx].item()
                current_encoder_out = encoder_out[idx, :current_len]
                
                # Project encoder output for this batch item
                current_proj = self.model.joiner.encoder_proj(current_encoder_out)
                
                current_beam = [beam_list[idx]]
                
                # Process each frame
                for t in range(current_len):
                    # Get encoder output for current frame
                    frame = current_proj[t:t+1]  # Shape: (1, encoder_dim)
                    
                    # Create new hypotheses by extending each existing one
                    new_beam = []
                    for hyp in current_beam:
                        new_hyps = self._extend_hypothesis(hyp, frame, t, device)
                        new_beam.extend(new_hyps)
                    
                    # Keep only top beam_size hypotheses
                    if len(new_beam) > self.beam_size:
                        current_beam = heapq.nlargest(self.beam_size, new_beam)
                    else:
                        current_beam = new_beam
                
                # Store best hypothesis for this batch
                hypotheses[indices[b].item()] = current_beam[0] if current_beam else beam_list[idx]
                
            # Order the results back to original batch order
            ordered_hypotheses = [hypotheses[i.item()] for i in reorder_indices]
            
        except Exception as e:
            # Fallback to simpler batch processing without packing
            logging.warning(f"Packed sequence processing failed, using fallback: {str(e)}")
            
            # Create a separate beam for each item in the batch
            hypotheses = []
            
            # Process each item in the batch separately
            for b in range(batch_size):
                current_len = encoder_out_lens[b].item()
                current_encoder_out = encoder_out[b, :current_len]
                
                # Project the encoder output
                current_proj = self.model.joiner.encoder_proj(current_encoder_out)
                
                current_beam = [beam_list[b]]
                
                # Process each frame
                for t in range(current_len):
                    frame = current_proj[t:t+1]  # Shape: (1, encoder_dim)
                    new_beam = []
                    
                    for hyp in current_beam:
                        new_hyps = self._extend_hypothesis(hyp, frame, t, device)
                        new_beam.extend(new_hyps)
                    
                    # Keep only top beam_size hypotheses
                    current_beam = heapq.nlargest(min(self.beam_size, len(new_beam)), new_beam)
                
                # Get the best hypothesis for this batch item
                hypotheses.append(current_beam[0] if current_beam else beam_list[b])
        
        # Return either token sequences or DecodingResults with timestamps
        if self.return_timestamps:
            return self._post_process_with_timestamps(hypotheses)
        else:
            return self._post_process_hyps(hypotheses)


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