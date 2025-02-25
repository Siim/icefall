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
        blank_penalty: float = 5.0,  # Increased blank penalty to prevent repetitions
        temperature: float = 1.4,  # Higher temperature for more diversity
        return_timestamps: bool = False,
        max_symb_per_frame: int = 3,  # Max symbols emitted per frame
        max_output_length: int = 200,  # Maximum allowed output length
    ):
        """Initialize beam search.
        
        Args:
            model: The transducer model
            beam_size: Width of beam for search
            blank_penalty: Penalty applied to blank token (higher reduces insertions)
            temperature: Softmax temperature for logits
            return_timestamps: Whether to return timestamps along with hypotheses
            max_symb_per_frame: Maximum symbols emitted per frame (prevents loops)
            max_output_length: Maximum allowed output length to prevent runaway generation
        """
        self.model = model
        self.beam_size = beam_size
        self.blank_penalty = blank_penalty
        self.temperature = temperature
        self.return_timestamps = return_timestamps
        self.max_symb_per_frame = max_symb_per_frame
        self.max_output_length = max_output_length
        
        self.blank_id = model.decoder.blank_id
        self.context_size = model.decoder.context_size
        self.unk_id = getattr(model, "unk_id", self.blank_id)
        
        # Keep track of number of symbols per frame to prevent loops
        self.symb_per_frame = {}
        
        # Set of detected patterns to avoid
        self.pattern_cache = set()
        
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
        # Input validation for better debugging
        if encoder_frame is None:
            logging.error("encoder_frame is None in _extend_hypothesis")
            return [hyp]  # Return current hypothesis without extension
        
        # Check for extremely long sequences - emergency stop to prevent runaway generation
        non_blank_tokens = [t for t in hyp.sequence if t not in (-1, self.blank_id)]
        if len(non_blank_tokens) > self.max_output_length:
            logging.warning(f"Hypothesis exceeded max length ({self.max_output_length}), stopping generation")
            return [hyp]  # Return current hypothesis without extension
            
        # Check for repetitive patterns in the sequence
        if len(hyp.sequence) >= 12:  # Need enough context to detect patterns
            # Look for repeating 2, 3, 4, or 6-grams
            for n in [2, 3, 4, 6]:
                if len(hyp.sequence) >= n * 3:  # Need at least 3 repetitions to detect
                    # Get the last n tokens
                    last_ngram = tuple(hyp.sequence[-n:])
                    # Check if we have a repeating pattern
                    if (hyp.sequence[-2*n:-n] == hyp.sequence[-n:] and 
                        hyp.sequence[-3*n:-2*n] == hyp.sequence[-n:]):
                        logging.warning(f"Detected {n}-gram repetition: {last_ngram}")
                        # Emergency stop - severe repetition detected
                        return [hyp]
            
        if not isinstance(hyp.decoder_out, torch.Tensor):
            logging.error(f"Invalid decoder_out type: {type(hyp.decoder_out)}")
            try:
                # Attempt to recreate decoder output
                decoder_input = torch.tensor(
                    [hyp.sequence[-self.context_size:] if len(hyp.sequence) >= self.context_size 
                     else ([-1] * (self.context_size - len(hyp.sequence)) + hyp.sequence)],
                    device=device,
                    dtype=torch.int64,
                )
                decoder_out = self.model.decoder(decoder_input, need_pad=False)
                hyp.decoder_out = self.model.joiner.decoder_proj(decoder_out)
            except Exception as e:
                logging.error(f"Failed to recover decoder_out: {str(e)}")
                return [hyp]  # Return current hypothesis without extension
        
        try:
            # Normalize encoder_frame dimensions based on its shape
            if encoder_frame.ndim == 1:  # (D,)
                encoder_frame = encoder_frame.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            elif encoder_frame.ndim == 2:  # (1, D) or (T, D)
                if encoder_frame.size(0) == 1:
                    encoder_frame = encoder_frame.unsqueeze(0)  # (1, 1, D)
                else:
                    encoder_frame = encoder_frame[0:1].unsqueeze(0)  # (1, 1, D)
            elif encoder_frame.ndim == 3:  # (B, T, D) or (1, 1, D)
                if encoder_frame.size(0) > 1 or encoder_frame.size(1) > 1:
                    encoder_frame = encoder_frame[0:1, 0:1]  # Take just first batch and time
            
            # Ensure decoder_out has the right dimensions (1, 1, D)
            decoder_out = hyp.decoder_out
            if decoder_out.ndim == 2:  # (1, D)
                decoder_out = decoder_out.unsqueeze(1)  # (1, 1, D)
            
            # Compute joiner output (logits)
            logits = self.model.joiner(
                encoder_frame,
                decoder_out,
                project_input=False
            )
            
            # After joiner, get to (1, vocab_size)
            logits = logits.squeeze(0).squeeze(0)  # (vocab_size,)
            
            # Apply blank penalty and temperature adjustments
            if self.blank_penalty != 0:
                logits[self.blank_id] -= self.blank_penalty
                
                # Add high penalty for repeated blank tokens
                if len(hyp.sequence) >= 3 and hyp.sequence[-1] == self.blank_id and hyp.sequence[-2] == self.blank_id:
                    logits[self.blank_id] -= 20.0  # Higher penalty for repeated blanks
            
            if self.temperature != 1.0:
                logits = logits / self.temperature
                
            # Stronger penalties for repetitive patterns
            if len(hyp.sequence) >= 6:
                # Check for simple repetition (same token repeated)
                last_token = hyp.sequence[-1]
                if last_token != self.blank_id and last_token == hyp.sequence[-3] and last_token == hyp.sequence[-5]:
                    # Penalize this token heavily to break the loop
                    logits[last_token] -= 30.0  # Increased penalty
                
                # Check for alternating pattern (A B A B A B...)
                if (hyp.sequence[-1] == hyp.sequence[-3] and 
                    hyp.sequence[-2] == hyp.sequence[-4] and
                    hyp.sequence[-3] == hyp.sequence[-5]):
                    # Penalize both tokens heavily
                    token1 = hyp.sequence[-1]
                    token2 = hyp.sequence[-2]
                    if token1 != self.blank_id:
                        logits[token1] -= 30.0  # Increased penalty
                    if token2 != self.blank_id:
                        logits[token2] -= 30.0  # Increased penalty
                
                # Penalize 3-grams that repeat
                if len(hyp.sequence) >= 9:
                    if (hyp.sequence[-3:] == hyp.sequence[-6:-3]):
                        # Get the 3-gram tokens and penalize them
                        for token in set(hyp.sequence[-3:]):
                            if token != self.blank_id and token != -1:
                                logits[token] -= 25.0
            
            # Apply softmax to convert to probabilities
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Top-k tokens from log probabilities
            k = min(self.beam_size, log_probs.size(-1))
            top_k_log_probs, top_k_indices = log_probs.topk(k)
            
            # Create new hypotheses
            new_hyps = []
            
            # Track if we've seen a non-blank token
            seen_non_blank = False
            
            # Limit number of non-blank tokens per frame to prevent infinite looping
            non_blank_count = 0
            max_non_blank = self.max_symb_per_frame
            
            for i in range(k):
                token_id = top_k_indices[i].item()
                token_log_prob = top_k_log_probs[i].item()
                
                # For blank tokens, always add to beam
                if token_id == self.blank_id:
                    # Create a copy of the current sequence (should not change for blank)
                    new_sequence = hyp.sequence.copy()
                    
                    # Create copies of timestamp and score arrays
                    new_timestamps = hyp.timestamps.copy() if hyp.timestamps is not None else []
                    new_scores = hyp.all_scores.copy() if hyp.all_scores is not None else []
                    
                    # Keep the same decoder output for blank tokens
                    new_decoder_out = decoder_out
                    
                    # Create the new hypothesis
                    new_hyp = Hypothesis(
                        sequence=new_sequence,
                        score=hyp.score + token_log_prob,
                        decoder_out=new_decoder_out,
                        timestamps=new_timestamps,
                        all_scores=new_scores
                    )
                    new_hyps.append(new_hyp)
                else:
                    # Token is non-blank - enforce max non-blank tokens per frame
                    if non_blank_count >= max_non_blank:
                        continue
                    
                    # Update counter
                    non_blank_count += 1
                    seen_non_blank = True
                    
                    # Create a copy of the current sequence
                    new_sequence = hyp.sequence.copy()
                    
                    # Create copies of timestamp and score arrays
                    new_timestamps = hyp.timestamps.copy() if hyp.timestamps is not None else []
                    new_scores = hyp.all_scores.copy() if hyp.all_scores is not None else []
                    
                    # Token is non-blank - update the sequence
                    new_sequence.append(token_id)
                    new_timestamps.append(frame_idx)
                    new_scores.append(token_log_prob)
                                    
                    # Get decoder output for new sequence
                    try:
                        context = new_sequence[-self.context_size:] if len(new_sequence) >= self.context_size else ([-1] * (self.context_size - len(new_sequence)) + new_sequence)
                        decoder_input = torch.tensor([context], device=device, dtype=torch.int64)
                        new_decoder_out = self.model.decoder(decoder_input, need_pad=False)
                        new_decoder_out = self.model.joiner.decoder_proj(new_decoder_out)
                    except Exception as e:
                        logging.error(f"Failed to get decoder output: {str(e)}")
                        # Use old decoder output if we fail
                        new_decoder_out = decoder_out
                    
                    # Create the new hypothesis
                    new_hyp = Hypothesis(
                        sequence=new_sequence,
                        score=hyp.score + token_log_prob,
                        decoder_out=new_decoder_out,
                        timestamps=new_timestamps,
                        all_scores=new_scores
                    )
                    new_hyps.append(new_hyp)
            
            # If all our candidates were blank tokens, force consider the best non-blank
            if not seen_non_blank and len(new_hyps) > 0:
                # Find the highest scoring non-blank token
                for i in range(k, min(log_probs.size(-1), k*2)):
                    token_id = log_probs.argsort(descending=True)[i].item()
                    if token_id != self.blank_id:
                        token_log_prob = log_probs[token_id].item()
                        
                        # Create new hypothesis with this token
                        new_sequence = hyp.sequence.copy()
                        new_timestamps = hyp.timestamps.copy() if hyp.timestamps is not None else []
                        new_scores = hyp.all_scores.copy() if hyp.all_scores is not None else []
                        
                        new_sequence.append(token_id)
                        new_timestamps.append(frame_idx)
                        new_scores.append(token_log_prob)
                        
                        try:
                            context = new_sequence[-self.context_size:] if len(new_sequence) >= self.context_size else ([-1] * (self.context_size - len(new_sequence)) + new_sequence)
                            decoder_input = torch.tensor([context], device=device, dtype=torch.int64)
                            new_decoder_out = self.model.decoder(decoder_input, need_pad=False)
                            new_decoder_out = self.model.joiner.decoder_proj(new_decoder_out)
                        except Exception as e:
                            new_decoder_out = decoder_out
                        
                        forced_hyp = Hypothesis(
                            sequence=new_sequence,
                            score=hyp.score + token_log_prob,
                            decoder_out=new_decoder_out,
                            timestamps=new_timestamps,
                            all_scores=new_scores
                        )
                        new_hyps.append(forced_hyp)
                        logging.debug(f"Forcing consideration of non-blank token {token_id}")
                        break
            
            return new_hyps
            
        except Exception as e:
            logging.error(f"Error in _extend_hypothesis: {str(e)}")
            logging.error("Exception details:", exc_info=True)
            return [hyp]  # Return current hypothesis if extension failed
    
    def _post_process_hyps(self, hyps: List[Hypothesis]) -> List[List[int]]:
        """Post-process hypotheses to return final token sequences."""
        results = []
        for hyp in hyps:
            # Strip initial context padding and blank tokens
            seq = []
            for token in hyp.sequence:
                if token not in (-1, self.blank_id):
                    seq.append(token)
            
            # Limit sequence length
            if len(seq) > 100:
                seq = seq[:100]
                
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
        """Perform beam search decoding on a batch of encoder outputs."""
        device = encoder_out.device
        
        # Validate input dimensions
        if encoder_out.ndim not in (2, 3, 4):
            logging.error(f"Unsupported encoder_out dimension: {encoder_out.ndim}")
            # Try to reshape to supported dimension
            if encoder_out.ndim == 1:
                encoder_out = encoder_out.unsqueeze(0).unsqueeze(0)  # Add batch and time dims
            else:
                # For higher dims, try to get to (batch, time, features)
                shape = encoder_out.shape
                encoder_out = encoder_out.reshape(1, -1, shape[-1])
            logging.warning(f"Reshaped encoder_out to {encoder_out.shape}")
            
        # Convert to the right shape - we need (batch, time, features)
        if encoder_out.ndim == 2:
            # Handle (batch, features) by adding time dimension
            if encoder_out.size(0) == 1:
                encoder_out = encoder_out.unsqueeze(1)  # (1, 1, features)
            else:
                # Multiple items but no time dim - unclear how to interpret
                # Assume it's (time, features) for a single item
                encoder_out = encoder_out.unsqueeze(0)  # (1, time, features)
        elif encoder_out.ndim == 4:
            # Handle (batch, time, s_range, features) by reshaping
            b, t, s, f = encoder_out.size()
            encoder_out = encoder_out.reshape(b, t * s, f)
        
        # Get batch size after reshaping
        batch_size = encoder_out.size(0)
        
        # Ensure encoder_out_lens matches batch size
        if encoder_out_lens.size(0) != batch_size:
            logging.warning(f"encoder_out_lens batch size ({encoder_out_lens.size(0)}) doesn't match encoder_out ({batch_size})")
            
            if encoder_out_lens.size(0) > batch_size:
                # Too many lengths, truncate
                encoder_out_lens = encoder_out_lens[:batch_size]
            else:
                # Not enough lengths, pad with max length
                max_len = encoder_out.size(1)
                padding = torch.full(
                    (batch_size - encoder_out_lens.size(0),), 
                    max_len, 
                    device=device, 
                    dtype=encoder_out_lens.dtype
                )
                encoder_out_lens = torch.cat([encoder_out_lens, padding])
        
        # Cap lengths to actual encoder output size
        max_seq_len = encoder_out.size(1)
        encoder_out_lens = torch.minimum(encoder_out_lens, torch.tensor(max_seq_len, device=device))
        
        # Ensure no zero-length sequences
        encoder_out_lens = torch.maximum(encoder_out_lens, torch.ones_like(encoder_out_lens))
        
        # Initialize initial hypotheses for each batch item
        beam_list = []
        for b in range(batch_size):
            beam_list.append(self._create_initial_hypothesis(device))
        
        # Initialize result container
        hypotheses = []
        
        # Process each batch item separately
        for b in range(batch_size):
            logging.debug(f"Processing batch item {b+1}/{batch_size}")
            current_len = encoder_out_lens[b].item()
            current_encoder_out = encoder_out[b, :current_len]
            
            # Initialize beam with single hypothesis
            current_beam = [beam_list[b]]
            
            # Set maximum consecutive non-token frames to prevent infinite loops
            consecutive_blanks = 0
            max_consecutive_blanks = min(current_len // 4, 10)  # Limit based on sequence length
            
            # Process each frame
            for t in range(current_len):
                # Get current encoder frame
                current_frame = current_encoder_out[t:t+1]  # (1, encoder_dim)
                
                # Extend each hypothesis in the beam
                new_beam = []
                non_blank_extensions = False
                
                for hyp in current_beam:
                    # Extend the hypothesis
                    extended_hyps = self._extend_hypothesis(hyp, current_frame, t, device)
                    
                    # Check if we got any non-blank extensions
                    for extended_hyp in extended_hyps:
                        if len(extended_hyp.sequence) > len(hyp.sequence):
                            non_blank_extensions = True
                            consecutive_blanks = 0
                            break
                    
                    # Add all extensions to the new beam
                    new_beam.extend(extended_hyps)
                
                # If we didn't get any non-blank extensions, increment counter
                if not non_blank_extensions:
                    consecutive_blanks += 1
                    
                # Break early if stuck in a loop of blanks
                if consecutive_blanks >= max_consecutive_blanks:
                    logging.warning(f"Breaking out of blank token loop at frame {t}/{current_len}")
                    break
                
                # Keep only top beam_size hypotheses
                if len(new_beam) > self.beam_size:
                    # Two scoring methods:
                    # 1. Raw score (sum of log probs)
                    # 2. Normalized score (average log prob per token)
                    
                    # For early frames (first 25%), use raw score to encourage longer sequences
                    if t < current_len * 0.25:
                        sorted_beam = sorted(new_beam, key=lambda hyp: hyp.score, reverse=True)
                    else:
                        # For later frames, use length-normalized score
                        sorted_beam = sorted(
                            new_beam,
                            key=lambda hyp: hyp.score / max(1, len([t for t in hyp.sequence if t != -1])),
                            reverse=True
                        )
                    
                    # Take top beam_size
                    current_beam = sorted_beam[:self.beam_size]
                else:
                    current_beam = new_beam
                
                # Occasionally log progress
                if t % max(1, current_len // 10) == 0:
                    logging.debug(f"Batch {b+1}, Frame {t+1}/{current_len}, beam size: {len(current_beam)}")
                    if current_beam:
                        top_hyp = current_beam[0]
                        num_tokens = len([t for t in top_hyp.sequence if t not in (-1, self.blank_id)])
                        logging.debug(f"Top hyp has {num_tokens} non-blank tokens, score: {top_hyp.score:.2f}")
            
            # Choose the best hypothesis for this batch item
            if current_beam:
                # Use length-normalized score for final selection
                best_hyp = max(
                    current_beam,
                    key=lambda hyp: hyp.score / max(1, len([t for t in hyp.sequence if t != -1]))
                )
                hypotheses.append(best_hyp)
            else:
                # Fallback to initial hypothesis
                hypotheses.append(beam_list[b])
        
        # Return appropriate format based on return_timestamps flag
        if self.return_timestamps:
            return self._post_process_with_timestamps(hypotheses)
        else:
            return self._post_process_hyps(hypotheses)


def beam_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam_size: int = 4,  # Paper's recommended beam width
    blank_penalty: float = 5.0,  # Increased blank penalty
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
        max_output_length=100,  # Limit output length as a safety measure
    )
    
    return searcher.search_batch(encoder_out, encoder_out_lens) 