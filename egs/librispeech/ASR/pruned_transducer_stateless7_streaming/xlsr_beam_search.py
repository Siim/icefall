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
        blank_penalty: float = 10.0,  # Even higher blank penalty to prevent repetitions
        temperature: float = 1.4,  # Higher temperature for more diversity
        return_timestamps: bool = False,
        max_symb_per_frame: int = 2,  # Reduced to prevent excessive tokens per frame
        max_output_length: int = 50,  # Shorter maximum output for early training
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
        
        # Track number of tokens emitted per frame
        self.tokens_per_frame = {}
        
        # Force early termination after specific number of repetitions
        self.max_pattern_repetitions = 2
        
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
    
    def _check_for_repetitions(self, sequence: List[int]) -> bool:
        """Check for repetitive patterns in sequence.
        
        Returns:
            True if repetitions detected, False otherwise
        """
        # Not enough tokens to check for repetitions
        if len(sequence) < 8:
            return False
        
        # Check for exact token repetitions
        seq = [t for t in sequence if t not in (-1, self.blank_id)]
        if len(seq) < 6:
            return False
        
        # Get just the actual tokens (no padding or blanks)
        for n in [1, 2, 3, 4]:
            # Need at least 3 repetitions to consider it a pattern
            if len(seq) >= n * 3:
                # Look for exact token repetitions
                for i in range(len(seq) - 2*n):
                    if seq[i:i+n] == seq[i+n:i+2*n] == seq[i+2*n:i+3*n]:
                        return True
                    
                # Check recent 2/3 of sequence for repetitions
                recent = seq[-int(len(seq)*0.67):]
                if len(recent) >= n * 3:
                    for i in range(len(recent) - 2*n):
                        if recent[i:i+n] == recent[i+n:i+2*n]:
                            return True
        
        # Check for token frequency anomalies
        if len(seq) >= 10:
            # Count frequency of each token
            counter = {}
            for token in seq:
                counter[token] = counter.get(token, 0) + 1
            
            # If any token appears more than 40% of the time, it's suspicious
            for token, count in counter.items():
                if count > 0.4 * len(seq) and len(seq) > 10:
                    return True
        
        return False
    
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
        # Initialize frame token counter if needed
        if frame_idx not in self.tokens_per_frame:
            self.tokens_per_frame[frame_idx] = 0
            
        # Check for token limit per frame
        if self.tokens_per_frame[frame_idx] >= self.max_symb_per_frame:
            # Return only a blank extension
            try:
                # Create blank hypothesis 
                new_hyp = Hypothesis(
                    sequence=hyp.sequence.copy(),
                    score=hyp.score,
                    decoder_out=hyp.decoder_out,
                    timestamps=hyp.timestamps.copy() if hyp.timestamps is not None else [],
                    all_scores=hyp.all_scores.copy() if hyp.all_scores is not None else []
                )
                return [new_hyp]
            except Exception:
                return [hyp]
                
        # Input validation 
        if encoder_frame is None:
            logging.error("encoder_frame is None in _extend_hypothesis")
            return [hyp]  # Return current hypothesis without extension
        
        # Check for very early in training - use only blank tokens for stability
        non_blank_tokens = [t for t in hyp.sequence if t not in (-1, self.blank_id)]
        very_early_training = frame_idx < 5 and not non_blank_tokens
        
        # Early termination for extremely long sequences
        if len(non_blank_tokens) > self.max_output_length:
            logging.warning(f"Hypothesis exceeded max length ({self.max_output_length}), stopping generation")
            return [hyp]  # Return current hypothesis without extension
            
        # Early termination for repetitive patterns
        if self._check_for_repetitions(hyp.sequence):
            logging.warning(f"Detected repetition pattern at frame {frame_idx}, stopping generation")
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
                # No blank penalty for very early in training
                if not very_early_training:
                    logits[self.blank_id] -= self.blank_penalty
                
                # Add extreme penalty for repeated blank tokens
                if len(hyp.sequence) >= 3 and hyp.sequence[-1] == self.blank_id and hyp.sequence[-2] == self.blank_id:
                    logits[self.blank_id] -= 50.0  # Much higher penalty
            
            if self.temperature != 1.0:
                logits = logits / self.temperature
                
            # Even stronger penalties for repetitive patterns
            if len(hyp.sequence) >= 6:
                # Check for simple repetition (same token repeated)
                last_token = hyp.sequence[-1]
                if last_token != self.blank_id and last_token == hyp.sequence[-3] and last_token == hyp.sequence[-5]:
                    # Extreme penalty - effectively remove this token
                    logits[last_token] = float('-inf')
                
                # Check for alternating pattern (A B A B A B...)
                if (hyp.sequence[-1] == hyp.sequence[-3] and 
                    hyp.sequence[-2] == hyp.sequence[-4]):
                    # Extreme penalty for both tokens
                    token1 = hyp.sequence[-1]
                    token2 = hyp.sequence[-2]
                    if token1 != self.blank_id:
                        logits[token1] = float('-inf')
                    if token2 != self.blank_id:
                        logits[token2] = float('-inf')
                
                # Check for 3-grams that repeat
                if len(hyp.sequence) >= 9:
                    if (hyp.sequence[-3:] == hyp.sequence[-6:-3]):
                        # Block all tokens in the 3-gram
                        for token in set(hyp.sequence[-3:]):
                            if token != self.blank_id and token != -1:
                                logits[token] = float('-inf')
            
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
            
            # For very early training, only allow blank tokens
            if very_early_training:
                # Only consider blank tokens
                blank_log_prob = log_probs[self.blank_id].item()
                new_hyp = Hypothesis(
                    sequence=hyp.sequence.copy(),
                    score=hyp.score + blank_log_prob,
                    decoder_out=decoder_out,
                    timestamps=hyp.timestamps.copy() if hyp.timestamps is not None else [],
                    all_scores=hyp.all_scores.copy() if hyp.all_scores is not None else []
                )
                return [new_hyp]
            
            # For normal processing, consider all tokens
            for i in range(k):
                token_id = top_k_indices[i].item()
                token_log_prob = top_k_log_probs[i].item()
                
                # Skip tokens with very low probability 
                if token_log_prob < -15.0 and token_id != self.blank_id:
                    continue
                    
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
                    
                    # Update frame token counter
                    self.tokens_per_frame[frame_idx] += 1
                    
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
            # But skip this for early frames to maintain stability
            if not seen_non_blank and len(new_hyps) > 0 and frame_idx > 10:
                # Find the highest scoring non-blank token
                for i in range(k, min(log_probs.size(-1), k*2)):
                    token_id = log_probs.argsort(descending=True)[i].item()
                    if token_id != self.blank_id:
                        token_log_prob = log_probs[token_id].item()
                        
                        # Skip tokens with very low probability
                        if token_log_prob < -10.0:
                            continue
                        
                        # Create new hypothesis with this token
                        new_sequence = hyp.sequence.copy()
                        new_timestamps = hyp.timestamps.copy() if hyp.timestamps is not None else []
                        new_scores = hyp.all_scores.copy() if hyp.all_scores is not None else []
                        
                        new_sequence.append(token_id)
                        new_timestamps.append(frame_idx)
                        new_scores.append(token_log_prob)
                        
                        # Update frame token counter
                        self.tokens_per_frame[frame_idx] += 1
                        
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
            
    def _fallback_greedy_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        device: torch.device
    ) -> List[List[int]]:
        """Fallback to simple greedy search if beam search fails."""
        logging.warning("Falling back to simple greedy search due to beam search issues")
        
        batch_size = encoder_out.size(0)
        results = []
        
        for b in range(batch_size):
            # Get single sequence
            seq_len = encoder_out_lens[b].item()
            current_encoder_out = encoder_out[b, :seq_len]
            
            # Initialize with blank
            hyp = self._create_initial_hypothesis(device)
            
            # Process frames
            for t in range(seq_len):
                # Get frame
                current_frame = current_encoder_out[t:t+1]
                
                # Extend with greedy search (just take best token)
                try:
                    # Normalize dimensions
                    if current_frame.ndim == 1:
                        current_frame = current_frame.unsqueeze(0).unsqueeze(0)
                        
                    # Get decoder output
                    decoder_out = hyp.decoder_out
                    if decoder_out.ndim == 2:
                        decoder_out = decoder_out.unsqueeze(1)
                    
                    # Get logits
                    logits = self.model.joiner(
                        current_frame,
                        decoder_out,
                        project_input=False
                    ).squeeze(0).squeeze(0)
                    
                    # Apply blank bias to favor blanks in early training
                    logits[self.blank_id] += 3.0
                    
                    # Get best token
                    token_id = logits.argmax().item()
                    
                    # If non-blank, add to hypothesis
                    if token_id != self.blank_id:
                        # Add token to sequence
                        hyp.sequence.append(token_id)
                        
                        # Update decoder output for next step
                        context = hyp.sequence[-self.context_size:] if len(hyp.sequence) >= self.context_size else ([-1] * (self.context_size - len(hyp.sequence)) + hyp.sequence)
                        decoder_input = torch.tensor([context], device=device, dtype=torch.int64)
                        hyp.decoder_out = self.model.joiner.decoder_proj(
                            self.model.decoder(decoder_input, need_pad=False)
                        )
                        
                    # Early stopping if sequence is too long
                    non_blanks = [t for t in hyp.sequence if t not in (-1, self.blank_id)]
                    if len(non_blanks) > self.max_output_length:
                        break
                except Exception as e:
                    logging.error(f"Error in greedy fallback: {str(e)}")
                    break
            
            # Post-process result
            seq = [token for token in hyp.sequence if token not in (-1, self.blank_id)]
            results.append(seq)
        
        return results
    
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
            non_blank_indices = [i for i, token in enumerate(hyp.sequence) 
                               if token not in (-1, self.blank_id)]
            
            # Get tokens and corresponding timestamps
            seq = [hyp.sequence[i] for i in non_blank_indices]
            ts = [hyp.timestamps[i] for i in range(len(hyp.timestamps)) 
                 if i < len(non_blank_indices)]
            
            # Limit sequence length
            if len(seq) > 100:
                seq = seq[:100]
                ts = ts[:100] if ts else []
                
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
            # Reset tokens per frame for this sequence
            self.tokens_per_frame = {}
            
            logging.debug(f"Processing batch item {b+1}/{batch_size}")
            current_len = encoder_out_lens[b].item()
            current_encoder_out = encoder_out[b, :current_len]
            
            # Skip empty or invalid sequences
            if current_len <= 0:
                hypotheses.append(beam_list[b])
                continue
                
            # Initialize beam with single hypothesis
            current_beam = [beam_list[b]]
            
            # Set maximum consecutive non-token frames to prevent infinite loops
            consecutive_blanks = 0
            max_consecutive_blanks = min(current_len // 6, 8)  # More restrictive
            
            # Force early stopping if we see repeated patterns
            pattern_repetitions = 0
            had_repetition = False
            
            # Process each frame
            failed = False
            for t in range(current_len):
                # Get current encoder frame
                current_frame = current_encoder_out[t:t+1]  # (1, encoder_dim)
                
                # Extend each hypothesis in the beam
                new_beam = []
                non_blank_extensions = False
                
                for hyp in current_beam:
                    # Detect and skip stuck hypotheses
                    if len(hyp.sequence) > 10:
                        # Count blanks at end of sequence
                        trailing_blanks = 0
                        for i in range(len(hyp.sequence) - 1, -1, -1):
                            if hyp.sequence[i] == self.blank_id:
                                trailing_blanks += 1
                            else:
                                break
                                
                        # Skip hypotheses with too many trailing blanks
                        if trailing_blanks > 5 and t > 10:
                            continue
                    
                    # Check for repetitions
                    if self._check_for_repetitions(hyp.sequence):
                        had_repetition = True
                        pattern_repetitions += 1
                        if pattern_repetitions >= self.max_pattern_repetitions:
                            # Switch to greedy search for this hypothesis
                            logging.warning(f"Excessive repetitions detected at frame {t}, breaking beam search")
                            failed = True
                            break
                        continue
                    
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
                
                # Break if we failed
                if failed:
                    break
                    
                # If we didn't get any new hypotheses, try the next frame
                if not new_beam:
                    consecutive_blanks += 1
                    new_beam = [current_beam[0]] if current_beam else [beam_list[b]]
                # If we didn't get any non-blank extensions, increment counter
                elif not non_blank_extensions:
                    consecutive_blanks += 1
                    
                # Break early if stuck in a loop of blanks
                if consecutive_blanks >= max_consecutive_blanks:
                    logging.warning(f"Breaking out of blank token loop at frame {t}/{current_len}")
                    break
                
                # Keep only top beam_size hypotheses
                if len(new_beam) > self.beam_size:
                    # Only use length-normalized score - more stable
                    sorted_beam = sorted(
                        new_beam,
                        key=lambda hyp: hyp.score / max(1, sum(1 for t in hyp.sequence if t not in (-1, self.blank_id))),
                        reverse=True
                    )
                    
                    # Take top beam_size
                    current_beam = sorted_beam[:self.beam_size]
                else:
                    current_beam = new_beam
                
                # Log occasionally
                if t % max(1, current_len // 5) == 0:
                    if current_beam:
                        top_hyp = current_beam[0]
                        num_tokens = sum(1 for t in top_hyp.sequence if t not in (-1, self.blank_id))
                        logging.debug(f"Frame {t+1}/{current_len}, tokens: {num_tokens}")
            
            # Fallback to greedy search if beam search failed
            if failed or had_repetition and pattern_repetitions >= self.max_pattern_repetitions:
                result = self._fallback_greedy_search(
                    encoder_out[b:b+1], 
                    encoder_out_lens[b:b+1],
                    device
                )
                hypotheses.append(Hypothesis(
                    sequence=[-1] * (self.context_size - 1) + [self.blank_id] + result[0],
                    score=0.0,
                    decoder_out=None,
                    timestamps=[],
                    all_scores=[]
                ))
                continue
            
            # Choose the best hypothesis for this batch item
            if current_beam:
                # Use length-normalized score for final selection
                best_hyp = max(
                    current_beam,
                    key=lambda hyp: hyp.score / max(1, sum(1 for t in hyp.sequence if t not in (-1, self.blank_id)))
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
    blank_penalty: float = 10.0,  # Higher blank penalty for stability
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
        max_output_length=50,  # More aggressive length limit for early training
    )
    
    return searcher.search_batch(encoder_out, encoder_out_lens) 