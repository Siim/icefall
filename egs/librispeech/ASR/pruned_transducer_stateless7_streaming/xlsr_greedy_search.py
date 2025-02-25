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

import torch
import torch.nn as nn
import logging
from typing import List, Optional, Tuple, Union, Dict, Set
from collections import Counter, deque
import math
from icefall.utils import DecodingResults

def greedy_search(
    model: nn.Module,
    encoder_out: torch.Tensor,
    max_sym_per_frame: int = 1,
    blank_penalty: float = 0.0,
    max_output_length: int = 100,
    repetition_penalty: float = 1.5,
    return_timestamps: bool = False,
) -> Union[List[int], DecodingResults]:
    """XLSR-specific greedy search for a single utterance.
    
    This implementation includes enhancements to handle repetition,
    enforce early stopping, and properly handle XLSR output dimensions.
    
    Args:
      model: The transducer model
      encoder_out: Encoder output, shape (1, T, C)
      max_sym_per_frame: Maximum symbols per frame
      blank_penalty: Penalty for blank token (negative = penalty, positive = favor blanks)
      max_output_length: Maximum output sequence length to prevent runaway sequences
      repetition_penalty: Penalty factor for repetitive tokens
      return_timestamps: Whether to return timestamps
      
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object with decoded result and timestamps.
    """
    assert encoder_out.ndim == 3, f"Expected 3D encoder output, got shape {encoder_out.shape}"
    assert encoder_out.size(0) == 1, f"Expected batch size 1, got {encoder_out.size(0)}"
    
    # Get model parameters
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size
    
    # Get device
    device = next(model.parameters()).device
    
    # Create initial decoder input
    decoder_input = torch.tensor(
        [-1] * (context_size - 1) + [blank_id], 
        device=device, 
        dtype=torch.int64
    ).reshape(1, context_size)
    
    # Project decoder output
    decoder_out = model.decoder(decoder_input, need_pad=False)
    decoder_out = model.joiner.decoder_proj(decoder_out)
    
    # Project encoder output for XLSR
    encoder_out = model.joiner.encoder_proj(encoder_out)
    
    # Get sequence length
    T = encoder_out.size(1)
    
    # Initialize hypothesis
    hyp = [blank_id] * context_size
    
    # Initialize counters and buffers
    t = 0  # Current frame index
    sym_per_frame = 0  # Symbols emitted in current frame
    sym_per_utt = 0  # Symbols emitted in total
    timestamp = []  # Timestamps for when tokens were emitted
    
    # Repetition detection
    n_gram_counters = {
        2: Counter(),  # Count 2-grams
        3: Counter(),  # Count 3-grams
        4: Counter(),  # Count 4-grams
    }
    recent_tokens = deque(maxlen=8)  # Keep track of recent tokens
    consecutive_blanks = 0  # Count consecutive blank tokens
    token_diversity = set()  # Track unique tokens
    
    # Process each frame
    while t < T and sym_per_utt < max_output_length:
        # Check if we've reached the limit for the current frame
        if sym_per_frame >= max_sym_per_frame:
            sym_per_frame = 0
            t += 1
            continue
        
        try:
            # Get current encoder output and prepare for joiner
            current_encoder_out = encoder_out[:, t:t+1, :].unsqueeze(2)
            
            # Join encoder and decoder outputs
            logits = model.joiner(
                current_encoder_out, 
                decoder_out.unsqueeze(1), 
                project_input=False
            ).squeeze(0).squeeze(0)  # Remove batch dims
            
            # Apply blank penalty
            if blank_penalty != 0:
                logits[blank_id] -= blank_penalty
            
            # Apply repetition penalties
            if len(recent_tokens) > 0:
                last_token = recent_tokens[-1]
                
                # Penalize recently seen tokens
                for token in recent_tokens:
                    if token != blank_id:
                        logits[token] /= repetition_penalty
                
                # Extra penalty for the immediately preceding token
                if last_token != blank_id:
                    logits[last_token] /= 1.5  # Additional penalty for immediate repetition
            
            # Check for severe repetition patterns
            severe_repetition = False
            if len(hyp) > context_size + 4:
                # Check for n-gram repetitions
                for n, counter in n_gram_counters.items():
                    if len(hyp) >= context_size + n:
                        # Create n-gram from the most recent tokens
                        ng = tuple(hyp[-(n):])
                        if ng in counter and counter[ng] > 3:  # If we've seen this n-gram more than 3 times
                            severe_repetition = True
                            break
                
                # Check token diversity ratio
                if len(token_diversity) > 0 and sym_per_utt > 20:
                    diversity_ratio = len(token_diversity) / sym_per_utt
                    if diversity_ratio < 0.2:  # Less than 20% unique tokens
                        severe_repetition = True
            
            # Hard-stop for severe repetition - favor blank token
            if severe_repetition:
                logits[:] = float('-inf')
                logits[blank_id] = 0.0
            
            # Get predicted token
            y = logits.argmax().item()
            
            # Update state based on predicted token
            if y == blank_id:
                consecutive_blanks += 1
                # Early stopping if too many consecutive blanks
                if consecutive_blanks > 20:
                    break
                # Move to next frame after blank
                sym_per_frame = 0
                t += 1
            else:
                # Non-blank token
                consecutive_blanks = 0
                
                # Add token to hypothesis
                hyp.append(y)
                timestamp.append(t)
                token_diversity.add(y)
                recent_tokens.append(y)
                
                # Update n-gram counters for repetition detection
                for n in n_gram_counters:
                    if len(hyp) >= context_size + n:
                        n_gram_counters[n][tuple(hyp[-n:])] += 1
                
                # Prepare decoder for next step
                decoder_input = torch.tensor([hyp[-context_size:]], device=device).reshape(
                    1, context_size
                )
                
                decoder_out = model.decoder(decoder_input, need_pad=False)
                decoder_out = model.joiner.decoder_proj(decoder_out)
                
                # Update counters
                sym_per_utt += 1
                sym_per_frame += 1
        
        except Exception as e:
            logging.warning(f"Error during greedy search: {e}")
            # Move to next frame on error
            sym_per_frame = 0
            t += 1
    
    # Remove context from final hypothesis
    hyp = hyp[context_size:]
    
    # Format and return results
    if not return_timestamps:
        return hyp
    else:
        return DecodingResults(
            hyps=[hyp],
            timestamps=[timestamp],
        )


def greedy_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    blank_penalty: float = 0.0,
    max_output_length: int = 100,
    repetition_penalty: float = 1.5,
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """XLSR-specific greedy search for a batch of utterances.
    
    This implementation handles batched inputs with proper dimension handling
    and includes the same improvements as the single-utterance version.
    
    Args:
      model: The transducer model
      encoder_out: Encoder output, shape (B, T, C)
      encoder_out_lens: Lengths of encoder outputs, shape (B,)
      blank_penalty: Penalty for blank token
      max_output_length: Maximum output sequence length
      repetition_penalty: Penalty factor for repetitive tokens
      return_timestamps: Whether to return timestamps
      
    Returns:
      If return_timestamps is False, return a list of decoded results.
      Else, return a DecodingResults object with decoded results and timestamps.
    """
    # Handle early training phase
    epoch = getattr(model, "cur_epoch", 100)
    early_training = epoch <= 2
    
    # Adjust parameters for early training
    if early_training:
        blank_penalty = max(blank_penalty, 2.0)  # Higher blank preference in early training
        repetition_penalty = max(repetition_penalty, 3.0)  # Stronger repetition penalty
        max_output_length = min(max_output_length, 50)  # Shorter outputs in early training
    
    # Input validation
    device = next(model.parameters()).device
    batch_size = encoder_out.size(0)
    
    # Ensure encoder_out is 3D (batch, time, feature_dim)
    if encoder_out.ndim == 2:
        if encoder_out.size(0) == 1:
            encoder_out = encoder_out.unsqueeze(1)  # (1, 1, feature_dim)
        else:
            encoder_out = encoder_out.unsqueeze(0)  # (1, time, feature_dim)
    elif encoder_out.ndim == 4:
        # Handle (batch, time, s_range, feature_dim) from pruned RNN-T loss
        b, t, s, f = encoder_out.size()
        encoder_out = encoder_out.reshape(b, t * s, f)
        
    # Ensure encoder_out_lens matches batch size
    if encoder_out_lens.size(0) != batch_size:
        logging.warning(
            f"encoder_out_lens batch size {encoder_out_lens.size(0)} doesn't match encoder_out {batch_size}"
        )
        # Fix lengths to match batch size
        if encoder_out_lens.size(0) > batch_size:
            encoder_out_lens = encoder_out_lens[:batch_size]
        else:
            encoder_out_lens = torch.cat([
                encoder_out_lens,
                torch.full((batch_size - encoder_out_lens.size(0),), 
                          encoder_out.size(1), 
                          device=device,
                          dtype=encoder_out_lens.dtype)
            ])
            
    # Clamp lengths to actual encoder output size
    encoder_out_lens = torch.clamp(encoder_out_lens, min=1, max=encoder_out.size(1))
    
    blank_id = model.decoder.blank_id
    
    # Process each item in the batch
    all_hyps = []
    all_timestamps = [] if return_timestamps else None
    
    for b in range(batch_size):
        # Extract a single item's encoder output
        enc_len = encoder_out_lens[b].item()
        enc_out_item = encoder_out[b:b+1, :enc_len]
        
        # Process with greedy search
        result = greedy_search(
            model=model,
            encoder_out=enc_out_item,
            max_sym_per_frame=1,  # Standard transducer allows 1 symbol per step
            blank_penalty=blank_penalty,
            max_output_length=max_output_length,
            repetition_penalty=repetition_penalty,
            return_timestamps=return_timestamps,
        )
        
        # Extract results
        if return_timestamps:
            all_hyps.append(result.hyps[0])
            all_timestamps.append(result.timestamps[0])
        else:
            all_hyps.append(result)
    
    # Format and return results
    if not return_timestamps:
        return all_hyps
    else:
        return DecodingResults(
            hyps=all_hyps,
            timestamps=all_timestamps,
        ) 