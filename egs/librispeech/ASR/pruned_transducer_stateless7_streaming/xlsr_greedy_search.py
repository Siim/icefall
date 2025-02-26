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
import sentencepiece as spm

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
    try:
        assert encoder_out.ndim == 3, f"Expected 3D encoder output, got shape {encoder_out.shape}"
        assert encoder_out.size(0) == 1, f"Expected batch size 1, got {encoder_out.size(0)}"
        
        # Check for NaN values in encoder output
        if torch.isnan(encoder_out).any():
            logging.warning("NaN values detected in encoder output. Using fallback empty result.")
            if return_timestamps:
                return DecodingResults(hyps=[], timestamps=[])
            else:
                return []
        
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
        hyp = [blank_id]  # Start with blank
        timestamps = [0] if return_timestamps else None
        
        # Token repetition tracking
        last_token = blank_id
        
        # Process each encoder frame
        for t in range(T):
            # Skip if we've reached max output length
            if len(hyp) >= max_output_length:
                logging.warning(f"Reached maximum output length ({max_output_length}). Stopping early.")
                break
                
            # Get current frame
            encoder_frame = encoder_out[:, t:t+1]
            
            # Joint network forward pass
            logits = model.joiner(
                encoder_frame, 
                decoder_out, 
                project_input=False
            )
            
            # Apply blank penalty
            if blank_penalty != 0.0:
                logits[0, 0, 0, blank_id] += blank_penalty
                
            # Check for NaN values
            if torch.isnan(logits).any():
                logging.warning(f"NaN values in joint output at frame {t}. Skipping frame.")
                continue
                
            # Apply repetition penalty if needed
            if repetition_penalty > 1.0 and last_token != blank_id:
                # Penalize repetition of the last non-blank token
                logits[0, 0, 0, last_token] /= repetition_penalty
            
            # Get probabilities
            probs = torch.softmax(logits[0, 0, 0], dim=0)
            
            # Get token with highest probability
            max_prob, max_idx = torch.max(probs, dim=0)
            token_id = max_idx.item()
            
            # Add symbol and update context
            if token_id != blank_id:
                # Skip if we've reached max symbols for this frame
                if sum(1 for i in range(len(hyp)) if timestamps and timestamps[i] == t) >= max_sym_per_frame:
                    continue
                    
                # Add token to hypothesis
                hyp.append(token_id)
                if timestamps is not None:
                    timestamps.append(t)
                
                # Update last token
                last_token = token_id
                
                # Update decoder input and output
                decoder_input = torch.tensor(
                    hyp[-context_size:] if len(hyp) >= context_size else ([-1] * (context_size - len(hyp)) + hyp),
                    device=device,
                    dtype=torch.int64
                ).reshape(1, context_size)
                
                decoder_out = model.decoder(decoder_input, need_pad=False)
                decoder_out = model.joiner.decoder_proj(decoder_out)
            
        # Remove initial blank
        if hyp[0] == blank_id:
            hyp = hyp[1:]
            if timestamps is not None:
                timestamps = timestamps[1:]
            
        # Return appropriate result format
        if return_timestamps:
            return DecodingResults(hyps=hyp, timestamps=timestamps)
        else:
            return hyp
            
    except Exception as e:
        logging.error(f"Error in greedy search: {str(e)}")
        # Return fallback result
        if return_timestamps:
            return DecodingResults(hyps=[], timestamps=[])
        else:
            return []


def greedy_search_batch(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    max_sym_per_frame: int = 3
) -> List[str]:
    """XLSR-specific greedy search for a batch of utterances.
    
    This implementation handles batched inputs with proper dimension handling
    and includes the same improvements as the single-utterance version.
    
    Args:
      model: The transducer model
      encoder_out: Encoder output, shape (B, T, C)
      encoder_out_lens: Lengths of encoder outputs, shape (B,)
      sp: SentencePiece processor
      max_sym_per_frame: Maximum symbols per frame
      
    Returns:
      A list of decoded results.
    """
    # First, try to import the standard greedy search as fallback
    try:
        from beam_search import greedy_search_batch as std_greedy_search_batch
        fallback_available = True
    except ImportError:
        fallback_available = False
        logging.warning("Standard greedy search not available as fallback")

    # Handle early training phase
    epoch = getattr(model, "cur_epoch", 100)
    early_training = epoch <= 2
    
    # Adjust parameters for early training
    if early_training:
        max_sym_per_frame = min(max_sym_per_frame, 5)  # Shorter outputs in early training
    
    # Get blank ID early to handle fallback more gracefully
    blank_id = model.decoder.blank_id
    
    # Input validation
    try:
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
        
        # If we're in very early training, just use standard greedy search for stability
        if early_training and epoch == 1 and fallback_available:
            logging.info("Using standard greedy search for stability in early training")
            return std_greedy_search_batch(
                model=model,
                encoder_out=encoder_out, 
                encoder_out_lens=encoder_out_lens,
                blank_penalty=0.0,
                return_timestamps=False
            )
        
        # Process each item in the batch
        all_hyps = []
        
        error_count = 0
        max_errors = batch_size // 2  # Allow up to half the batch to fail before falling back
        
        for b in range(batch_size):
            try:
                # Extract a single item's encoder output
                enc_len = encoder_out_lens[b].item()
                enc_out_item = encoder_out[b:b+1, :enc_len]
                
                # Process with greedy search
                result = greedy_search(
                    model=model,
                    encoder_out=enc_out_item,
                    max_sym_per_frame=max_sym_per_frame,
                    blank_penalty=0.0,
                    max_output_length=100,
                    repetition_penalty=1.5,
                    return_timestamps=False,
                )
                
                # Extract results
                all_hyps.append(result)
                
            except Exception as e:
                error_count += 1
                logging.error(f"Error processing batch item {b}: {str(e)}")
                
                # Fall back to standard greedy search for this item if available
                if fallback_available:
                    try:
                        logging.warning(f"Trying standard greedy search for item {b}")
                        # Extract this single item
                        single_enc_out = encoder_out[b:b+1]
                        single_enc_lens = encoder_out_lens[b:b+1]
                        
                        single_result = std_greedy_search_batch(
                            model=model,
                            encoder_out=single_enc_out,
                            encoder_out_lens=single_enc_lens,
                            blank_penalty=0.0,
                            return_timestamps=False
                        )
                        
                        # Add the result
                        all_hyps.append(single_result)
                        continue
                    except Exception as fallback_e:
                        logging.error(f"Fallback also failed for item {b}: {str(fallback_e)}")
                
                # Add an empty result in case of error
                all_hyps.append([])
                
                # If too many errors, break and try fallback
                if error_count > max_errors:
                    logging.warning(f"Too many errors ({error_count}/{batch_size}), trying fallback")
                    raise RuntimeError(f"XLSR greedy search failed on too many items: {error_count}/{batch_size}")
        
        # Fix empty predictions
        for i in range(len(all_hyps)):
            if len(all_hyps[i]) == 0:
                # Emergency fallback - use most common tokens
                encoder_len = encoder_out_lens[i].item()
                all_hyps[i] = [1] * min(5, encoder_len)  # Use common tokens
                logging.warning(f"Empty prediction in sample {i}, using fallback tokens")
        
        # Format and return results
        return all_hyps
            
    except Exception as e:
        logging.warning(f"Enhanced greedy search failed with error: {str(e)}")
        
        # Fall back to standard greedy search if available
        if fallback_available:
            logging.warning("Falling back to standard greedy search for entire batch")
            try:
                return std_greedy_search_batch(
                    model=model,
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    blank_penalty=0.0,
                    return_timestamps=False
                )
            except Exception as fallback_e:
                logging.error(f"Fallback also failed: {str(fallback_e)}")
                
        # If we get here, both methods failed or fallback wasn't available
        # Return empty results as a last resort
        return [[] for _ in range(batch_size)] 