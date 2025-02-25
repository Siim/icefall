#!/usr/bin/env python3
"""
This module provides a CTC decoder for XLSR models to compare with beam search results.
It can be used to verify if the issue is with the encoder input processing or the decoder/joiner.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from typing import List, Optional, Tuple

class XLSRCTCDecoder:
    """CTC decoder for XLSR models that adds a CTC head on top of encoder outputs."""
    
    def __init__(
        self,
        encoder_dim: int = 1024,  # XLSR output dimension
        vocab_size: int = 500,
        blank_id: int = 0
    ):
        """Initialize CTC decoder.
        
        Args:
            encoder_dim: Dimension of encoder output
            vocab_size: Size of vocabulary
            blank_id: ID of blank token
        """
        self.encoder_dim = encoder_dim
        self.vocab_size = vocab_size
        self.blank_id = blank_id
        
        # Create a simple CTC head (linear projection)
        self.ctc_head = nn.Linear(encoder_dim, vocab_size)
    
    def to(self, device):
        """Move model to device."""
        self.ctc_head = self.ctc_head.to(device)
        return self
    
    def decode(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Decode using CTC.
        
        Args:
            encoder_out: Output from encoder, shape (B, T, C)
            encoder_out_lens: Length of encoder output, shape (B,)
            
        Returns:
            Tuple[List[List[int]], torch.Tensor]:
                - List of token sequences
                - Log probabilities, shape (B, T, V)
        """
        # Project encoder output to vocabulary size
        logits = self.ctc_head(encoder_out)  # Shape: (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get best path (greedy) decoding
        pred_tokens = torch.argmax(log_probs, dim=-1)  # Shape: (B, T)
        
        # Convert to list of token lists (handle batching)
        batch_size = encoder_out.size(0)
        results = []
        
        for b in range(batch_size):
            # Get sequence up to length
            length = encoder_out_lens[b]
            token_seq = pred_tokens[b, :length].cpu().tolist()
            
            # Apply CTC decoding rules (collapse repeated tokens, remove blanks)
            decoded = []
            prev_token = -1
            for token in token_seq:
                if token != self.blank_id and token != prev_token:
                    decoded.append(token)
                prev_token = token
            
            results.append(decoded)
        
        return results, log_probs


def decode_with_ctc(
    encoder: nn.Module,
    encoder_proj: Optional[nn.Module],
    audio_input: torch.Tensor,
    audio_lens: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
    blank_id: int
) -> Tuple[List[str], List[List[int]]]:
    """
    Decode audio input using CTC instead of beam search.
    
    Args:
        encoder: XLSR encoder
        encoder_proj: Projection layer if any
        audio_input: Audio waveform tensor (B, T)
        audio_lens: Audio lengths (B,)
        sp: SentencePieceProcessor for tokenization
        vocab_size: Size of vocabulary
        blank_id: ID of blank token
        
    Returns:
        Tuple of decoded text strings and token sequences
    """
    device = audio_input.device
    
    # Create CTC decoder
    ctc_decoder = XLSRCTCDecoder(
        encoder_dim=1024,  # XLSR output dimension
        vocab_size=vocab_size,
        blank_id=blank_id
    ).to(device)
    
    # Encode the input
    with torch.no_grad():
        # Get encoder output
        encoder_out, encoder_out_lens = encoder(audio_input, audio_lens)
        
        # Apply projection if needed
        if encoder_proj is not None:
            encoder_out = encoder_proj(encoder_out)
        
        # Get CTC decoding
        token_sequences, _ = ctc_decoder.decode(encoder_out, encoder_out_lens)
        
        # Convert tokens to text
        text_outputs = []
        for tokens in token_sequences:
            text = sp.decode(tokens)
            text_outputs.append(text)
    
    return text_outputs, token_sequences 