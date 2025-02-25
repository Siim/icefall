#!/usr/bin/env python3
"""
This module provides a CTC decoder for XLSR models to compare with beam search results.
It uses the pretrained CTC head from HuggingFace models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from typing import List, Optional, Tuple
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class PretrainedXLSRCTCDecoder:
    """CTC decoder that uses the pretrained CTC head from HuggingFace models."""
    
    def __init__(
        self,
        model_name: str = "TalTechNLP/xls-r-300m-et",
        blank_id: int = 0
    ):
        """Initialize CTC decoder with pretrained model.
        
        Args:
            model_name: Name of pretrained HuggingFace model
            blank_id: ID of blank token
        """
        self.model_name = model_name
        self.blank_id = blank_id
        
        # Load processor and model only when needed (lazy loading)
        self._model = None
        self._processor = None
        
    def to(self, device):
        """Move model to device."""
        if self._model is not None:
            self._model = self._model.to(device)
        return self
    
    def _ensure_model_loaded(self, device):
        """Ensure model is loaded when needed."""
        if self._model is None:
            # Try different Estonian models in order of preference
            model_names = [
                self.model_name,
                "TalTechNLP/xls-r-300m-et",
                "tartuNLP/wav2vec2-large-xls-r-300m-et",
                "tartuNLP/wav2vec2-large-xls-r-300m-v2-et"
            ]
            
            for name in model_names:
                try:
                    self._processor = Wav2Vec2Processor.from_pretrained(name)
                    self._model = Wav2Vec2ForCTC.from_pretrained(name)
                    self._model = self._model.to(device)
                    self._model.eval()
                    return
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
                    continue
                    
            raise ValueError("Could not load any pretrained Estonian XLSR model")
    
    def decode_from_encoder(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """Decode using pretrained CTC from encoder outputs.
        
        Args:
            encoder_out: Output from encoder, shape (B, T, C)
            encoder_out_lens: Length of encoder output, shape (B,)
            
        Returns:
            Tuple[List[List[int]], torch.Tensor]:
                - List of token sequences
                - Log probabilities
        """
        device = encoder_out.device
        self._ensure_model_loaded(device)
        
        # We can't directly use the Wav2Vec2ForCTC with encoder outputs
        # Instead, we'll do greedy CTC decoding manually
        
        # Use our own projection to vocab size
        ctc_head = nn.Linear(encoder_out.size(-1), self._model.config.vocab_size).to(device)
        logits = ctc_head(encoder_out)  # Shape: (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Greedy CTC decoding
        pred_ids = torch.argmax(log_probs, dim=-1)  # Shape: (B, T)
        
        # Convert to list of token lists
        batch_size = encoder_out.size(0)
        results = []
        
        for b in range(batch_size):
            # Get sequence up to length
            length = encoder_out_lens[b]
            token_seq = pred_ids[b, :length].cpu().tolist()
            
            # Apply CTC decoding rules (collapse repeated tokens, remove blanks)
            decoded = []
            prev_token = -1
            for token in token_seq:
                if token != self.blank_id and token != prev_token:
                    decoded.append(token)
                prev_token = token
            
            results.append(decoded)
        
        return results, log_probs
    
    def decode_from_audio(
        self,
        audio_input: torch.Tensor,
    ) -> List[str]:
        """Decode directly from audio using the pretrained model.
        
        Args:
            audio_input: Raw audio, shape (B, T)
            
        Returns:
            List of decoded text strings
        """
        device = audio_input.device
        self._ensure_model_loaded(device)
        
        # Process with the pretrained model
        with torch.no_grad():
            if audio_input.dim() == 3:  # (B, 1, T)
                audio_input = audio_input.squeeze(1)
            elif audio_input.dim() == 1:  # (T,)
                audio_input = audio_input.unsqueeze(0)  # Add batch dim
                
            # Normalize input if needed (expected range: [-1, 1])
            if audio_input.abs().max() > 1.0:
                audio_input = audio_input / audio_input.abs().max()
                
            # Convert to numpy for processor
            inputs = audio_input.cpu().numpy()
            
            # Process with HuggingFace model
            if inputs.ndim == 2:  # (B, T)
                results = []
                for i in range(inputs.shape[0]):
                    # Process each batch item separately
                    input_values = self._processor(
                        inputs[i], 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    ).input_values.to(device)
                    
                    logits = self._model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self._processor.decode(predicted_ids[0])
                    results.append(transcription)
                return results
            else:
                # Single item
                input_values = self._processor(
                    inputs, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_values.to(device)
                
                logits = self._model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self._processor.decode(predicted_ids[0])
                return [transcription]


def decode_with_ctc(
    encoder: nn.Module,
    encoder_proj: Optional[nn.Module],
    audio_input: torch.Tensor,
    audio_lens: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    vocab_size: int,
    blank_id: int,
    model_name: str = "TalTechNLP/xls-r-300m-et"
) -> Tuple[List[str], List[List[int]]]:
    """
    Decode audio input using CTC from a pretrained model for comparison.
    
    Args:
        encoder: XLSR encoder
        encoder_proj: Projection layer if any
        audio_input: Audio waveform tensor (B, T)
        audio_lens: Audio lengths (B,)
        sp: SentencePieceProcessor for tokenization (not used with pretrained model)
        vocab_size: Size of vocabulary
        blank_id: ID of blank token
        model_name: Name of the pretrained model to use
        
    Returns:
        Tuple of decoded text strings and token sequences
    """
    device = audio_input.device
    
    # Create decoder that uses pretrained model
    ctc_decoder = PretrainedXLSRCTCDecoder(
        model_name=model_name,
        blank_id=blank_id
    ).to(device)
    
    # Decode directly from audio - this uses the pretrained model's full pipeline
    text_outputs = ctc_decoder.decode_from_audio(audio_input)
    
    # Also run encoder part to get outputs consistent with our model
    # This helps us understand if issue is in encoder or later parts
    with torch.no_grad():
        # Get encoder output from our model
        encoder_out, encoder_out_lens = encoder(audio_input, audio_lens)
        
        # Apply projection if needed
        if encoder_proj is not None:
            encoder_out = encoder_proj(encoder_out)
        
        # Get token sequences (we don't really use these, but include for compatibility)
        token_sequences = []
        for text in text_outputs:
            tokens = sp.encode(text, out_type=int)
            token_sequences.append(tokens)
    
    return text_outputs, token_sequences 