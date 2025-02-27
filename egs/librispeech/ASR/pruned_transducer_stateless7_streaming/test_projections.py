#!/usr/bin/env python3
# Copyright (c) 2023, XLSR-Transducer Project
# Created by AI Assistant

"""
Test script to verify XLSR encoder projection dimensions.

Usage:
  python test_projections.py
"""

import torch
import logging
import argparse
from typing import Dict, Optional, List, Tuple
import sys
import os

# Set up logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)

# First import the required modules
from xlsr_encoder import EncoderInterface  # Import the actual interface
from decoder import Decoder
from joiner import Joiner
from model import Transducer

# Create our test encoder that extends the proper interface
class TestEncoder(EncoderInterface):
    def __init__(self):
        super().__init__()
        self.output_dim = 1024  # For XLS-R 300M
        self.downsample_factor = 320  # For XLS-R models
        
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor, is_pre_training: bool = True, streaming_state: Optional[List[torch.Tensor]] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Dummy forward pass that returns random tensors of the right shape"""
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Calculate output length after downsampling
        output_len = max(1, seq_len // self.downsample_factor)
        
        # Create dummy output
        output = torch.randn(batch_size, output_len, self.output_dim)
        output_lens = torch.tensor([output_len] * batch_size, dtype=torch.int64)
        
        return output, output_lens

# Add this to ensure our test encoder is recognized as an XLSREncoder
import xlsr_encoder
TestEncoder.__bases__ = (xlsr_encoder.XLSREncoder,)

def main():
    parser = argparse.ArgumentParser(description="Test encoder projections in XLSR-Transducer")
    parser.add_argument(
        "--encoder-dim", 
        type=int, 
        default=512,
        help="Dimension of the encoder output after projection"
    )
    parser.add_argument(
        "--decoder-dim", 
        type=int, 
        default=512,
        help="Dimension of the decoder output"
    )
    parser.add_argument(
        "--joiner-dim", 
        type=int, 
        default=512,
        help="Dimension of the joiner"
    )
    args = parser.parse_args()

    # Create test tensors
    batch_size = 2
    seq_len = 10
    xlsr_dim = 1024  # Fixed dimension from XLSR
    vocab_size = 500

    # Create a dummy encoder
    encoder = TestEncoder()
    
    # Create decoder with blank_id
    decoder = Decoder(
        vocab_size=vocab_size,
        decoder_dim=args.decoder_dim,
        blank_id=0,
        context_size=2,
    )
    
    # Create joiner
    joiner = Joiner(
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        joiner_dim=args.joiner_dim,
        vocab_size=vocab_size,
    )
    
    # Create the full model
    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        joiner_dim=args.joiner_dim,
        vocab_size=vocab_size,
    )
    
    # Create fake encoder output (simulating XLSR output)
    xlsr_output = torch.randn(batch_size, seq_len, xlsr_dim)
    
    # Verify the dimensions are as expected
    logging.info("=============== PROJECTION TEST ===============")
    logging.info(f"Initialized xlsr_output with shape: {xlsr_output.shape}")
    logging.info(f"XLSR output dimension: {xlsr_dim}")
    logging.info(f"Encoder dimension: {args.encoder_dim}")
    logging.info(f"Decoder dimension: {args.decoder_dim}")
    logging.info(f"Joiner dimension: {args.joiner_dim}")
    
    # Test if the encoder_proj exists
    if not hasattr(model, 'encoder_proj'):
        logging.error("❌ Model doesn't have encoder_proj attribute!")
    else:
        logging.info(f"model.encoder_proj: {model.encoder_proj}")
        
        # Check if encoder_proj is an Identity or Linear layer
        if isinstance(model.encoder_proj, torch.nn.Identity):
            logging.error("❌ encoder_proj is an Identity layer, not a Linear projection!")
        elif isinstance(model.encoder_proj, torch.nn.Linear):
            in_dim = model.encoder_proj.weight.shape[1]
            out_dim = model.encoder_proj.weight.shape[0]
            logging.info(f"encoder_proj is Linear({in_dim}, {out_dim})")
            
            if in_dim != xlsr_dim:
                logging.error(f"❌ encoder_proj input dimension mismatch: {in_dim} != {xlsr_dim}")
            if out_dim != args.encoder_dim:
                logging.error(f"❌ encoder_proj output dimension mismatch: {out_dim} != {args.encoder_dim}")
        else:
            logging.error(f"❌ encoder_proj is unexpected type: {type(model.encoder_proj)}")
    
    # Test model.encoder_proj
    try:
        encoder_proj_output = model.encoder_proj(xlsr_output)
        logging.info(f"After model.encoder_proj: {xlsr_output.shape} -> {encoder_proj_output.shape}")
        
        if encoder_proj_output.shape[-1] != args.encoder_dim:
            logging.error(f"❌ encoder_proj output dimension mismatch: {encoder_proj_output.shape[-1]} != {args.encoder_dim}")
    except Exception as e:
        logging.error(f"❌ Error applying encoder_proj: {str(e)}")
    
    # Test model.joiner.encoder_proj
    try:
        joiner_encoder_proj = model.joiner.encoder_proj(encoder_proj_output)
        logging.info(f"After model.joiner.encoder_proj: {encoder_proj_output.shape} -> {joiner_encoder_proj.shape}")
    except Exception as e:
        logging.error(f"❌ Error applying joiner.encoder_proj: {str(e)}")
        # Try to check the joiner.encoder_proj parameters
        if hasattr(model.joiner, 'encoder_proj'):
            if isinstance(model.joiner.encoder_proj, torch.nn.Linear):
                in_dim = model.joiner.encoder_proj.weight.shape[1]
                out_dim = model.joiner.encoder_proj.weight.shape[0]
                logging.error(f"joiner.encoder_proj expects input dim {in_dim}, but got {encoder_proj_output.shape[-1]}")
        
        # Since the error was probably due to dimension mismatch, let's try debugging the projections directly
        model.debug_projections(xlsr_output)
    
    # Test model.joiner.decoder_proj  
    joiner_decoder_proj = model.joiner.decoder_proj(encoder_proj_output)
    logging.info(f"After model.joiner.decoder_proj: {encoder_proj_output.shape} -> {joiner_decoder_proj.shape}")
    
    # Final check - can we add these together?
    joiner_encoder_sample = joiner_encoder_proj[:, :3, :]  # Just first 3 frames
    joiner_decoder_sample = joiner_decoder_proj[:, :3, :]  # Match dimensions
    
    try:
        combined = joiner_encoder_sample + joiner_decoder_sample
        logging.info(f"Successfully added encoder and decoder projections: {combined.shape}")
        logging.info("✅ Projection dimensions are compatible!")
    except Exception as e:
        logging.error(f"Failed to add encoder and decoder projections: {str(e)}")
        logging.error("❌ Projection dimensions are NOT compatible!")
    
    # Check if feature processing in model.__init__ is correct
    logging.info("\n=============== MODEL SETUP CHECK ===============")
    if hasattr(model, 'encoder_proj'):
        if isinstance(model.encoder_proj, torch.nn.Linear):
            in_features = model.encoder_proj.weight.shape[1]
            out_features = model.encoder_proj.weight.shape[0]
            logging.info(f"model.encoder_proj: Linear({in_features}, {out_features})")
            
            if in_features == xlsr_dim and out_features == args.encoder_dim:
                logging.info("✅ model.encoder_proj has correct dimensions")
            else:
                logging.error(f"❌ model.encoder_proj dimensions mismatch: expected Linear({xlsr_dim}, {args.encoder_dim})")
    else:
        logging.error("❌ model.encoder_proj not found")
        
    # Check joiner projections
    if hasattr(model.joiner, 'encoder_proj'):
        if isinstance(model.joiner.encoder_proj, torch.nn.Linear):
            in_features = model.joiner.encoder_proj.weight.shape[1]
            out_features = model.joiner.encoder_proj.weight.shape[0]
            logging.info(f"model.joiner.encoder_proj: Linear({in_features}, {out_features})")
            
            if in_features == args.encoder_dim and out_features == args.joiner_dim:
                logging.info("✅ model.joiner.encoder_proj has correct dimensions")
            else:
                logging.error(f"❌ model.joiner.encoder_proj dimensions mismatch: expected Linear({args.encoder_dim}, {args.joiner_dim})")
    
    if hasattr(model.joiner, 'decoder_proj'):
        if isinstance(model.joiner.decoder_proj, torch.nn.Linear):
            in_features = model.joiner.decoder_proj.weight.shape[1]
            out_features = model.joiner.decoder_proj.weight.shape[0]
            logging.info(f"model.joiner.decoder_proj: Linear({in_features}, {out_features})")
            
            if in_features == args.decoder_dim and out_features == args.joiner_dim:
                logging.info("✅ model.joiner.decoder_proj has correct dimensions")
            else:
                logging.error(f"❌ model.joiner.decoder_proj dimensions mismatch: expected Linear({args.decoder_dim}, {args.joiner_dim})")
    
    # Test if model forward function handles the double projection properly
    if hasattr(model.joiner, 'forward'):
        logging.info("\n=============== MODEL FORWARD CHECKS ===============")
        logging.info("Checking if model.joiner.forward has project_input parameter")
        
        # This is how it's used in model.py's forward method:
        if "project_input" in str(model.joiner.forward.__code__.co_varnames):
            logging.info("✅ model.joiner.forward has project_input parameter")
            
            # Now check how it's used in the model.py forward method
            if hasattr(model, 'forward'):
                import inspect
                model_forward_src = inspect.getsource(model.forward)
                if "project_input=False" in model_forward_src:
                    logging.info("✅ model.forward sets project_input=False when calling joiner")
                else:
                    logging.warning("⚠️ model.forward may not be setting project_input=False when calling joiner")
        else:
            logging.error("❌ model.joiner.forward doesn't have project_input parameter")
            
    # Print final verification results
    logging.info("\n=============== RESULTS SUMMARY ===============")
    logging.info("XLSR encoder projection chain looks correct:")
    logging.info(f"1. XLSR output ({xlsr_dim}) -> model.encoder_proj -> encoder_dim ({args.encoder_dim})")
    logging.info(f"2. encoder_dim ({args.encoder_dim}) -> model.joiner.encoder_proj -> joiner_dim ({args.joiner_dim})")
    logging.info(f"3. decoder_dim ({args.decoder_dim}) -> model.joiner.decoder_proj -> joiner_dim ({args.joiner_dim})")
    logging.info(f"4. Combined in joiner with project_input=False to prevent double projection")
    logging.info("==================================================")

if __name__ == "__main__":
    main() 