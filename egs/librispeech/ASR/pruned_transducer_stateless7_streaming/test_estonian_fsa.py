#!/usr/bin/env python3

import argparse
import logging
import torch
from pathlib import Path
from typing import List, Tuple
import k2

from xlsr_encoder import XLSREncoder
from estonian_decoder import (
    create_estonian_token_table,
    create_estonian_decoding_graph,
    fast_beam_search_one_best
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-file",
        type=str,
        required=True,
        help="Path to Estonian vocabulary file"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test file with audio paths and transcripts"
    )
    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="Beam size for FSA search"
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="Maximum FSA states to keep"
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="Maximum right contexts to keep"
    )
    return parser.parse_args()

def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create token table and decoding graph
    token_table = create_estonian_token_table(args.vocab_file)
    logging.info(f"Created token table with {len(token_table)} tokens")

    decoding_graph = create_estonian_decoding_graph(
        token_table=token_table,
        num_tokens=len(token_table),
        device=device
    )
    logging.info(f"Created decoding graph with {decoding_graph.num_states} states")

    # Load test data
    from estonian_dataset import EstonianASRDataset, collate_fn
    test_dataset = EstonianASRDataset(args.test_file)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collate_fn
    )

    # Initialize encoder
    encoder = XLSREncoder(
        model_name="facebook/wav2vec2-xls-r-300m",
        decode_chunk_size=20480  # 1280ms at 16kHz
    ).to(device)
    encoder.eval()

    # Test decoding
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Get encoder output
            inputs = batch["inputs"].to(device)
            input_lens = batch["supervisions"]["num_frames"].to(device)
            encoder_out, encoder_out_lens = encoder(inputs, input_lens)

            # Run FSA decoding
            hyp_tokens = fast_beam_search_one_best(
                model=None,  # We don't need the full model for this test
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                beam=args.beam,
                max_states=args.max_states,
                max_contexts=args.max_contexts,
                decoding_graph=decoding_graph
            )

            # Convert tokens to text
            ref_text = batch["supervisions"]["text"][0]
            hyp_text = " ".join([token_table[i] for i in hyp_tokens[0]])

            logging.info(f"\nTest sample {batch_idx + 1}:")
            logging.info(f"Reference: {ref_text}")
            logging.info(f"Hypothesis: {hyp_text}")

            if batch_idx >= 4:  # Test first 5 samples
                break

if __name__ == "__main__":
    main()