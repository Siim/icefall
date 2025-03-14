#!/usr/bin/env python3
# Copyright 2023

import os
import argparse
import logging
from pathlib import Path
import torch
from lhotse import S3PRLSSL, CutSet, NumpyFilesWriter, S3PRLSSLConfig
from lhotse.recipes.utils import read_manifests_if_cached
import platform

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slows things down.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def compute_xlsr_features(args):
    """
    Extract XLSR features from audio files in manifests using lhotse
    """
    src_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which manifest parts to process
    if args.dataset_parts:
        dataset_parts = args.dataset_parts.split(',')
    else:
        # Use all available parts if not specified
        dataset_parts = [p.stem.replace(f"{args.prefix}_", "") for p in src_dir.glob(f"{args.prefix}_*.jsonl.gz")]
    
    logging.info(f"Processing dataset parts: {dataset_parts}")
    
    # Read manifests
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
    )
    
    assert manifests is not None, "Failed to read manifests"
    
    # Create the SSL feature extractor with specific frame parameters
    logging.info(f"Creating feature extractor with model: {args.model_name}")
    logging.info(f"Frame duration: {args.frame_duration}ms, Frame stride: {args.frame_stride}ms")
    
    # Create config with frame parameters
    ssl_config = S3PRLSSLConfig(
        ssl_model=args.ssl_model,
        model_name=args.model_name,
        device=args.device,
        # Add frame duration and stride parameters
        window_ms=args.frame_duration,  # Frame duration in ms
        stride_ms=args.frame_stride,    # Frame stride in ms
    )
    
    extractor = S3PRLSSL(ssl_config)
    
    # Process each partition
    for partition, m in manifests.items():
        cuts_filename = f"{args.prefix}_cuts_{partition}.{args.suffix}"
        if (output_dir / cuts_filename).is_file() and not args.force:
            logging.info(f"{partition} already exists - skipping.")
            continue
        
        logging.info(f"Processing {partition}")
        
        # Create CutSet from manifests
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )
        
        # Resample to 16kHz if needed
        if args.resample:
            logging.info(f"Resampling audio to 16kHz")
            cut_set = cut_set.resample(16000)
        
        # Check if we should use parallel processing
        if args.num_jobs > 1 and platform.system() == "Darwin":
            logging.warning("Parallel processing may not work on macOS. Falling back to single process.")
            logging.warning("This is due to a limitation with pickling in the S3PRL library.")
            logging.info("Using 1 job for feature extraction")
            num_jobs = 1
        else:
            logging.info(f"Using {args.num_jobs} parallel jobs for feature extraction")
            num_jobs = args.num_jobs
        
        # Compute and store features
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/{args.prefix}_feats_{partition}",
            storage_type=NumpyFilesWriter,
            num_jobs=num_jobs,
        )
        
        # Save the cuts with features
        cut_set.to_file(output_dir / cuts_filename)
    
    logging.info(f"Feature extraction complete! Saved to {output_dir}")

def main():
    # Set multiprocessing start method to 'fork' on Unix systems
    if platform.system() != "Windows":
        try:
            import torch.multiprocessing as mp
            mp.set_start_method('fork', force=True)
            logging.info("Set multiprocessing start method to 'fork'")
        except RuntimeError:
            logging.warning("Could not set multiprocessing start method to 'fork'")
    
    parser = argparse.ArgumentParser(description="Extract XLSR features using lhotse")
    parser.add_argument("--manifest-dir", type=str, default="data/manifests", 
                        help="Directory containing the manifest files")
    parser.add_argument("--output-dir", type=str, default="data/ssl", 
                        help="Output directory for features")
    parser.add_argument("--ssl-model", type=str, default="wav2vec2", 
                        help="SSL model to use (default: wav2vec2)")
    parser.add_argument("--model-name", type=str, default="TalTechNLP/xls-r-300m-et",
                        help="Specific model name/path to use (default: TalTechNLP/xls-r-300m-et)")
    parser.add_argument("--frame-duration", type=int, default=25,
                        help="Frame duration in milliseconds (default: 25ms as per paper)")
    parser.add_argument("--frame-stride", type=int, default=20,
                        help="Frame stride in milliseconds (default: 20ms as per paper)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use for computation")
    parser.add_argument("--prefix", type=str, default="et", 
                        help="Prefix for manifest files")
    parser.add_argument("--suffix", type=str, default="jsonl.gz", 
                        help="Suffix for manifest files")
    parser.add_argument("--dataset-parts", type=str, 
                        help="Comma-separated list of dataset parts to process (e.g., 'train,dev,test')")
    parser.add_argument("--force", action="store_true", 
                        help="Force recomputation of features even if they already exist")
    parser.add_argument("--resample", action="store_true", default=True,
                        help="Resample audio to 16kHz if needed (required for XLSR models)")
    parser.add_argument("--num-jobs", type=int, default=3,
                        help="Number of parallel jobs to use for feature extraction")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    compute_xlsr_features(args)

if __name__ == "__main__":
    main() 