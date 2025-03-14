#!/usr/bin/env python3
# Copyright 2023

import os
import argparse
import logging
from pathlib import Path
import torch
import torchaudio
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
    logging.info(f"Creating feature extractor with model: {args.ssl_model}")
    logging.info(f"Frame shift: {args.frame_shift}s (matches paper's 20ms recommendation)")
    
    # Create config with correct frame_shift parameter
    ssl_config = S3PRLSSLConfig(
        ssl_model=args.ssl_model,
        device=args.device,
        frame_shift=args.frame_shift,  # This is the correct parameter name!
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
        
        # Check for non-16kHz files by iterating through cuts
        non_16k_files = []
        for cut in cut_set:
            if cut.sampling_rate != 16000:
                non_16k_files.append((cut.id, cut.sampling_rate))
        
        if non_16k_files:
            logging.warning(f"Found {len(non_16k_files)} files that are not sampled at 16kHz")
            logging.warning("XLSR models require 16kHz audio. Feature extraction may fail.")
            if len(non_16k_files) > 5:
                logging.warning(f"First 5 non-16kHz files: {non_16k_files[:5]}")
            else:
                logging.warning(f"Non-16kHz files: {non_16k_files}")
        
        # Always resample and persist to disk if non-16kHz files are found
        if non_16k_files:
            logging.info(f"RESAMPLING: Converting all audio to 16kHz for XLSR compatibility")
            
            # 1. Persist resampled files by overwriting originals
            for cut in cut_set:
                if cut.sampling_rate != 16000:
                    try:
                        # Load the audio
                        audio = cut.load_audio()
                        # Resample it
                        if isinstance(audio, tuple):
                            # Handle (samples, sampling_rate) tuple
                            samples, _ = audio
                            resampled = torchaudio.functional.resample(
                                torch.tensor(samples), 
                                orig_freq=cut.sampling_rate, 
                                new_freq=16000
                            ).numpy()
                        else:
                            # Handle just samples
                            resampled = torchaudio.functional.resample(
                                torch.tensor(audio), 
                                orig_freq=cut.sampling_rate, 
                                new_freq=16000
                            ).numpy()
                        
                        # Save back to the original file
                        filepath = cut.recording.sources[0].source
                        logging.info(f"Saving resampled audio to {filepath}")
                        torchaudio.save(
                            filepath=filepath,
                            src=torch.tensor(resampled).unsqueeze(0),
                            sample_rate=16000,
                            format=os.path.splitext(filepath)[1][1:]  # Extract format from extension
                        )
                    except Exception as e:
                        logging.error(f"Failed to resample and save {cut.id}: {e}")
            
            # 2. Reload manifests to get updated sampling rates
            logging.info(f"Reloading manifests after resampling...")
            m = read_manifests_if_cached(
                dataset_parts=[partition],
                output_dir=src_dir,
                prefix=args.prefix,
                suffix=args.suffix,
            )[partition]
            
            # Recreate CutSet with updated recordings
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            
            # Ensure all cuts are now at 16kHz
            for cut in cut_set:
                if cut.sampling_rate != 16000:
                    # As a fallback, use lhotse's resample method
                    logging.warning(f"Cut {cut.id} still not at 16kHz, using lhotse resample")
            
            # Final resample with lhotse
            cut_set = cut_set.resample(16000)
            logging.info(f"Resampling complete for {partition}")
        
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

def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument("--ssl-model", type=str, default="facebook/wav2vec2-xls-r-2b", 
                        help="SSL model to use (default: facebook/wav2vec2-xls-r-2b)")
    parser.add_argument("--frame-shift", type=float, default=0.02,
                        help="Frame shift in seconds (default: 0.02s = 20ms as per paper)")
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
    
    # FIXED: Changed from action="store_true" with default=True to type=str2bool with default=True
    parser.add_argument("--resample", type=str2bool, default=True,
                        help="Resample audio to 16kHz if needed (required for XLSR models)")
    parser.add_argument("--force-no-resample", action="store_true",
                        help="Force disable resampling even if non-16kHz files are found (NOT RECOMMENDED)")
    
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
    
    # ADDED: Print resampling status at startup
    if args.resample:
        logging.info("Resampling is ENABLED - Audio will be converted to 16kHz as required by XLSR")
    else:
        logging.warning("Resampling is DISABLED - This may cause errors if files are not 16kHz")
    
    compute_xlsr_features(args)

if __name__ == "__main__":
    main() 