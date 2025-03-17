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
    
    # Create directory for resampled audio files
    resampled_dir = Path(args.output_dir) / "resampled_audio"
    os.makedirs(resampled_dir, exist_ok=True)
    logging.info(f"Resampled audio will be saved to {resampled_dir}")
    
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
            
            # Minimum valid duration for XLS-R model (in samples at 16kHz)
            min_valid_samples = 400  # Ensure at least 25ms (400 samples at 16kHz) for the convolution kernel
            
            # Track resampled file paths
            resampled_file_map = {}
            
            # 1. Create resampled copies instead of overwriting originals
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
                                torch.tensor(samples, dtype=torch.float32), 
                                orig_freq=cut.sampling_rate, 
                                new_freq=16000
                            )
                        else:
                            # Handle just samples
                            resampled = torchaudio.functional.resample(
                                torch.tensor(audio, dtype=torch.float32), 
                                orig_freq=cut.sampling_rate, 
                                new_freq=16000
                            )
                        
                        # Ensure tensor is 2D [channels, samples]
                        if resampled.dim() == 1:
                            # If 1D tensor (just samples), reshape to [1, samples]
                            resampled = resampled.unsqueeze(0)
                        
                        # Check if audio is too short and pad if necessary
                        if resampled.size(1) < min_valid_samples:
                            logging.warning(f"Audio {cut.id} is too short ({resampled.size(1)} samples). Padding to {min_valid_samples} samples.")
                            padding = torch.zeros(resampled.size(0), min_valid_samples - resampled.size(1), dtype=resampled.dtype)
                            resampled = torch.cat([resampled, padding], dim=1)
                        
                        # Create a new filename for the resampled audio instead of overwriting
                        original_filepath = cut.recording.sources[0].source
                        filename = os.path.basename(original_filepath)
                        resampled_filepath = resampled_dir / f"resampled_{filename}"
                        
                        logging.info(f"Saving resampled audio to {resampled_filepath}")
                        
                        # Verify tensor shape before saving
                        if resampled.dim() != 2:
                            logging.warning(f"Tensor shape is not 2D: {resampled.shape}. Reshaping...")
                            if resampled.dim() == 1:
                                resampled = resampled.unsqueeze(0)
                            elif resampled.dim() > 2:
                                # Take the first channel if multidimensional
                                resampled = resampled[0:1]
                        
                        logging.info(f"Final tensor shape: {resampled.shape}")
                        
                        # Save with proper format - always use WAV for consistency
                        torchaudio.save(
                            str(resampled_filepath),  # Convert Path to string
                            src=resampled,
                            sample_rate=16000,
                            format="wav"  # Use WAV format for maximum compatibility
                        )
                        
                        # Verify the file was saved correctly
                        if os.path.exists(resampled_filepath) and os.path.getsize(resampled_filepath) > 0:
                            logging.info(f"Successfully saved {resampled_filepath}")
                            # Store the mapping from original to resampled file
                            resampled_file_map[original_filepath] = str(resampled_filepath)
                        else:
                            logging.error(f"Failed to save or empty file: {resampled_filepath}")
                            
                    except Exception as e:
                        logging.error(f"Failed to resample and save {cut.id}: {e}")
                        logging.exception(e)  # Add full traceback for debugging
            
            # Modify the manifest to point to resampled files
            if resampled_file_map:
                logging.info(f"Updating manifest with {len(resampled_file_map)} resampled file paths")
                for recording in m["recordings"]:
                    for source in recording.sources:
                        if source.source in resampled_file_map:
                            source.source = resampled_file_map[source.source]
                            # Update the sampling rate in the recording metadata
                            recording.sampling_rate = 16000
            
            # Recreate CutSet with updated recordings
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            
            # Verify all cuts are now at 16kHz
            non_16k_after_resample = []
            for cut in cut_set:
                if cut.sampling_rate != 16000:
                    non_16k_after_resample.append(cut.id)
            
            if non_16k_after_resample:
                logging.warning(f"After resampling, {len(non_16k_after_resample)} cuts still not at 16kHz")
                logging.warning("Using lhotse's resample method for these cuts")
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
        
        # Add custom pre-processing for short recordings before feature extraction
        # Custom function to pad audio to minimum length if needed
        def pad_short_recordings(cut):
            if cut.duration < 0.025:  # 25ms minimum for kernel size
                try:
                    # Load audio
                    audio = cut.load_audio()
                    samples = audio[0] if isinstance(audio, tuple) else audio
                    samples_tensor = torch.tensor(samples, dtype=torch.float32)
                    
                    # Ensure it's at least 400 samples (25ms at 16kHz)
                    min_samples = 400
                    if len(samples_tensor) < min_samples:
                        # Pad with zeros to reach minimum length
                        padding_length = min_samples - len(samples_tensor)
                        padded_audio = torch.nn.functional.pad(samples_tensor, (0, padding_length))
                        
                        # Ensure tensor is 2D [channels, samples] before saving
                        if padded_audio.dim() == 1:
                            padded_audio = padded_audio.unsqueeze(0)  # Add channel dimension
                        
                        # Save back to file
                        filepath = cut.recording.sources[0].source
                        logging.info(f"Padded audio {cut.id} from {len(samples_tensor)} to {min_samples} samples")
                        
                        # Debug dimensions to ensure correctness
                        logging.info(f"Tensor shape before saving: {padded_audio.shape}")
                        
                        file_ext = os.path.splitext(filepath)[1][1:]
                        torchaudio.save(
                            filepath,
                            padded_audio,  # Already properly shaped as [channels, samples]
                            sample_rate=16000,
                            format=file_ext if file_ext else None
                        )
                except Exception as e:
                    logging.error(f"Failed to pad short recording {cut.id}: {e}")
                    logging.exception(e)  # Add full stack trace
            return cut
        
        # Apply padding to short recordings
        cut_set = CutSet.from_cuts(pad_short_recordings(cut) for cut in cut_set)
        
        # Compute and store features
        try:
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{args.prefix}_feats_{partition}",
                storage_type=NumpyFilesWriter,
                num_jobs=num_jobs,
            )
            
            # Save the cuts with features
            cut_set.to_file(output_dir / cuts_filename)
        except Exception as e:
            logging.error(f"Error during feature extraction: {e}")
            logging.exception(e)
            raise
    
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