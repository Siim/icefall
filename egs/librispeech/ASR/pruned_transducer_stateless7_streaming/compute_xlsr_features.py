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
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import multiprocessing

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slows things down.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Function to resample a single file using ffmpeg
def resample_file_with_ffmpeg(args):
    original_filepath, resampled_filepath, cut_id = args
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(resampled_filepath), exist_ok=True)
        
        # Construct and run ffmpeg command with higher thread count for better CPU utilization
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-i", original_filepath,  # Input file
            "-ar", "16000",  # Target sample rate
            "-ac", "1",      # Mono audio
            "-sample_fmt", "s16",  # 16-bit PCM
            "-threads", "4",  # Use more threads
            str(resampled_filepath)  # Output file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"ffmpeg error for {cut_id}: {result.stderr}")
            return None
        
        # Verify the file was created successfully
        if os.path.exists(resampled_filepath) and os.path.getsize(resampled_filepath) > 0:
            return (original_filepath, str(resampled_filepath))
        else:
            logging.error(f"Failed to create or empty file: {resampled_filepath}")
            return None
    except Exception as e:
        logging.error(f"Failed to resample {cut_id}: {e}")
        return None

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
    
    # After loading manifests, add a debug section
    logging.info(f"Creating feature extractor with model: {args.ssl_model}")
    logging.info(f"Frame shift: {args.frame_shift}s (matches paper's 20ms recommendation)")
    
    # Add debug summary of manifests
    for part_name, manifest in manifests.items():
        recordings_count = len(manifest["recordings"])
        supervisions_count = len(manifest["supervisions"])
        logging.info(f"Loaded manifest for {part_name}: {recordings_count} recordings, {supervisions_count} supervisions")
        
        # Sample a few recordings to verify paths
        if recordings_count > 0:
            sample_recordings = manifest["recordings"][:3]
            logging.info(f"Sample recording paths for {part_name}:")
            for rec in sample_recordings:
                for source in rec.sources:
                    logging.info(f"  - {source.source} (sampling rate: {rec.sampling_rate})")
    
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
            
            # Prepare list of files to resample
            resample_tasks = []
            for cut in cut_set:
                if cut.sampling_rate != 16000:
                    original_filepath = cut.recording.sources[0].source
                    filename = os.path.basename(original_filepath)
                    resampled_filepath = resampled_dir / f"resampled_{filename}"
                    resample_tasks.append((original_filepath, resampled_filepath, cut.id))
            
            # Set up parallel processing
            total_cores = multiprocessing.cpu_count()
            min_cores_to_use = max(int(total_cores * 2/3), 2)  # Use at least 2/3 of cores, minimum 2
            num_workers = max(min(args.num_jobs, total_cores), min_cores_to_use)
            logging.info(f"Using {num_workers} parallel processes for resampling (out of {total_cores} available cores)")
            
            # Process files in parallel with progress bar
            successful_resamplings = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Map tasks to executor with progress bar
                for result in tqdm(
                    executor.map(resample_file_with_ffmpeg, resample_tasks),
                    total=len(resample_tasks),
                    desc="Resampling audio files",
                    unit="file"
                ):
                    if result:
                        successful_resamplings.append(result)
            
            # Convert results to map
            for original_path, resampled_path in successful_resamplings:
                resampled_file_map[original_path] = resampled_path
            
            logging.info(f"Successfully resampled {len(resampled_file_map)} of {len(resample_tasks)} files")
            
            # Modify the manifest to point to resampled files
            if resampled_file_map:
                logging.info(f"Updating manifest with {len(resampled_file_map)} resampled file paths")
                
                # Store duration and num_samples updates
                recording_updates = {}
                
                # First calculate the correct new num_samples for each resampled file
                for filepath, resampled_filepath in resampled_file_map.items():
                    try:
                        # Load the audio to get actual info
                        info = torchaudio.info(resampled_filepath)
                        # Store the actual number of samples and duration for later update
                        recording_updates[filepath] = {
                            "num_samples": info.num_frames,
                            "duration": info.num_frames / 16000  # Calculate duration from samples at 16kHz
                        }
                        logging.info(f"Updated metadata for {resampled_filepath}: {info.num_frames} samples, {info.num_frames/16000:.3f}s")
                    except Exception as e:
                        logging.error(f"Failed to get info for resampled file {resampled_filepath}: {e}")
                
                # Update recordings with new paths and corrected metadata
                for recording in m["recordings"]:
                    for source in recording.sources:
                        original_path = source.source
                        if original_path in resampled_file_map:
                            # Update path
                            source.source = resampled_file_map[original_path]
                            # Update sampling rate
                            recording.sampling_rate = 16000
                            
                            # Update num_samples and duration if available
                            if original_path in recording_updates:
                                recording.num_samples = recording_updates[original_path]["num_samples"]
                                recording.duration = recording_updates[original_path]["duration"]
                                logging.info(f"Updated recording {recording.id}: {recording.num_samples} samples, {recording.duration:.3f}s")
                
                # Update supervision segments to ensure they don't exceed recording durations
                recording_durations = {rec.id: rec.duration for rec in m["recordings"]}
                for supervision in m["supervisions"]:
                    if supervision.recording_id in recording_durations:
                        rec_duration = recording_durations[supervision.recording_id]
                        
                        # If supervision extends past recording end, adjust it
                        if supervision.start + supervision.duration > rec_duration:
                            old_duration = supervision.duration
                            supervision.duration = max(0.001, rec_duration - supervision.start)
                            logging.info(f"Adjusted supervision {supervision.id} duration from {old_duration:.3f}s to {supervision.duration:.3f}s to match recording duration {rec_duration:.3f}s")
                
                # Save the updated manifest back to disk with updated paths
                manifest_path = Path(src_dir) / f"{args.prefix}_{partition}.{args.suffix}"
                logging.info(f"Saving updated manifest to {manifest_path}")
                from lhotse import RecordingSet, SupervisionSet
                
                # Create recording and supervision sets
                recording_set = RecordingSet.from_recordings(m["recordings"])
                supervision_set = SupervisionSet.from_segments(m["supervisions"])
                
                # Save them back to disk
                recording_set.to_file(manifest_path.with_suffix(".recordings.jsonl.gz"))
                supervision_set.to_file(manifest_path.with_suffix(".supervisions.jsonl.gz"))
                
                # Update our in-memory reference too
                m = {
                    "recordings": recording_set,
                    "supervisions": supervision_set
                }
                
                # After resampling, add verification of file paths in the updated manifest
                logging.info(f"Manifest updated and saved with resampled file paths")
                
                # Verify file paths in updated manifest
                logging.info(f"Verifying updated file paths in manifest for {partition}:")
                sample_count = 0
                for rec in recording_set:
                    if sample_count >= 3:  # Just check first 3 recordings
                        break
                    for source in rec.sources:
                        if os.path.exists(source.source):
                            file_status = "exists"
                        else:
                            file_status = "MISSING"
                        logging.info(f"  - {source.source} (sampling rate: {rec.sampling_rate}) - {file_status}")
                    sample_count += 1
            
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
                        
                        # Get current file path (which should now point to the resampled file)
                        filepath = cut.recording.sources[0].source
                        logging.info(f"Padded audio {cut.id} from {len(samples_tensor)} to {min_samples} samples")
                        
                        # Debug dimensions to ensure correctness
                        logging.info(f"Tensor shape before saving: {padded_audio.shape}")
                        
                        # Extract file extension or default to wav
                        file_ext = os.path.splitext(filepath)[1][1:] or "wav"
                        
                        # Save with proper format
                        torchaudio.save(
                            filepath,
                            padded_audio,  # Already properly shaped as [channels, samples]
                            sample_rate=16000,
                            format=file_ext
                        )
                        
                        # Verify the file was saved correctly
                        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                            logging.info(f"Successfully padded and saved {filepath}")
                        else:
                            logging.error(f"Failed to save or empty file after padding: {filepath}")
                except Exception as e:
                    logging.error(f"Failed to pad short recording {cut.id}: {e}")
                    logging.exception(e)  # Add full stack trace
            return cut
        
        # Apply padding to short recordings
        cut_set = CutSet.from_cuts(pad_short_recordings(cut) for cut in cut_set)
        
        # Verify audio files exist before feature extraction
        missing_files = []
        sample_cuts = list(cut_set)[:5]
        logging.info(f"Verifying audio files exist for {partition} before feature extraction:")
        for cut in sample_cuts:
            filepath = cut.recording.sources[0].source
            if os.path.exists(filepath):
                file_status = "exists"
            else:
                file_status = "MISSING"
                missing_files.append(filepath)
            logging.info(f"  - {cut.id}: {filepath} ({cut.sampling_rate}Hz) - {file_status}")
        
        if missing_files:
            logging.warning(f"Found {len(missing_files)} missing audio files before feature extraction!")
            logging.warning(f"Example missing files: {missing_files[:5]}")
        
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