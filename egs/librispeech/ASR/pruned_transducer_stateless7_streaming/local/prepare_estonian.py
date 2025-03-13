#!/usr/bin/env python3
# Copyright 2023

import os
import argparse
import logging
from pathlib import Path
import soundfile as sf
from typing import Dict, List, Tuple, Optional
from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet, AudioSource
from lhotse.utils import Seconds

def load_estonian_dataset(filepath: str, max_files: Optional[int] = None) -> List[Dict]:
    """Load Estonian dataset from a file with format 'audio_path|text|speaker_id'"""
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|', 2)
            if len(parts) >= 2:
                audio_path = parts[0]
                text = parts[1]
                speaker_id = parts[2] if len(parts) > 2 else "unknown"
                
                # Skip if file doesn't exist
                if not os.path.exists(audio_path):
                    logging.warning(f"File not found: {audio_path}")
                    continue
                
                items.append({
                    "audio_path": audio_path,
                    "text": text,
                    "speaker_id": speaker_id
                })
                
                # Stop if we've reached the maximum number of files
                if max_files is not None and len(items) >= max_files:
                    break
    return items

def prepare_estonian(
    train_list: str,
    val_list: str,
    output_dir: str,
    prefix: str = "et",
    warn_on_wrong_sr: bool = True,
    max_files_per_split: Optional[int] = None
) -> None:
    """
    Prepare Estonian dataset in lhotse format
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    non_16k_files = []
    
    for split, list_path in [("train", train_list), ("val", val_list)]:
        recordings = []
        supervisions = []
        
        logging.info(f"Processing {split} set from {list_path}")
        items = load_estonian_dataset(list_path, max_files=max_files_per_split)
        logging.info(f"Found {len(items)} items in {split} set")
        
        for idx, item in enumerate(items):
            audio_path = item["audio_path"]
            text = item["text"]
            speaker_id = item["speaker_id"]
            
            # Get audio info
            info = sf.info(audio_path)
            duration = info.duration
            
            # Check if the sampling rate is 16kHz
            if warn_on_wrong_sr and info.samplerate != 16000:
                non_16k_files.append((audio_path, info.samplerate))
                
            # Create a recording ID
            recording_id = f"{split}_{idx:08d}"
            
            # Create recording
            recordings.append(
                Recording(
                    id=recording_id,
                    sources=[
                        AudioSource(
                            type="file",
                            channels=[0],
                            source=audio_path
                        )
                    ],
                    sampling_rate=info.samplerate,
                    num_samples=info.frames,
                    duration=duration
                )
            )
            
            # Create supervision
            supervisions.append(
                SupervisionSegment(
                    id=recording_id,
                    recording_id=recording_id,
                    start=0.0,
                    duration=duration,
                    channel=0,
                    text=text,
                    language="et",
                    speaker=speaker_id
                )
            )
        
        # Create manifest sets
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        
        # Save to disk
        recording_set.to_file(output_dir / f"{prefix}_recordings_{split}.jsonl.gz")
        supervision_set.to_file(output_dir / f"{prefix}_supervisions_{split}.jsonl.gz")
        
        logging.info(f"Saved {split} set with {len(recordings)} recordings and {len(supervisions)} supervisions")
    
    # Report files with wrong sampling rate
    if non_16k_files:
        logging.warning(f"Found {len(non_16k_files)} files that are not sampled at 16kHz")
        logging.warning("XLSR models require 16kHz audio. Feature extraction may fail.")
        logging.warning(f"First 5 non-16kHz files: {non_16k_files[:5]}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Estonian dataset in lhotse format")
    parser.add_argument("--train-list", type=str, required=True, 
                        help="Path to the training list file")
    parser.add_argument("--val-list", type=str, required=True, 
                        help="Path to the validation list file")
    parser.add_argument("--output-dir", type=str, default="data/manifests", 
                        help="Output directory for manifests")
    parser.add_argument("--prefix", type=str, default="et", 
                        help="Prefix for manifest files")
    parser.add_argument("--warn-on-wrong-sr", action="store_true", default=True,
                        help="Warn if audio files don't have 16kHz sampling rate")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum number of files to process per split (for testing)")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    prepare_estonian(
        train_list=args.train_list,
        val_list=args.val_list,
        output_dir=args.output_dir,
        prefix=args.prefix,
        warn_on_wrong_sr=args.warn_on_wrong_sr,
        max_files_per_split=args.max_files
    )

if __name__ == "__main__":
    main() 