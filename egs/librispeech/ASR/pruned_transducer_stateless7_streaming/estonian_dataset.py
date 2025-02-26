import os
import torch
import torchaudio
from torch.utils.data import Dataset
import logging
from lhotse import CutSet, Recording, SupervisionSegment, Cut
from lhotse.audio import AudioSource
from lhotse.cut import MixedCut
from typing import Dict, List, Optional, Union, Any
import numpy as np
import random
from pathlib import Path

class EstonianASRDataset(Dataset):
    def __init__(self, txt_path: str, base_path: str = None, transform=None, sp=None, max_duration: float = 10.0) -> None:
        """
        Args:
            txt_path: Path to the text file containing wav paths and transcripts
            base_path: Base path to prepend to the audio file paths
            transform: Optional transform to apply to the audio
            sp: SentencePieceProcessor for tokenization
            max_duration: Maximum audio duration in seconds
        """
        self.samples = []  # list of tuples (wav_path, transcript)
        self.transform = transform
        self.sp = sp  # Store SentencePieceProcessor
        
        if self.sp is None:
            raise ValueError("SentencePieceProcessor (sp) must be provided")
        
        # Duration limits (in samples at 16kHz)
        self.min_samples = 16000  # 1 sec minimum
        self.max_samples = int(max_duration * 16000)  # Convert max_duration to samples
        
        # Add logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure base_path ends with a slash if provided
        if base_path:
            base_path = os.path.join(base_path, '')
        
        filtered_short = 0
        filtered_long = 0
        total_files = 0
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: wav_path|transcript|metadata
                parts = line.split('|')
                if len(parts) < 2:  # Need at least path and transcript
                    continue
                wav_path, transcript = parts[0], parts[1]
                total_files += 1
                
                # Skip paths that don't exist
                full_path = wav_path if not base_path else os.path.join(base_path, wav_path)
                if not os.path.exists(full_path):
                    self.logger.warning(f"File not found: {full_path}")
                    continue
                
                try:
                    # Check the audio duration
                    info = torchaudio.info(full_path)
                    if info.num_frames < self.min_samples:
                        filtered_short += 1
                        continue
                    if info.num_frames > self.max_samples:
                        filtered_long += 1
                        continue
                    
                    # Add the sample
                    self.samples.append((full_path, transcript))
                
                except Exception as e:
                    self.logger.error(f"Error loading {full_path}: {e}")
                    continue
        
        self.logger.info(f"Loaded {len(self.samples)} files from {txt_path}")
        self.logger.info(f"Filtered {filtered_short} files shorter than {self.min_samples/16000}s")
        self.logger.info(f"Filtered {filtered_long} files longer than {self.max_samples/16000}s")
        self.logger.info(f"Total files in text: {total_files}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        wav_path, transcript = self.samples[idx]
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # Ensure mono audio
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Apply transform if available
            if self.transform:
                waveform = self.transform(waveform)
            
            # Tokenize text if SentencePieceProcessor is available
            if self.sp:
                tokens = self.sp.encode(transcript, out_type=int)
                token_lens = torch.tensor([len(tokens)], dtype=torch.int32)
            else:
                tokens = []
                token_lens = torch.tensor([0], dtype=torch.int32)
            
            # Create supervision information
            supervisions = {
                'num_frames': torch.tensor([waveform.size(1)], dtype=torch.int32),
                'text': transcript,
                'audio_paths': wav_path,
                'tokens': tokens,
                'token_lens': token_lens
            }
            
            return {
                'inputs': waveform,  # Shape: (1, time)
                'supervisions': supervisions
            }
            
        except Exception as e:
            self.logger.error(f"Error loading {wav_path}: {e}")
            # Return a minimal filler item - will be filtered in collate_fn
            return {
                'inputs': torch.zeros(1, self.min_samples),
                'supervisions': {
                    'num_frames': torch.tensor([0], dtype=torch.int32),
                    'text': '',
                    'audio_paths': wav_path,
                    'tokens': [],
                    'token_lens': torch.tensor([0], dtype=torch.int32)
                }
            }

def collate_fn(batch: list, max_duration: float = 10.0) -> dict:
    """
    Custom collate function for the Estonian ASR dataset.
    
    Args:
        batch: List of samples from the dataset
        max_duration: Maximum audio duration in seconds
        
    Returns:
        Batched dictionary with padded inputs and supervisions
    """
    # Filter out samples with zero frames or other issues
    filtered_batch = []
    max_samples = int(max_duration * 16000)  # Maximum samples at 16kHz
    
    for item in batch:
        num_frames = item['supervisions']['num_frames']
        if num_frames.item() <= 0:
            continue
        if num_frames.item() > max_samples:
            # Truncate long samples
            item['inputs'] = item['inputs'][:, :max_samples]
            item['supervisions']['num_frames'] = torch.tensor([max_samples])
        filtered_batch.append(item)
    
    # If all samples were filtered out, return a minimal batch
    if len(filtered_batch) == 0:
        item = batch[0].copy()  # Use the first sample as a template
        item['inputs'] = torch.zeros(1, max_samples)
        item['supervisions']['num_frames'] = torch.tensor([max_samples])
        filtered_batch = [item]
    
    # Get maximum length in batch
    max_len = max(item['inputs'].size(1) for item in filtered_batch)
    
    # Calculate maximum token length across all samples
    max_token_len = 0
    for item in filtered_batch:
        tokens = item['supervisions']['tokens']
        if isinstance(tokens, torch.Tensor):
            if tokens.dim() == 0:  # Scalar tensor
                token_len = 1
            else:
                token_len = tokens.size(0)
        else:
            token_len = len(tokens)
        max_token_len = max(max_token_len, token_len)
    
    padded_waveforms = []
    padded_tokens = []
    num_frames = []
    token_lens = []
    texts = []
    audio_paths = []
    
    for item in filtered_batch:
        # Handle waveform padding - raw audio should be shape (1, time)
        waveform = item['inputs']  # shape: (1, time)
        pad_len = max_len - waveform.size(1)
        if pad_len > 0:
            # Pad with zeros while keeping the batch dimension
            pad = torch.zeros(1, pad_len, dtype=waveform.dtype, device=waveform.device)
            waveform = torch.cat([waveform, pad], dim=1)
        padded_waveforms.append(waveform)
        
        # Handle token padding
        tokens = item['supervisions']['tokens']
        if isinstance(tokens, torch.Tensor):
            if tokens.dim() == 0:  # Scalar tensor
                tokens = tokens.unsqueeze(0)
                token_len = 1
            else:
                token_len = tokens.size(0)
        else:
            token_len = len(tokens)
            tokens = torch.tensor(tokens)
        
        # Calculate how much padding we need to add
        token_pad_len = max_token_len - token_len
        
        if token_pad_len > 0:
            # Pad with zeros
            token_pad = torch.zeros(token_pad_len, dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, token_pad])
        padded_tokens.append(tokens)
        
        # Collect other supervision data
        num_frames.append(item['supervisions']['num_frames'])
        token_lens.append(item['supervisions']['token_lens'])
        texts.append(item['supervisions']['text'])
        audio_paths.append(item['supervisions']['audio_paths'])
    
    # Stack to get tensors - for raw audio, shape should be (batch, time)
    inputs = torch.cat(padded_waveforms, dim=0)  # (batch, time)
    tokens = torch.stack(padded_tokens)  # (batch, max_token_len)
    token_lens = torch.cat(token_lens)  # (batch,)
    
    supervisions = {
        'num_frames': torch.tensor(num_frames, dtype=torch.int32),
        'text': texts,
        'audio_paths': audio_paths,
        'tokens': tokens,
        'token_lens': token_lens
    }
    
    return {'inputs': inputs, 'supervisions': supervisions}


class EstonianCutSet:
    """Wrapper class that adapts EstonianASRDataset to look like a CutSet for the Icefall pipeline."""
    
    def __init__(self, cuts: List[dict] = None):
        """
        Args:
            cuts: List of dictionaries representing cuts
        """
        self.cuts = cuts if cuts is not None else []
        
    def __getitem__(self, idx: Union[int, List[int]]) -> Dict:
        """
        Get a single cut or a batch of cuts.
        
        Args:
            idx: Index or list of indices
            
        Returns:
            Dictionary with batch of inputs and supervisions
        """
        if isinstance(idx, int):
            return self.cuts[idx]
        elif isinstance(idx, list):
            selected_cuts = [self.cuts[i] for i in idx]
            return collate_fn(selected_cuts)
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")
    
    def __len__(self) -> int:
        return len(self.cuts)


class EstonianDataModule:
    """Data module for loading Estonian speech data."""
    
    def __init__(self, train_list: str, val_list: str, audio_base_path: str = None, sp=None):
        """
        Args:
            train_list: Path to train list file
            val_list: Path to validation list file
            audio_base_path: Base path for audio files
            sp: SentencePieceProcessor for tokenization
        """
        self.train_list = train_list
        self.val_list = val_list
        self.audio_base_path = audio_base_path
        self.sp = sp
        self.logger = logging.getLogger(__name__)
        
        self.train_dataset = None
        self.val_dataset = None
        
    def train_cuts(self) -> EstonianCutSet:
        """Load and return training data."""
        if self.train_dataset is None:
            self.logger.info(f"Loading training data from {self.train_list}")
            self.train_dataset = EstonianASRDataset(
                txt_path=self.train_list,
                base_path=self.audio_base_path,
                sp=self.sp
            )
            
        # Convert dataset items to cut-like format
        cuts = []
        for i in range(len(self.train_dataset)):
            cuts.append(self.train_dataset[i])
            
        self.logger.info(f"Loaded {len(cuts)} training samples")
        return EstonianCutSet(cuts)
    
    def val_cuts(self) -> EstonianCutSet:
        """Load and return validation data."""
        if self.val_dataset is None:
            self.logger.info(f"Loading validation data from {self.val_list}")
            self.val_dataset = EstonianASRDataset(
                txt_path=self.val_list,
                base_path=self.audio_base_path,
                sp=self.sp
            )
            
        # Convert dataset items to cut-like format
        cuts = []
        for i in range(len(self.val_dataset)):
            cuts.append(self.val_dataset[i])
            
        self.logger.info(f"Loaded {len(cuts)} validation samples")
        return EstonianCutSet(cuts) 