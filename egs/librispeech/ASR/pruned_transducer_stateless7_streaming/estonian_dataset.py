import os
import torch
import torchaudio
from torch.utils.data import Dataset
import logging

class EstonianASRDataset(Dataset):
    def __init__(self, txt_path: str, base_path: str = None, transform=None) -> None:
        """
        Args:
            txt_path: Path to the text file containing wav paths and transcripts
            base_path: Base path to prepend to the audio file paths
            transform: Optional transform to apply to the audio
        """
        self.samples = []  # list of tuples (wav_path, transcript)
        self.transform = transform
        
        # Duration limits (in samples at 16kHz)
        self.min_samples = 16000  # 1 sec minimum
        self.max_samples = 320000  # 20 sec maximum
        
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
                
                # If base_path is provided, join it with wav_path
                if base_path:
                    wav_path = os.path.join(base_path, wav_path)
                else:
                    # If no base_path, treat wav_path as absolute or relative to current directory
                    if not os.path.isabs(wav_path):
                        wav_path = os.path.abspath(wav_path)
                
                try:
                    # Get audio info without loading the whole file
                    info = torchaudio.info(wav_path)
                    
                    # Calculate expected samples after resampling to 16kHz
                    expected_samples = int(info.num_frames * (16000 / info.sample_rate))
                    
                    # Add some margin for resampling artifacts
                    margin = 100  # Small safety margin
                    if expected_samples - margin < self.min_samples:
                        filtered_short += 1
                        self.logger.debug(f"File too short: {wav_path} ({expected_samples} samples)")
                        continue
                    if expected_samples + margin > self.max_samples:
                        filtered_long += 1
                        self.logger.debug(f"File too long: {wav_path} ({expected_samples} samples)")
                        continue
                    
                    self.samples.append((wav_path, transcript))
                except Exception as e:
                    self.logger.warning(f"Error checking file {wav_path}: {str(e)}")
                    continue
        
        self.logger.info(f"Loaded {len(self.samples)} samples from {txt_path}")
        self.logger.info(f"Filtered out {filtered_short} samples shorter than {self.min_samples/16000:.1f}s")
        self.logger.info(f"Filtered out {filtered_long} samples longer than {self.max_samples/16000:.1f}s")
        self.logger.info(f"Total acceptance rate: {len(self.samples)/total_files*100:.1f}%")
        self.logger.info(f"Using base path: {base_path if base_path else 'None'}")
        # Print first few samples for verification
        for i, (wav_path, transcript) in enumerate(self.samples[:3]):
            self.logger.info(f"Sample {i}: {wav_path} | {transcript[:50]}...")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        wav_path, transcript = self.samples[idx]
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Normalize to [-1, 1]
            if waveform.dtype == torch.int16:
                waveform = waveform.float() / 32768.0
            elif waveform.dtype == torch.int32:
                waveform = waveform.float() / 2147483648.0
            elif waveform.dtype == torch.uint8:
                waveform = (waveform.float() - 128) / 128.0
            
            # Ensure values are clamped to [-1, 1]
            waveform = torch.clamp(waveform, min=-1.0, max=1.0)
            
            # Double check length constraints with some margin for resampling
            margin = 100  # Small safety margin
            if waveform.size(1) < self.min_samples - margin:
                raise ValueError(f"Audio too short after processing: {waveform.size(1)} samples")
            if waveform.size(1) > self.max_samples + margin:
                raise ValueError(f"Audio too long after processing: {waveform.size(1)} samples")
            
            # Validate audio duration vs text length
            min_chars_per_second = 3  # Estonian ~4.5 chars/sec avg
            audio_duration = waveform.size(1) / 16000
            if len(transcript) / audio_duration < min_chars_per_second:
                self.logger.warning(f"Suspicious sample {wav_path} - {len(transcript)} chars in {audio_duration:.1f}s")
            
            # Return a dictionary with raw waveform and supervision
            return {
                'inputs': waveform,  # shape: (1, time)
                'supervisions': {
                    'num_frames': waveform.size(1),
                    'text': transcript,
                    'audio_paths': wav_path
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error loading file {wav_path}: {str(e)}")
            # Skip this sample by loading the next one
            next_idx = (idx + 1) % len(self)
            return self[next_idx]


def collate_fn(batch: list) -> dict:
    # Collate function pads raw waveforms along the time dimension
    max_len = max(item['inputs'].size(1) for item in batch)
    padded_waveforms = []
    num_frames = []
    texts = []
    audio_paths = []
    
    for item in batch:
        waveform = item['inputs']  # shape: (1, time)
        pad_len = max_len - waveform.size(1)
        if pad_len > 0:
            # Pad with zeros while keeping the channel dimension
            pad = torch.zeros(1, pad_len, dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=1)
        padded_waveforms.append(waveform)
        num_frames.append(item['supervisions']['num_frames'])
        texts.append(item['supervisions']['text'])
        audio_paths.append(item['supervisions']['audio_paths'])
    
    # Stack to get a tensor of shape (batch, channel, time)
    inputs = torch.cat(padded_waveforms, dim=0)  # (batch, time)
    # Reshape to (batch, time, channel)
    inputs = inputs.unsqueeze(-1)  # Add channel dimension at the end
    
    # Double check normalization
    if torch.any(inputs > 1.0) or torch.any(inputs < -1.0):
        inputs = torch.clamp(inputs, min=-1.0, max=1.0)
    
    supervisions = {
        'num_frames': torch.tensor(num_frames, dtype=torch.int32),
        'text': texts,
        'audio_paths': audio_paths
    }
    
    return {'inputs': inputs, 'supervisions': supervisions} 