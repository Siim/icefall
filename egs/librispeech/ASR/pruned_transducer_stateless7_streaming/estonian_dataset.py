# Copyright    2024                      (authors: Siim Haugas)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import torch
import torchaudio
import sentencepiece as spm
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)

class EstonianDataset(Dataset):
    """
    Dataset for Estonian speech recognition using XLSR.
    
    Args:
        data_file: Path to text file containing audio paths and transcripts.
        sp_model: Path to SentencePiece model file for tokenization.
        is_training: Whether this dataset is used for training.
        sample_rate: Target sample rate (should be 16kHz for XLSR).
        max_duration: Maximum allowed duration in seconds.
        min_duration: Minimum allowed duration in seconds.
    """
    
    def __init__(
        self,
        data_file: str,
        sp_model: str,
        is_training: bool = True,
        sample_rate: int = 16000,
        max_duration: float = 20.0,
        min_duration: float = 0.5,
    ):
        super().__init__()
        
        self.data_file = data_file
        self.is_training = is_training
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        
        # Load the data file
        self.items = self._load_data()
        
        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model)
        
        logging.info(f"Loaded {len(self.items)} utterances from {data_file}")
    
    def _load_data(self) -> List[Dict[str, str]]:
        """Load audio paths and transcriptions from data file."""
        items = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    audio_path, text = parts[0], parts[1]
                    
                    # Check if file exists
                    if not os.path.exists(audio_path):
                        logging.warning(f"File not found: {audio_path}, skipping")
                        continue
                    
                    items.append({
                        "audio_path": audio_path,
                        "text": text
                    })
        
        return items
    
    def __len__(self) -> int:
        return len(self.items)
    
    def _convert_to_16k(self, audio_path: str) -> str:
        """Convert audio file to 16kHz if needed."""
        # Replace _24k.wav with _16k.wav in filename
        output_path = audio_path.replace("_24k.wav", "_16k.wav")
        
        # If 16kHz version already exists, return it
        if os.path.exists(output_path):
            return output_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load and resample
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            
            # Save as 16kHz file
            torchaudio.save(output_path, waveform, 16000)
        
        return output_path
    
    def _verify_audio(self, waveform: torch.Tensor, sample_rate: int, file_path: str = None) -> torch.Tensor:
        """Verify that audio meets XLSR requirements."""
        # Check sample rate
        if sample_rate != 16000:
            raise ValueError(f"Sample rate must be 16kHz, got {sample_rate}Hz")
        
        # Check that audio is mono
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            logging.warning(f"Audio should be mono, got {waveform.shape[0]} channels for {file_path}. Converting to mono.")
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Check normalization range
        min_val, max_val = waveform.min().item(), waveform.max().item()
        if min_val < -1.01 or max_val > 1.01:  # Allow small floating point errors
            logging.warning(f"Audio values outside range [-1, 1]: min={min_val:.4f}, max={max_val:.4f} for {file_path}")
            # Normalize to [-1, 1]
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
        
        # Log zero or constant audio
        if waveform.std() < 1e-6:
            logging.warning(f"Audio has very low variance (possibly silent or DC): std={waveform.std():.8f} for {file_path}")
            
        return waveform
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a dataset item by index."""
        item = self.items[idx]
        audio_path = item["audio_path"]
        text = item["text"]
        
        # Convert to 16kHz if needed
        if "_24k.wav" in audio_path:
            audio_path = self._convert_to_16k(audio_path)
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            logging.error(f"Error loading audio {audio_path}: {e}")
            # Return a dummy item to prevent training crash
            return {
                "inputs": torch.zeros(1, 16000),  # 1 second of silence
                "input_lens": torch.tensor([16000]),
                "text": torch.tensor([]),
                "text_lens": torch.tensor([0]),
            }
        
        # Resample if needed (should already be 16kHz at this point)
        if sample_rate != self.sample_rate:
            logging.info(f"Resampling {audio_path} from {sample_rate}Hz to {self.sample_rate}Hz")
            resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.sample_rate
        
        # Verify and normalize audio
        waveform = self._verify_audio(waveform, sample_rate, audio_path)
        
        # Check duration constraints
        duration = waveform.size(1) / sample_rate
        if duration > self.max_duration:
            logging.warning(f"Audio {audio_path} duration ({duration:.2f}s) exceeds max_duration ({self.max_duration}s), trimming")
            waveform = waveform[:, :int(self.max_duration * sample_rate)]
        elif duration < self.min_duration:
            logging.warning(f"Audio {audio_path} duration ({duration:.2f}s) below min_duration ({self.min_duration}s)")
            # Pad with silence if too short
            padding = int((self.min_duration - duration) * sample_rate)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Tokenize text
        tokens = self.sp.encode(text)
        tokens_tensor = torch.tensor(tokens, dtype=torch.int64)
        
        return {
            "inputs": waveform,  # Shape: [1, time]
            "input_lens": torch.tensor([waveform.size(1)]),
            "text": tokens_tensor,  # Shape: [U]
            "text_lens": torch.tensor([len(tokens)]),
            "supervisions": {
                "text": text,
                "audio_path": audio_path
            }
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch for training."""
        # Remove any None entries from failed loads
        batch = [item for item in batch if item is not None]
        
        if len(batch) == 0:
            logging.warning("Empty batch after filtering")
            return None
        
        # Get max lengths
        max_audio_len = max(item["input_lens"].item() for item in batch)
        max_text_len = max(item["text_lens"].item() for item in batch)
        
        # Prepare tensors
        inputs = torch.zeros(len(batch), 1, max_audio_len)
        input_lens = torch.zeros(len(batch), dtype=torch.int64)
        texts = torch.zeros(len(batch), max_text_len, dtype=torch.int64)
        text_lens = torch.zeros(len(batch), dtype=torch.int64)
        
        # Fill tensors
        for i, item in enumerate(batch):
            audio_len = item["input_lens"].item()
            text_len = item["text_lens"].item()
            
            inputs[i, :, :audio_len] = item["inputs"]
            input_lens[i] = audio_len
            
            if text_len > 0:
                texts[i, :text_len] = item["text"]
            text_lens[i] = text_len
        
        # Convert supervision data
        supervisions = [item["supervisions"] for item in batch]
        
        return {
            "inputs": inputs,
            "input_lens": input_lens,
            "text": texts,
            "text_lens": text_lens,
            "supervisions": supervisions
        } 