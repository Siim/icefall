import os
import torch
import torchaudio
from torch.utils.data import Dataset
import logging

class EstonianASRDataset(Dataset):
    def __init__(self, txt_path: str, base_path: str = None, transform=None, sp=None) -> None:
        """
        Args:
            txt_path: Path to the text file containing wav paths and transcripts
            base_path: Base path to prepend to the audio file paths
            transform: Optional transform to apply to the audio
            sp: SentencePiece processor for tokenization
        """
        self.samples = []  # list of tuples (wav_path, transcript)
        self.transform = transform
        self.sp = sp  # Store SentencePiece processor
        
        if self.sp is None:
            raise ValueError("SentencePiece processor (sp) must be provided")
        
        # Duration limits (in samples at 16kHz)
        self.min_samples = 16000  # 1 sec minimum
        self.max_samples = 320000  # 20 sec maximum
        
        # Add logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize processor
        from transformers import Wav2Vec2Processor
        self.processor = Wav2Vec2Processor.from_pretrained("TalTechNLP/xls-r-300m-et")
        
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
                    
                    # Tokenize the transcript
                    try:
                        tokens = self.sp.encode(transcript, out_type=int)
                        if len(tokens) == 0:
                            self.logger.warning(f"Empty tokens for transcript: {transcript}")
                            continue
                        self.samples.append((wav_path, transcript, tokens))
                    except Exception as e:
                        self.logger.warning(f"Failed to tokenize: {transcript} - {str(e)}")
                        continue
                        
                except Exception as e:
                    self.logger.warning(f"Error checking file {wav_path}: {str(e)}")
                    continue
        
        self.logger.info(f"Loaded {len(self.samples)} samples from {txt_path}")
        self.logger.info(f"Filtered out {filtered_short} samples shorter than {self.min_samples/16000:.1f}s")
        self.logger.info(f"Filtered out {filtered_long} samples longer than {self.max_samples/16000:.1f}s")
        self.logger.info(f"Total acceptance rate: {len(self.samples)/total_files*100:.1f}%")
        self.logger.info(f"Using base path: {base_path if base_path else 'None'}")
        # Print first few samples for verification
        for i, (wav_path, transcript, tokens) in enumerate(self.samples[:3]):
            self.logger.info(f"Sample {i}: {wav_path} | {transcript[:50]}... | tokens: {tokens[:10]}...")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        wav_path, transcript, tokens = self.samples[idx]
        
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
            
            # Process using HuggingFace processor (same as xl_exp.py)
            input_values = self.processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values.squeeze(0)  # Remove batch dimension from processor
            
            # Double check length constraints with some margin for resampling
            margin = 100  # Small safety margin
            if input_values.size(0) < self.min_samples - margin:
                raise ValueError(f"Audio too short after processing: {input_values.size(0)} samples")
            if input_values.size(0) > self.max_samples + margin:
                raise ValueError(f"Audio too long after processing: {input_values.size(0)} samples")
            
            # Validate audio duration vs text length
            min_chars_per_second = 3  # Estonian ~4.5 chars/sec avg
            audio_duration = input_values.size(0) / 16000
            if len(transcript) / audio_duration < min_chars_per_second:
                self.logger.warning(f"Suspicious sample {wav_path} - {len(transcript)} chars in {audio_duration:.1f}s")
            
            # Convert tokens to tensor
            tokens_tensor = torch.tensor(tokens, dtype=torch.int32)
            
            # Return a dictionary with processed input values
            return {
                'inputs': input_values.unsqueeze(0),  # Add channel dimension to make (1, time)
                'supervisions': {
                    'num_frames': input_values.size(0),
                    'text': transcript,
                    'audio_paths': wav_path,
                    'tokens': tokens_tensor,
                    'token_lens': torch.tensor([len(tokens)], dtype=torch.int32)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error loading file {wav_path}: {str(e)}")
            # Skip this sample by loading the next one
            next_idx = (idx + 1) % len(self)
            return self[next_idx]


def collate_fn(batch: list, max_duration: float = 10.0) -> dict:
    """Collate function that respects maximum duration limits
    Args:
        batch: List of samples from the dataset
        max_duration: Maximum audio duration in seconds
    Returns:
        Batched data with padded sequences
    """
    # Convert max duration to samples (at 16kHz)
    max_samples = int(max_duration * 16000)
    
    # Filter and truncate sequences
    filtered_batch = []
    for item in batch:
        if item['inputs'].size(1) > max_samples:
            # Truncate sequence
            item['inputs'] = item['inputs'][:, :max_samples]
            item['supervisions']['num_frames'] = torch.tensor([max_samples])
        filtered_batch.append(item)
    
    if not filtered_batch:
        # If all sequences were filtered out, take first sequence and truncate it
        item = batch[0]
        item['inputs'] = item['inputs'][:, :max_samples]
        item['supervisions']['num_frames'] = torch.tensor([max_samples])
        filtered_batch = [item]
    
    # Get maximum length in batch
    max_len = max(item['inputs'].size(1) for item in filtered_batch)
    
    padded_waveforms = []
    padded_tokens = []
    num_frames = []
    token_lens = []
    texts = []
    audio_paths = []
    
    for item in filtered_batch:
        # Handle waveform padding
        waveform = item['inputs']  # shape: (1, time)
        pad_len = max_len - waveform.size(1)
        if pad_len > 0:
            # Pad with zeros while keeping the channel dimension
            pad = torch.zeros(1, pad_len, dtype=waveform.dtype)
            waveform = torch.cat([waveform, pad], dim=1)
        padded_waveforms.append(waveform)
        
        # Handle token padding
        tokens = item['supervisions']['tokens']
        token_pad_len = max(len(t) for t in tokens) - len(tokens)
        if token_pad_len > 0:
            # Pad with zeros
            token_pad = torch.zeros(token_pad_len, dtype=tokens.dtype)
            tokens = torch.cat([tokens, token_pad])
        padded_tokens.append(tokens)
        
        # Collect other supervision data
        num_frames.append(item['supervisions']['num_frames'])
        token_lens.append(item['supervisions']['token_lens'])
        texts.append(item['supervisions']['text'])
        audio_paths.append(item['supervisions']['audio_paths'])
    
    # Stack to get tensors
    inputs = torch.cat(padded_waveforms, dim=0)  # (batch, time)
    inputs = inputs.unsqueeze(-1)  # Add channel dimension at the end
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