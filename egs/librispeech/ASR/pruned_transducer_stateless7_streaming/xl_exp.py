#!/usr/bin/env python3

import argparse
import torch
import torchaudio
import logging
import numpy as np
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/wav2vec2-xls-r-300m",
        help="Model name or path",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to audio file",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="Number of frames to show",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
        help="Number of features to show per frame",
    )
    return parser.parse_args()

def load_audio(audio_path: str):
    """Load and preprocess audio file."""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, sample_rate

def main():
    args = get_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    
    # Load and preprocess audio
    waveform, sample_rate = load_audio(args.audio_path)
    logging.info(f"Loaded audio: shape={waveform.shape}, sample_rate={sample_rate}")
    
    # Try different pre-trained models in order of preference
    model_names = [
        "TalTechNLP/xls-r-300m-et",  # TalTechNLP Estonian XLSR
        "tartuNLP/wav2vec2-large-xls-r-300m-et",  # Estonian from TartuNLP
        "tartuNLP/wav2vec2-large-xls-r-300m-v2-et",  # Another Estonian variant
        "facebook/wav2vec2-xls-r-300m",  # Base multilingual
    ]
    
    for model_name in model_names:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            logging.info(f"Successfully loaded {model_name}")
            break
        except:
            logging.info(f"Could not load {model_name}, trying next...")
    else:
        raise ValueError("Could not load any pre-trained model")
    
    # Set to eval mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Process audio input
    input_values = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_values
    input_values = input_values.to(device)
    
    # Process audio
    with torch.no_grad():
        # Get model outputs
        outputs = model(input_values, output_hidden_states=True)
        
        # Get text prediction
        logits = outputs.logits[0]  # Remove batch dimension
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(pred_ids)
        
        logging.info("\nTranscription:")
        logging.info("-" * 50)
        logging.info(transcription)
        logging.info("-" * 50)
        
        # Get feature statistics from last hidden state
        last_hidden = outputs.hidden_states[-1]
        logging.info("\nOutput Features:")
        logging.info("-" * 50)
        logging.info(f"Shape: {last_hidden.shape}")
        logging.info(f"Mean: {last_hidden.mean().item():.4f}")
        logging.info(f"Std: {last_hidden.std().item():.4f}")
        logging.info(f"Min: {last_hidden.min().item():.4f}")
        logging.info(f"Max: {last_hidden.max().item():.4f}")
        logging.info("-" * 50)
        
        # Show sample of actual features
        logging.info("\nFeature samples:")
        logging.info("-" * 50)
        
        # Get features as numpy for easier handling
        features = last_hidden[0].cpu().numpy()  # Shape: [frames, features]
        
        # Show features for first few frames
        for frame_idx in range(min(args.num_frames, features.shape[0])):
            frame = features[frame_idx]
            
            # Get top features by absolute magnitude
            abs_frame = np.abs(frame)
            top_indices = np.argsort(abs_frame)[-args.num_features:][::-1]
            
            logging.info(f"\nFrame {frame_idx}:")
            logging.info("Feature_idx: value (abs_value)")
            for feat_idx in top_indices:
                value = frame[feat_idx]
                abs_value = abs_frame[feat_idx]
                logging.info(f"  {feat_idx:4d}: {value:7.4f} ({abs_value:.4f})")
        
        logging.info("-" * 50)

if __name__ == "__main__":
    main() 