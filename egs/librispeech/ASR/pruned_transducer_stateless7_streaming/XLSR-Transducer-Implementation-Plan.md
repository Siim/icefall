# XLSR-Transducer Implementation Plan

## Overview

This document outlines the implementation plan for the XLSR-Transducer model as described in the paper "XLSR-Transducer: Streaming ASR for Self-Supervised Pretrained Models." Based on our analysis, we'll use a **feature extraction approach** rather than on-the-fly processing for better memory efficiency, training speed, and compatibility with the existing icefall infrastructure.

## Current Status

We currently have:
- A working Zipformer-based Transducer model in the icefall framework
- An Estonian dataset with train/val splits in the format: `file_path|transcription|speaker_id_int`
- BPE tokenizer with 2500 vocabulary size in `Data/lang_bpe_2500`

## Implementation Progress

### Completed:
- [x] Created the XLSR encoder implementation (both non-streaming and streaming versions)
- [x] Implemented the feature extraction script to pre-compute XLSR features
- [x] Updated the training script to use our XLSR encoder
- [x] Added command-line parameters for XLSR-specific configurations
- [x] Created a shell script to extract features from the Estonian dataset
- [x] Fixed AudioSource import issue in prepare_estonian.py
- [x] Implemented automatic resampling to 16kHz for XLSR model compatibility
- [x] Created a test script for verifying extraction on a small subset
- [x] Added parallel processing to feature extraction for improved speed
- [x] Fixed pickling issues on macOS by implementing single-process fallback

### In Progress:
- [ ] Running feature extraction on the full Estonian dataset
- [ ] Testing the XLSR-Transducer model with extracted features

### To Do:
- [ ] Train the non-streaming XLSR-Transducer model
- [ ] Evaluate the model on the validation set
- [ ] Implement and test the streaming capabilities
- [ ] Experiment with different chunk sizes and attention sink configurations
- [ ] Create a comprehensive evaluation pipeline

## Implementation Strategy

We've implemented the XLSR-Transducer in three main stages:

1. **Feature Extraction Stage**: Extract features from the XLSR-53 model
2. **Non-Streaming Transducer Stage**: Train a transducer model using the extracted features
3. **Streaming Implementation Stage**: Implement streaming capabilities and attention sinks

## Phase 1: Feature Extraction

We've created a feature extraction pipeline with:
- A script to convert Estonian dataset to lhotse manifests
- A feature extraction script that processes audio through XLSR model
- Automatic resampling to 16kHz as required by XLSR models
- Parallel processing to speed up extraction (using `num_jobs` parameter)
- Single-process fallback for macOS to avoid pickling errors
- Options for testing with small subsets

The extraction script can be run using:
```bash
# For full dataset
./extract_features.sh                     # Using parallel processing (3 jobs)
./extract_features.sh --single-process    # Using single process (for macOS)

# For small test subset
./extract_features_test.sh                # Using parallel processing (2 jobs)
./extract_features_test.sh --single-process  # Using single process (for macOS)
```

## Phase 2: XLSR-Transducer Encoder Implementation

We've implemented two encoder classes:
1. `XLSREncoder`: A basic encoder that uses pre-extracted XLSR features
2. `StreamingXLSREncoder`: A streaming version with chunked attention and attention sinks

These encoders implement the `EncoderInterface` required by the transducer model and can be used as drop-in replacements for the Zipformer encoder.

## Phase 3: Integrating with the Transducer Framework

We've updated the training script to use our XLSR encoder:
- Modified the `get_encoder_model` function to use our encoder based on a flag
- Added command-line parameters for XLSR-specific configurations
- Updated the default parameters in `get_params` function

## Next Steps

1. **Complete Feature Extraction**:
   ```bash
   ./extract_features.sh
   ```

2. **Train the Non-Streaming Model**:
   ```bash
   ./train.sh --use-xlsr-encoder 1 --feature-dim 768
   ```

3. **Train the Streaming Model**:
   ```bash
   ./train.sh --use-xlsr-encoder 1 --streaming 1 --feature-dim 768 --chunk-size 32 --left-context-chunks 1
   ```

4. **Evaluate Different Configurations**:
   - Test different chunk sizes: 16, 32, 64, 128
   - Test different left context sizes: 1, 2, 4
   - Test different attention sink sizes: 0, 4, 8, 16

## Implementation Timeline and Progress Tracking

### Week 1: Feature Extraction and Basic Implementation
- [x] Set up the development environment
- [x] Implement the feature extraction script
- [x] Implement the basic XLSR encoder
- [x] Create parallel processing for faster extraction

### Week 2: Non-Streaming Model Training
- [x] Integrate the XLSR encoder with the transducer framework
- [x] Create the feature dataloader
- [ ] Train the non-streaming XLSR-Transducer model
- [ ] Evaluate the model on the validation set

### Week 3: Streaming Implementation
- [x] Implement the streaming encoder with chunked attention
- [x] Add attention sink functionality
- [ ] Test different chunk sizes and attention sink configurations
- [ ] Visualize attention patterns

### Week 4: Evaluation and Fine-tuning
- [ ] Evaluate and compare different streaming configurations
- [ ] Fine-tune the model to improve performance
- [ ] Document the results and learnings
- [ ] Create a demo script for real-time streaming inference

## Requirements

This implementation requires:
1. PyTorch >= 1.12
2. transformers >= 4.20.0
3. k2 >= 1.24.4
4. sentencepiece >= 0.1.96
5. torchaudio (for audio processing)
6. soundfile (for reading audio files)
7. numpy (for feature storage)
8. icefall (for the transducer framework)
9. lhotse (for data preparation and feature extraction)
10. s3prl (for XLSR feature extraction)

## Conclusion

We've made significant progress in implementing the XLSR-Transducer model for the Estonian dataset. The next steps involve completing the feature extraction, training the model, and evaluating different streaming configurations to find the optimal trade-off between accuracy and latency. 