#!/bin/bash

# Training script for XLSR-Transducer model
# Usage: ./train.sh [options]

set -e  # Exit on error

# Configuration
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# Default settings
num_epochs=10
start_epoch=1 
exp_dir=exp/xlsr_transducer
world_size=1  # Number of GPUs
seed=42
max_duration=30  # Maximum utterance duration in seconds
use_xlsr_encoder=0
feature_dim=80  # Default: 80 for fbank, should be 768 for XLSR
streaming=0
chunk_size=32
left_context_chunks=1
attention_sink_size=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-epochs)
      num_epochs=$2
      shift 2
      ;;
    --start-epoch)
      start_epoch=$2
      shift 2
      ;;
    --exp-dir)
      exp_dir=$2
      shift 2
      ;;
    --use-xlsr-encoder)
      use_xlsr_encoder=$2
      shift 2
      ;;
    --feature-dim)
      feature_dim=$2
      shift 2
      ;;
    --streaming)
      streaming=$2
      shift 2
      ;;
    --chunk-size)
      chunk_size=$2
      shift 2
      ;;
    --left-context-chunks)
      left_context_chunks=$2
      shift 2
      ;;
    --attention-sink-size)
      attention_sink_size=$2
      shift 2
      ;;
    --stage)
      stage=$2
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create experiment directory
mkdir -p ${exp_dir}

# Add CUDA debugging environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1  # Use deterministic algorithms where possible

# Train the XLSR-Transducer model
python train.py \
  --world-size ${world_size} \
  --num-epochs ${num_epochs} \
  --start-epoch ${start_epoch} \
  --exp-dir ${exp_dir} \
  --seed ${seed} \
  --max-duration ${max_duration} \
  --bpe-model Data/lang_bpe_2500/bpe.model \
  --use-fp16 1 \
  --use-xlsr-encoder ${use_xlsr_encoder} \
  --streaming ${streaming} \
  --feature-dim ${feature_dim} \
  --chunk-size ${chunk_size} \
  --left-context-chunks ${left_context_chunks} \
  --attention-sink-size ${attention_sink_size}

echo "Training completed!" 