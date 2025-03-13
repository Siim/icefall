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
encoder_dim=512  # Encoder output dimension
streaming=0
chunk_size=32
left_context_chunks=1
attention_sink_size=0
et_manifest_dir="data/ssl"
bpe_model="Data/lang_bpe_2500/bpe.model"
use_custom_dataset=0
filter_cuts=0

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
      # We can't pass this directly to train.py, but we'll print a message about it
      echo "Note: XLSR features have dimension $2, this is handled automatically in the encoder"
      shift 2
      ;;
    --encoder-dim)
      encoder_dim=$2
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
    --et-manifest-dir)
      et_manifest_dir=$2
      shift 2
      ;;
    --manifest-dir)
      # For backward compatibility, treat as et-manifest-dir
      et_manifest_dir=$2
      shift 2
      ;;
    --bpe-model)
      bpe_model=$2
      shift 2
      ;;
    --use-custom-dataset)
      use_custom_dataset=$2
      shift 2
      ;;
    --filter-cuts)
      filter_cuts=$2
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
  --bpe-model ${bpe_model} \
  --use-fp16 1 \
  --use-xlsr-encoder ${use_xlsr_encoder} \
  --streaming ${streaming} \
  --encoder-dim ${encoder_dim} \
  --chunk-size ${chunk_size} \
  --left-context-chunks ${left_context_chunks} \
  --attention-sink-size ${attention_sink_size} \
  --et-manifest-dir ${et_manifest_dir} \
  --mini-libri 0 \
  --full-libri 0 \
  --use-custom-dataset ${use_custom_dataset} \
  --filter-cuts ${filter_cuts}

echo "Training completed!" 