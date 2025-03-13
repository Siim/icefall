#!/bin/bash

# Training script for XLSR-Transducer model
# Usage: ./train.sh

set -e  # Exit on error

# Configuration
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# Training settings
num_epochs=10
start_epoch=1 
exp_dir=xlsr_et_exp
world_size=1  # Number of GPUs
seed=42
batch_size=16
max_duration=30  # Maximum utterance duration in seconds

# Train the XLSR-Transducer model
python train.py \
  --world-size ${world_size} \
  --num-epochs ${num_epochs} \
  --start-epoch ${start_epoch} \
  --exp-dir ${exp_dir} \
  --seed ${seed} \
  --max-duration ${max_duration} \
  --bpe-model Data/lang_bpe_2500/bpe.model \
  --valid-subset Data/val_list.txt \
  --use-fp16 1 \
  --feature-dim 1024 \
  --encoder-dim 512 \
  --batch-size ${batch_size}

# Add CUDA debugging environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1  # Use deterministic algorithms where possible

# Command starts here
# ... existing code ... 