#!/bin/bash

# Training script with memory optimizations for streaming phase

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

python pruned_transducer_stateless7_streaming/train.py \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 6 \
  --dataset estonian \
  --use-xlsr True \
  --exp-dir pruned_transducer_stateless7_streaming/exp_et \
  --pretrain-epochs 5 \
  --batch-size 4 \
  --use-fp16 True \
  --audio-base-path "/C/XLSR-Transducer" \
  --train-txt "Data/train_list.txt" \
  --val-txt "Data/val_list.txt" \
  --attention-sink-size 16 \
  "$@" 