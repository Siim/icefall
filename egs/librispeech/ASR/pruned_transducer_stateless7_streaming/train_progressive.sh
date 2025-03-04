#!/bin/bash

# Memory-optimized progressive training script
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8

# Stage 1: Pre-training with smaller batch and warmup
python pruned_transducer_stateless7_streaming/train.py \
  --world-size 1 \
  --num-epochs 5 \
  --dataset estonian \
  --use-xlsr True \
  --exp-dir pruned_transducer_stateless7_streaming/exp_et_small \
  --pretrain-epochs 5 \
  --batch-size 4 \
  --use-bf16 True \
  --audio-base-path "/C/XLSR-Transducer" \
  --train-txt "Data/train_list.txt" \
  --val-txt "Data/val_list.txt" \
  --grad-accum-steps 4 \
  --strict-length-limit True \
  --attention-sink-size 16 \
  --pre-train-lr 0.00005 \
  --warm-step 3000 \
  --encoder-lr 1e-6 \
  --grad-clip 3.0 \
  --freeze-encoder-layers 12 \
  "$@"

# Stage 2: Streaming phase with multi-chunk training - tiny chunks first (epochs 6-8)
python pruned_transducer_stateless7_streaming/train.py \
  --world-size 1 \
  --num-epochs 8 \
  --start-epoch 6 \
  --dataset estonian \
  --use-xlsr True \
  --exp-dir pruned_transducer_stateless7_streaming/exp_et_small \
  --pretrain-epochs 5 \
  --batch-size 2 \
  --use-bf16 True \
  --decode-chunk-size 2560 \
  --grad-accum-steps 8 \
  --strict-length-limit True \
  --attention-sink-size 16 \
  --multi-chunk-training True \
  --audio-base-path "/C/XLSR-Transducer" \
  --train-txt "Data/train_list.txt" \
  --val-txt "Data/val_list.txt" \
  "$@"

# Stage 3: Streaming phase with multi-chunk training (epochs 9-15)
python pruned_transducer_stateless7_streaming/train.py \
  --world-size 1 \
  --num-epochs 15 \
  --start-epoch 9 \
  --dataset estonian \
  --use-xlsr True \
  --exp-dir pruned_transducer_stateless7_streaming/exp_et_small \
  --pretrain-epochs 5 \
  --batch-size 2 \
  --use-bf16 True \
  --decode-chunk-size 5120 \
  --grad-accum-steps 8 \
  --strict-length-limit True \
  --attention-sink-size 16 \
  --multi-chunk-training True \
  --chunk-size-min 2560 \
  --chunk-size-max 10240 \
  --audio-base-path "/C/XLSR-Transducer" \
  --train-txt "Data/train_list.txt" \
  --val-txt "Data/val_list.txt" \
  "$@" 