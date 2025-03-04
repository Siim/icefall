#!/bin/bash

# Evaluation script for XLSR-Transducer Estonian ASR model in streaming mode
# Based on the paper: "XLSR-Transducer: Streaming ASR for Self-Supervised Pretrained Models"

set -e  # Exit on error

# Define directories
WORKSPACE_DIR="/Users/siimhaugas/Desktop/Projects/haugasdev/XLSR-Transducer"
EXP_DIR="$WORKSPACE_DIR/pruned_transducer_stateless7_streaming/exp/xlsr_transducer_estonian"
DATA_DIR="$WORKSPACE_DIR/pruned_transducer_stateless7_streaming/Data"

# Check for checkpoint
CHECKPOINT="$EXP_DIR/checkpoints/best_model.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found at $CHECKPOINT"
    echo "Please specify a valid checkpoint path or train the model first."
    read -p "Enter checkpoint path (or press Enter to exit): " CUSTOM_CHECKPOINT
    if [ -z "$CUSTOM_CHECKPOINT" ]; then
        echo "Exiting."
        exit 1
    fi
    CHECKPOINT="$CUSTOM_CHECKPOINT"
fi

echo "Starting XLSR-Transducer evaluation in streaming mode..."
echo "Data directory: $DATA_DIR"
echo "Experiment directory: $EXP_DIR"
echo "Using checkpoint: $CHECKPOINT"

# Evaluate with different chunk sizes
echo "Evaluating with different chunk sizes..."
python "$WORKSPACE_DIR/pruned_transducer_stateless7_streaming/evaluate_streaming.py" \
    --checkpoint="$CHECKPOINT" \
    --test-list="$DATA_DIR/val_list.txt" \
    --sp-model="$DATA_DIR/lang_bpe_2500/bpe.model" \
    --chunk-sizes="5120,10240,20480,40960" \
    --left-context-chunks=1 \
    --use-attention-sink=true \
    --attention-sink-size=16 \
    --num-test-samples=100 \
    --beam-size=4

# Evaluate with different attention sink sizes
echo "Evaluating with different attention sink sizes..."
for sink_size in 4 8 16 32; do
    echo "Testing attention sink size: $sink_size"
    python "$WORKSPACE_DIR/pruned_transducer_stateless7_streaming/evaluate_streaming.py" \
        --checkpoint="$CHECKPOINT" \
        --test-list="$DATA_DIR/val_list.txt" \
        --sp-model="$DATA_DIR/lang_bpe_2500/bpe.model" \
        --chunk-sizes="10240" \
        --left-context-chunks=1 \
        --use-attention-sink=true \
        --attention-sink-size=$sink_size \
        --num-test-samples=100 \
        --beam-size=4
done

# Evaluate with different left context sizes
echo "Evaluating with different left context sizes..."
for context_size in 0 1 2 4; do
    echo "Testing left context size: $context_size chunks"
    python "$WORKSPACE_DIR/pruned_transducer_stateless7_streaming/evaluate_streaming.py" \
        --checkpoint="$CHECKPOINT" \
        --test-list="$DATA_DIR/val_list.txt" \
        --sp-model="$DATA_DIR/lang_bpe_2500/bpe.model" \
        --chunk-sizes="10240" \
        --left-context-chunks=$context_size \
        --use-attention-sink=true \
        --attention-sink-size=16 \
        --num-test-samples=100 \
        --beam-size=4
done

# Evaluate with attention sink disabled
echo "Evaluating with attention sink disabled..."
python "$WORKSPACE_DIR/pruned_transducer_stateless7_streaming/evaluate_streaming.py" \
    --checkpoint="$CHECKPOINT" \
    --test-list="$DATA_DIR/val_list.txt" \
    --sp-model="$DATA_DIR/lang_bpe_2500/bpe.model" \
    --chunk-sizes="10240" \
    --left-context-chunks=1 \
    --use-attention-sink=false \
    --num-test-samples=100 \
    --beam-size=4

echo "Evaluation completed." 