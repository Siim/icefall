#!/bin/bash

# Training script for XLSR-Transducer Estonian ASR model
# Based on the paper: "XLSR-Transducer: Streaming ASR for Self-Supervised Pretrained Models"

set -e  # Exit on error

# Define directories
WORKSPACE_DIR="/C/XLSR-Transducer/pruned_transducer_stateless7_streaming"
EXP_DIR="$WORKSPACE_DIR/exp/xlsr_transducer_estonian"
DATA_DIR="$WORKSPACE_DIR/Data"

# Create experiment directory if it doesn't exist
mkdir -p "$EXP_DIR"
mkdir -p "$EXP_DIR/checkpoints"
mkdir -p "$EXP_DIR/tensorboard"

# Check for CUDA availability
if python -c "import torch; print(torch.cuda.is_available());" | grep -q "True"; then
    echo "CUDA is available. Using GPU for training."
else
    echo "WARNING: CUDA is not available. Training will be slow on CPU."
    echo "Do you want to continue with CPU training? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        echo "Training aborted."
        exit 0
    fi
fi

# Install dependencies if needed
if [ -f "$WORKSPACE_DIR/streaming_requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r "$WORKSPACE_DIR/streaming_requirements.txt"
fi

echo "Starting XLSR-Transducer training for Estonian ASR..."
echo "Data directory: $DATA_DIR"
echo "Experiment directory: $EXP_DIR"

# Define BPE model path
BPE_MODEL="$DATA_DIR/lang_bpe_2500/bpe.model"

# Run training with corrected arguments
python "$WORKSPACE_DIR/train.py" \
    --use-xlsr=true \
    --xlsr-model-name="facebook/wav2vec2-large-xlsr-53" \
    --decode-chunk-size=8000 \
    --attention-sink-size=16 \
    --left-context-chunks=1 \
    --num-epochs=20 \
    --lr-epochs=10 \
    --base-lr=3e-5 \
    --train-txt="$DATA_DIR/train_list.txt" \
    --val-txt="$DATA_DIR/val_list.txt" \
    --bpe-model="$BPE_MODEL" \
    --exp-dir="$EXP_DIR" \
    --tensorboard=true \
    --save-every-n=1000 \
    --keep-last-k=5 \
    --seed=42 \
    --num-workers=4 \
    --world-size=1 \
    --dataset="estonian"

echo "Training completed. Models saved to $EXP_DIR" 