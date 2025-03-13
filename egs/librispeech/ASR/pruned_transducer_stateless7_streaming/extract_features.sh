#!/bin/bash

# Script to extract XLSR features from the Estonian dataset
# Usage: ./extract_features.sh [--force] [--single-process]

set -e  # Exit on error

# Check for flags
FORCE_FLAG=""
SINGLE_PROCESS=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_FLAG="--force"
            echo "Will force overwrite existing features"
            shift
            ;;
        --single-process)
            SINGLE_PROCESS=1
            echo "Will use single process for feature extraction"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./extract_features.sh [--force] [--single-process]"
            exit 1
            ;;
    esac
done

# Auto-detect macOS and recommend single process
if [[ "$(uname)" == "Darwin" && $SINGLE_PROCESS -eq 0 ]]; then
    echo "MacOS detected. Parallel processing may cause issues with the wav2vec model."
    echo "Consider using --single-process if you encounter errors."
fi

# Number of parallel jobs to use (adjust based on CPU cores available)
if [[ $SINGLE_PROCESS -eq 1 ]]; then
    NUM_JOBS=1
    echo "Using single process for feature extraction"
else
    NUM_JOBS=10
    echo "Using $NUM_JOBS parallel jobs for feature extraction"
fi

# Create output directories
mkdir -p data/manifests
mkdir -p data/ssl

# Step 1: Convert Estonian dataset to lhotse manifests
echo "Creating manifests from Estonian dataset..."
python local/prepare_estonian.py \
    --train-list Data/train_list.txt \
    --val-list Data/val_list.txt \
    --output-dir data/manifests \
    --prefix et

# Step 2: Extract features for training set
echo "Extracting features for training set..."
python compute_xlsr_features.py \
    --manifest-dir data/manifests \
    --output-dir data/ssl \
    --ssl-model wav2vec2 \
    --prefix et \
    --dataset-parts train \
    --num-jobs $NUM_JOBS \
    $FORCE_FLAG

# Step 3: Extract features for validation set
echo "Extracting features for validation set..."
python compute_xlsr_features.py \
    --manifest-dir data/manifests \
    --output-dir data/ssl \
    --ssl-model wav2vec2 \
    --prefix et \
    --dataset-parts val \
    --num-jobs $NUM_JOBS \
    $FORCE_FLAG

echo "Feature extraction complete!"
echo "The features are saved in data/ssl" 