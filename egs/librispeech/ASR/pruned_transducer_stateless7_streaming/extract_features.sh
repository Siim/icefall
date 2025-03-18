#!/bin/bash

# Script to extract XLSR features from the Estonian dataset
# Usage: ./extract_features.sh [--force] [--no-single-process-s3prl]

set -e  # Exit on error

# Check for flags
FORCE_FLAG=""
SINGLE_PROCESS_S3PRL="--single-process-s3prl"

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_FLAG="--force"
            echo "Will force overwrite existing features"
            shift
            ;;
        --no-single-process-s3prl)
            SINGLE_PROCESS_S3PRL=""
            echo "Will attempt to use multiple processes for S3PRL feature extraction (may cause pickling errors)"
            shift
            ;;
        --single-process)
            echo "WARNING: --single-process is deprecated, use --no-single-process-s3prl instead"
            SINGLE_PROCESS_S3PRL="--single-process-s3prl"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./extract_features.sh [--force] [--no-single-process-s3prl]"
            exit 1
            ;;
    esac
done

# Auto-detect macOS and recommend single process
if [[ "$(uname)" == "Darwin" && -z "$SINGLE_PROCESS_S3PRL" ]]; then
    echo "MacOS detected. Parallel processing will likely cause pickling issues with S3PRL."
    echo "Forcing --single-process-s3prl for macOS compatibility"
    SINGLE_PROCESS_S3PRL="--single-process-s3prl"
fi

# Number of parallel jobs to use for resampling (adjust based on CPU cores available)
NUM_JOBS=10
echo "Using $NUM_JOBS parallel jobs for feature extraction"
echo "S3PRL feature extraction will use single process by default to avoid pickling errors"

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
    $SINGLE_PROCESS_S3PRL \
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
    $SINGLE_PROCESS_S3PRL \
    $FORCE_FLAG

echo "Feature extraction complete!"
echo "The features are saved in data/ssl" 