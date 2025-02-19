#!/bin/bash

set -eou pipefail

# Parse command line arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 /path/to/Data/dir"
    echo "Example: $0 /Users/username/project/Data"
    exit 1
fi

DATA_DIR=$(realpath "$1")
echo "Using data directory: $DATA_DIR"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ICEFALL_ROOT="$( cd "$SCRIPT_DIR/../../../../" &> /dev/null && pwd )"

cd "$ICEFALL_ROOT"

# Directory setup
lang_dir="$ICEFALL_ROOT/data/lang_bpe_2500"
mkdir -p "$lang_dir"
mkdir -p "$ICEFALL_ROOT/data"

# Extract transcripts from train_list.txt for BPE training
echo "Extracting transcripts from train_list.txt..."
# Extract second field (transcript) from pipe-separated format
cut -d'|' -f2 "$DATA_DIR/train_list.txt" > "$lang_dir/transcripts.txt"

# Train BPE model
echo "Training BPE model..."
python "$ICEFALL_ROOT/egs/librispeech/ASR/local/train_bpe_model.py" \
  --lang-dir "$lang_dir" \
  --transcript "$lang_dir/transcripts.txt" \
  --vocab-size 2500

# The above will create:
# - $lang_dir/bpe.model: the trained BPE model
# - $lang_dir/tokens.txt: mapping between tokens and their IDs

echo "BPE model training completed. Model saved to $lang_dir/bpe.model"
echo "Token list saved to $lang_dir/tokens.txt"

# Create exp directory for storing checkpoints
exp_dir="$SCRIPT_DIR/exp"
mkdir -p "$exp_dir"

# Create symlinks to data files for consistency
ln -sf "$DATA_DIR/train_list.txt" "$ICEFALL_ROOT/data/train.txt"
ln -sf "$DATA_DIR/val_list.txt" "$ICEFALL_ROOT/data/val.txt"

# Update notepad with preparation steps
echo "
## Data Preparation ($(date))
- Using data directory: $DATA_DIR
- Created data directory structure
- Trained BPE model with vocab_size=2500 using transcripts from train_list.txt
- Created exp directory for checkpoints
- Created symlinks to data files
- Files created:
  * $lang_dir/bpe.model
  * $lang_dir/tokens.txt
  * $lang_dir/transcripts.txt
  * $ICEFALL_ROOT/data/train.txt -> $DATA_DIR/train_list.txt
  * $ICEFALL_ROOT/data/val.txt -> $DATA_DIR/val_list.txt
" >> "$ICEFALL_ROOT/NOTEPAD.md"

echo "Preparation completed successfully!"
echo "You can now run training with:"
echo "python $SCRIPT_DIR/train.py \\"
echo "  --world-size 1 \\"
echo "  --num-epochs 30 \\"
echo "  --dataset estonian \\"
echo "  --train-txt data/train.txt \\"
echo "  --val-txt data/val.txt \\"
echo "  --batch-size 4 \\"
echo "  --use-xlsr 1 \\"
echo "  --xlsr-model-name facebook/wav2vec2-xls-r-300m" 