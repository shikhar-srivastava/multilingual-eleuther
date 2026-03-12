#!/usr/bin/env bash
# Train Split: Take 1 GB * Byte Premium of each monolingual dataset
# Eval Split: Take last 8,000 lines of each monolingual dataset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"

python "${SCRIPT_DIR}/scripts/create_bp_splits.py" \
  --input_root "${DATA_ROOT}/monolingual_training_data" \
  --output_root "${DATA_ROOT}/monolingual_training_data_bp"
