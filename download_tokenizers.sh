#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='catherinearnett/montok',
    repo_type='dataset',
    allow_patterns=['bpe_unscaled_tokenizers/*', 'unigram_unscaled_tokenizers/*'],
    local_dir='${DATA_ROOT}/monolingual-tokenizers'
)
print('Done!')
"