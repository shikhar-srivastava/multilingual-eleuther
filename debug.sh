#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local.env"

LOG_DIR="${SCRIPT_DIR}/logs"; mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S); LOG="$LOG_DIR/vie_run_${TS}.log"
ln -sfn "$LOG" "$LOG_DIR/vie_run_latest.log"

export goldfish=True

python -c 'import transformers, torch; print("env ok")' || { echo "env check failed"; exit 1; }

stdbuf -oL -eL bash "${SCRIPT_DIR}/launch_monolingual_vie.sh" |& tee -a "$LOG"
