#!/usr/bin/env bash
# Backs up /root/data/ to /mnt/ssd-3/shikhar/data/
# Safe to re-run: rsync skips already-transferred files.
set -euo pipefail

SRC="/root/data/"
DEST="/mnt/ssd-3/shikhar/data/"

echo "[backup] $(date '+%Y-%m-%d %H:%M:%S')  $SRC  ->  $DEST"

mkdir -p "$DEST"

rsync -av --progress \
  --human-readable \
  --checksum \
  "$SRC" "$DEST"

echo "[backup] Done at $(date '+%Y-%m-%d %H:%M:%S')"
