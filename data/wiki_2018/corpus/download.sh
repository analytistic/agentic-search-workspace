#!/usr/bin/env bash
# Download the wiki-18 corpus (2018 English Wikipedia passages) used by the
# Search-R1 / SSP dense-retriever stack.
#
# Official source: PeterJinGo/wiki-18-corpus on Hugging Face
#   file: wiki-18.jsonl.gz  (~5.1 GB compressed; ~13-14 GB when decompressed)
# Format: JSONL, one doc per line: {"id": "...", "contents": "<title>\n<text>"}
#
# Usage:
#   bash download.sh                  # into ./
#   OUTPUT_DIR=/path bash download.sh
# For HF mirrors (e.g. hf-mirror.com): HF_ENDPOINT=https://hf-mirror.com bash download.sh

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
GZ_PATH="$OUTPUT_DIR/wiki-18.jsonl.gz"
OUT_PATH="$OUTPUT_DIR/wiki-18.jsonl"

mkdir -p "$OUTPUT_DIR"

echo "==> Downloading wiki-18.jsonl.gz (~5.1 GB) ..."
curl -L --fail --retry 3 -C - \
    "$HF_ENDPOINT/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz" \
    -o "$GZ_PATH"

echo "==> Decompressing (takes a few minutes, needs ~14 GB free) ..."
gzip -d -f "$GZ_PATH"

echo "Done: $OUT_PATH"
