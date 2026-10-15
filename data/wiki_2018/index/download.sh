#!/usr/bin/env bash
# Download the e5_Flat FAISS index (dense retrieval index over wiki-18).
#
# Official source: PeterJinGo/wiki-18-e5-index on Hugging Face
#   part_aa (~43 GB) + part_ab (~22 GB) -> cat -> e5_Flat.index (~65 GB)
#
# Usage:
#   bash download.sh                  # into ./
#   OUTPUT_DIR=/path bash download.sh
# For HF mirrors (e.g. hf-mirror.com): HF_ENDPOINT=https://hf-mirror.com bash download.sh

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
BASE="$HF_ENDPOINT/datasets/PeterJinGo/wiki-18-e5-index/resolve/main"

mkdir -p "$OUTPUT_DIR"

echo "==> Downloading part_aa (~43 GB) ..."
curl -L --fail --retry 3 -C - "$BASE/part_aa" -o "$OUTPUT_DIR/part_aa"

echo "==> Downloading part_ab (~22 GB) ..."
curl -L --fail --retry 3 -C - "$BASE/part_ab" -o "$OUTPUT_DIR/part_ab"

echo "==> Merging into e5_Flat.index ..."
cat "$OUTPUT_DIR/part_aa" "$OUTPUT_DIR/part_ab" > "$OUTPUT_DIR/e5_Flat.index"
rm "$OUTPUT_DIR/part_aa" "$OUTPUT_DIR/part_ab"

echo "Done: $OUTPUT_DIR/e5_Flat.index"
echo "Next: point RETRIEVER_INDEX_PATH at this file."
