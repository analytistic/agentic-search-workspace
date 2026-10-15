#!/usr/bin/env bash
# Download the Wikidata JSON dump (knowledge-graph source for the
# kg_subgraph_extraction step of the SSP-KG training pipeline).
#
# Official source: dumps.wikimedia.org/wikidatawiki/entities/
#   file: wikidata-<DATE>-all.json.bz2  (~30-40 GB compressed, streamed by the
#   extraction code — no need to decompress)
#
# Usage:
#   bash download.sh                  # latest dump -> ./
#   WIKIDATA_DATE=20261001 bash download.sh   # pinned dump date
#   OUTPUT_DIR=/path bash download.sh
#
# The extraction client accepts this file directly via
#   --wikidata-file <path>/wikidata-<DATE>-all.json.bz2

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DATE="${WIKIDATA_DATE:-latest}"

mkdir -p "$OUTPUT_DIR"

URL="https://dumps.wikimedia.org/wikidatawiki/entities/wikidata-${DATE}-all.json.bz2"
OUT_PATH="$OUTPUT_DIR/wikidata-${DATE}-all.json.bz2"

echo "==> Downloading Wikidata dump ($URL) ..."
curl -L --fail --retry 3 -C - "$URL" -o "$OUT_PATH"

echo "Done: $OUT_PATH"
echo "Next: use it as --wikidata-file in kg_subgraph_extraction.main"
