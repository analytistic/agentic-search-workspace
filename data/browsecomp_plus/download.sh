#!/usr/bin/env bash
# Download the BrowseComp-Plus datasets (queries + corpus) from Hugging Face.
#
# Official sources (Tevatron on Hugging Face):
#   - Tevatron/browsecomp-plus        split=test  revision 144cff8e35b5eaef7e526346aa60774a9deb941f
#   - Tevatron/browsecomp-plus-corpus split=train revision 89e412cede7d9904f0e8b717aebd9b317c46dfdc
#
# Usage:
#   bash download.sh                 # into ./raw (default)
#   DATA_DIR=/path bash download.sh  # custom target dir
#
# For HF mirrors (e.g. hf-mirror.com), set HF_ENDPOINT before running:
#   HF_ENDPOINT=https://hf-mirror.com bash download.sh

set -euo pipefail

DATA_DIR="${DATA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/raw}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_ENDPOINT

QUERY_REPO="Tevatron/browsecomp-plus"
QUERY_REV="144cff8e35b5eaef7e526346aa60774a9deb941f"
CORPUS_REPO="Tevatron/browsecomp-plus-corpus"
CORPUS_REV="89e412cede7d9904f0e8b717aebd9b317c46dfdc"

mkdir -p "$DATA_DIR"

echo "==> [1/2] queries  ($QUERY_REPO @ $QUERY_REV)"
huggingface-cli download "$QUERY_REPO" --repo-type dataset \
    --revision "$QUERY_REV" --local-dir "$DATA_DIR/queries"

echo "==> [2/2] corpus   ($CORPUS_REPO @ $CORPUS_REV)"
huggingface-cli download "$CORPUS_REPO" --repo-type dataset \
    --revision "$CORPUS_REV" --local-dir "$DATA_DIR/corpus"

echo "Done. Datasets under: $DATA_DIR"
