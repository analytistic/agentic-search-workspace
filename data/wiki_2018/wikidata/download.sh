#!/usr/bin/env bash
# Download the Wikidata JSON dump (knowledge-graph source for the
# kg_subgraph_extraction step of the SSP-KG training pipeline).
#
# Official source: dumps.wikimedia.org/wikidatawiki/entities/
#   file: wikidata-<DATE>-all.json.bz2  (~103 GB compressed; the extraction
#   code streams the bz2 directly — no decompression needed)
#
# Strategy: rsync preferred (resumable, chunk-verified); falls back to HTTP
# (curl -C -) automatically if the rsync service is unreachable. The dump
# date is auto-detected from the HTTP index — there is no "latest" alias.
#
# Usage:
#   bash download.sh                          # newest complete dump -> ./
#   WIKIDATA_DATE=20261012 bash download.sh   # pinned dump date
#   OUTPUT_DIR=/path bash download.sh
#   DOWNLOAD_METHOD=rsync|http bash download.sh   # force a method
#
# The transfer is verified against the official sha1sums.txt afterwards.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RSYNC_BASE="${RSYNC_BASE:-rsync://rsync.wikimedia.org/wikidatawiki/entities}"
HTTP_BASE="https://dumps.wikimedia.org/wikidatawiki/entities"

mkdir -p "$OUTPUT_DIR"

# Resolve dump date: explicit WIKIDATA_DATE wins; otherwise scan the HTTP
# index of dated directories (newest first) for one that serves all.json.bz2.
resolve_date() {
    if [ -n "${WIKIDATA_DATE:-}" ]; then
        echo "$WIKIDATA_DATE"
        return 0
    fi
    local index
    index="$(curl -fsSL --max-time 30 "$HTTP_BASE/" 2>/dev/null)"
    for d in $(printf '%s' "$index" | grep -oE 'href="[0-9]{8}/"' | grep -oE '[0-9]{8}' | sort -r); do
        if curl -fsS --max-time 30 "$HTTP_BASE/$d/" 2>/dev/null | grep -q "all.json.bz2"; then
            echo "$d"
            return 0
        fi
    done
    echo "error: could not auto-detect a complete dump; set WIKIDATA_DATE explicitly" >&2
    exit 1
}

DATE="$(resolve_date)"
echo "==> Dump date: $DATE"
FILE="wikidata-$DATE-all.json.bz2"
SUMS="wikidata-$DATE-sha1sums.txt"

# Download via rsync if available; fall back to HTTP.
METHOD="${DOWNLOAD_METHOD:-rsync}"
if [ "$METHOD" = "rsync" ]; then
    if command -v rsync >/dev/null 2>&1 && rsync --list-only "$RSYNC_BASE/$DATE/" >/dev/null 2>&1; then
        echo "==> [rsync] Syncing $FILE (resumable) ..."
        rsync -av --partial --progress "$RSYNC_BASE/$DATE/$FILE" "$OUTPUT_DIR/"
    else
        echo "==> rsync unavailable for this date; falling back to HTTP ..."
        METHOD="http"
    fi
fi

if [ "$METHOD" = "http" ]; then
    echo "==> [http] Downloading $FILE (resumable) ..."
    curl -L --fail --retry 3 -C - "$HTTP_BASE/$DATE/$FILE" -o "$OUTPUT_DIR/$FILE"
fi

# Verify against the official checksums.
echo "==> Verifying sha1 ..."
if [ "$METHOD" = "rsync" ]; then
    rsync -a "$RSYNC_BASE/$DATE/$SUMS" "$OUTPUT_DIR/"
else
    curl -fsSL --max-time 60 "$HTTP_BASE/$DATE/$SUMS" -o "$OUTPUT_DIR/$SUMS"
fi
(cd "$OUTPUT_DIR" && grep "all.json.bz2" "$SUMS" | sha1sum -c - && rm -f "$SUMS")

echo "Done: $OUTPUT_DIR/$FILE"
echo "Next: use it as --wikidata-file in kg_subgraph_extraction.main"
