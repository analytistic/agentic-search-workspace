#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start the retrieval server for rollout_eval (and training rollouts).
#
#   bash start_retriever.sh                # dense e5 + HNSW (what training uses)
#   bash start_retriever.sh bm25           # Lucene BM25 fallback
#   PORT=8001 bash start_retriever.sh      # search_tool_fn hardcodes 8000 unless
#                                          # RETRIEVAL_URL is set, so keep 8000
#
# Why dense needs two non-obvious flags (both learned the hard way):
#
# 1. --no-index-mmap
#    faiss' IO_FLAG_MMAP reads the graph/vector blob lazily. On Lustre the
#    mmap read of a 70G file fails outright with
#      RuntimeError: Error in void faiss::read_xb_vector(...) at index_read.cpp:27
#    and the server dies during startup, before uvicorn binds. Loading the
#    whole index instead costs ~155 s and 70.5G RSS, which is fine on a 200G
#    box, and gives 2 ms/query. Prefer it whenever RAM allows.
#
# 2. PREWARM=1 (default)
#    Offsets are pread'd straight out of the 14G corpus jsonl for every hit.
#    With the file cold on Lustre a query costs ~85 ms; once it sits in page
#    cache it drops to ~20 ms and 128-way concurrency finishes in 1.4 s.
#    We touch the file sequentially (dd) after the server is ready, which is
#    cheaper than letting random reads fault it in.
#
# Memory budget on the 2x4090D box (200G cgroup):
#   index 70.5G (anon) + corpus page cache 13.3G + torch/sglang ~ a few G
# ---------------------------------------------------------------------------
set -uo pipefail

MODE="${1:-dense}"

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$EVAL_DIR/../../../.." && pwd)}"
RETRIEVAL_DIR="$REPO_ROOT/agentic-search/src/agentic_search/training/examples/retrieval"

PYTHON_BIN_DIR="${PYTHON_BIN_DIR:-/root/miniconda3/bin}"
DATA_ROOT="${DATA_ROOT:-/root/data}"
INDEX_PATH="${INDEX_PATH:-$DATA_ROOT/wiki/index/hnsw64/merged_hnsw64.index}"
BM25_INDEX_PATH="${BM25_INDEX_PATH:-$DATA_ROOT/wiki/index/bm25}"
CORPUS_PATH="${CORPUS_PATH:-$DATA_ROOT/wiki/corpus/jsonl/wiki_dump.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-$DATA_ROOT/models/e5-base-v2}"
PORT="${PORT:-8000}"
TOPK="${TOPK:-3}"
CACHE_ENTRIES="${CACHE_ENTRIES:-10000}"
MAX_PROCESS_NUM="${MAX_PROCESS_NUM:-8}"
PREWARM="${PREWARM:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
LOG="${LOG:-/root/retriever_${MODE}.log}"

export PATH="$PYTHON_BIN_DIR:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# pyserini instantiates an OpenAI client at import time and aborts without a key
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

if [[ ! -d "$RETRIEVAL_DIR" ]]; then
    echo "retrieval dir not found: $RETRIEVAL_DIR" >&2
    exit 1
fi

case "$MODE" in
    dense)
        [[ -f "$INDEX_PATH" ]]  || { echo "index not found: $INDEX_PATH"  >&2; exit 1; }
        ARGS=(
            --index_path "$INDEX_PATH"
            --corpus_path "$CORPUS_PATH"
            --topk "$TOPK"
            --retriever_name e5
            --retriever_model "$RETRIEVER_MODEL"
            --cache-entries "$CACHE_ENTRIES"
            --faiss-device cpu
            --no-index-mmap
            --port "$PORT"
        )
        ;;
    bm25)
        [[ -d "$BM25_INDEX_PATH" ]] || { echo "bm25 index not found: $BM25_INDEX_PATH" >&2; exit 1; }
        # Lucene's default heap is sized from *host* memory; cap it or the JVM
        # tries to reserve tens of GB inside the container.
        export _JAVA_OPTIONS="${_JAVA_OPTIONS:--Xmx4g}"
        ARGS=(
            --index_path "$BM25_INDEX_PATH"
            --corpus_path "$CORPUS_PATH"
            --topk "$TOPK"
            --retriever_name bm25
            --cache-entries "$CACHE_ENTRIES"
            --max-process-num "$MAX_PROCESS_NUM"
            --port "$PORT"
        )
        ;;
    *)
        echo "usage: $0 [dense|bm25]" >&2
        exit 2
        ;;
esac

if curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/status"; then
    echo "something is already listening on :$PORT -- stop it first" >&2
    exit 1
fi

echo "starting $MODE retriever on :$PORT (log: $LOG)"
: > "$LOG"
cd "$RETRIEVAL_DIR"
setsid nohup python retrieval_server.py "${ARGS[@]}" >> "$LOG" 2>&1 < /dev/null &

echo -n "waiting for /status"
deadline=$(( SECONDS + READY_TIMEOUT ))
while (( SECONDS < deadline )); do
    if out=$(curl -s -m 3 "http://127.0.0.1:$PORT/status" 2>/dev/null) && [[ "$out" == *'"ready":true'* ]]; then
        echo
        echo "$out"
        if [[ "$MODE" == "dense" && "$PREWARM" == "1" ]]; then
            echo "prewarming corpus page cache ($(du -h "$CORPUS_PATH" | cut -f1))..."
            dd if="$CORPUS_PATH" of=/dev/null bs=8M 2>&1 | tail -1
        fi
        exit 0
    fi
    echo -n "."
    sleep 5
done
echo
echo "timed out after ${READY_TIMEOUT}s -- last log lines:" >&2
tail -20 "$LOG" >&2
exit 1
