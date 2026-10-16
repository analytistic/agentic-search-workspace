#!/usr/bin/env bash
# CPU retrieval server smoke test + latency benchmark.
#
# 1) 启动检索服务（CPU 模式，不加 --faiss_gpu）
# 2) 等待 /status 就绪
# 3) 发送 3 条 query 批量测速
# 4) 发送 5 条 query 批量测速
# 5) 自动停止服务
#
# 用法：直接修改下方【配置区】的路径，然后执行:
#   bash test_retrieval.sh

set -euo pipefail

# ---- 仓库根目录（脚本位于 <repo>/agentic-search/src/agentic_search/training/examples/retrieval/test/）----
# 自动探测一次；如果探测不对，手动改掉即可
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)}"

# ========================== 配置区 ==========================
# 按你的机器环境直接改这里，不需要命令行参数
# -----------------------------------------------------------

# Python 解释器（uv 项目环境；服务器上可按需换成绝对路径）
PYTHON="$REPO_ROOT/agentic-search/.venv/bin/python"

# 检索服务脚本所在目录
SERVER_DIR="$REPO_ROOT/agentic-search/src/agentic_search/training/examples/retrieval"

# 索引文件（demo 用 e5_demo.index；服务器全量用 e5_HNSW64.index）
INDEX="$REPO_ROOT/data/wiki-demo/e5_demo.index"

# 语料文件（demo 用 wiki-demo；服务器全量用 wiki-18.jsonl）
CORPUS="$REPO_ROOT/data/wiki-demo/corpus.jsonl"

# e5 编码模型路径
MODEL="$REPO_ROOT/data/wiki_2018/model/intfloat/e5-base-v2"

# 服务端口
PORT="8000"

# 每批返回 top-k 条
TOPK="3"
# ============================================================

LOG="/tmp/retrieval_test.log"

echo "==> 启动 CPU 检索服务（无 --faiss_gpu）..."
# OMP_NUM_THREADS=1: 避免 macOS 上 torch+faiss 线程冲突（段错误/死锁）；Linux 上也无害
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false "$PYTHON" "$SERVER_DIR/retrieval_server.py" \
    --index_path "$INDEX" \
    --corpus_path "$CORPUS" \
    --topk "$TOPK" \
    --retriever_name e5 \
    --retriever_model "$MODEL" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "==> 等待就绪..."
ready=0
for i in $(seq 1 90); do
    if curl -fsS -o /dev/null "http://localhost:$PORT/status" 2>/dev/null; then
        ready=1
        echo "    就绪（${i}s）"
        break
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    echo "服务未就绪，日志尾部："
    tail -5 "$LOG"
    exit 1
fi

post_batch() {
    local label="$1" payload="$2"
    local start_ns end_ns elapsed_ms
    start_ns=$(python3 -c 'import time; print(time.time_ns())')
    local body
    body="$(curl -s -X POST "http://localhost:$PORT/retrieve" \
        -H "Content-Type: application/json" -d "$payload")"
    end_ns=$(python3 -c 'import time; print(time.time_ns())')
    elapsed_ms=$(( (end_ns - start_ns) / 1000000 ))
    local count
    count="$(printf '%s' "$body" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["result"]))')"
    echo "    $label: ${elapsed_ms} ms（返回 ${count} 组结果）"
}

echo "==> 3 queries 批量测速"
post_batch "3 queries" '{"queries": ["Who founded the University of Cambridge?", "What is the capital of France?", "Albert Einstein theory of relativity"], "topk": '"$TOPK"', "return_scores": true}'

echo "==> 5 queries 批量测速"
post_batch "5 queries" '{"queries": ["Who founded the University of Cambridge?", "What is the capital of France?", "Albert Einstein theory of relativity", "Python programming language creator", "Great Wall of China history"], "topk": '"$TOPK"', "return_scores": true}'

echo "==> 完成，服务已停止"
