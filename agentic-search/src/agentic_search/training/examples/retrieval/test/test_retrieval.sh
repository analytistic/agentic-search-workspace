#!/usr/bin/env bash
# Retrieval server benchmark: batch-size matrix + concurrency matrix.
#
# 1) 启动检索服务（CPU 模式）
# 2) 等待 /status 就绪（每 5 秒刷新服务日志）
# 3) 预热 1 轮（让模型/缓存热起来）
# 4) 批量测速: 对每个 QUERY_SIZES 中的条数, 重复 REPEATS 次取平均
# 5) 并发测速: 对每个 CONCURRENCY_LEVELS 中的并发数, 同时发 N 个请求测吞吐
# 6) 自动停止服务
#
# 用法: 改【配置区】后执行: bash test_retrieval.sh

set -euo pipefail

# ---- 仓库根目录（脚本位于 <repo>/agentic-search/src/agentic_search/training/examples/retrieval/test/）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)}"

# ========================== 配置区 ==========================
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

# 就绪超时（秒）：服务器全量加载 10-20 分钟；demo 几秒
READY_TIMEOUT="1800"

# ---- 测速场景设置 ----

# 批量测速：每请求塞多少条 query（空格分隔，逐个测）
QUERY_SIZES="1 3 5 8"

# 并发测速：同时发多少个请求（每个请求 1 条 query）
# ⚠️ 已知问题：本机 + 服务器（Linux/torch_npu）上并发独立请求都会死锁
# （多个线程同时调用 torch 前向）。默认留空 = 不测并发。
# 正确用法：需要并发/多条时，把 query 塞进【一个请求的 queries 列表】批量发。
# 想诊断并发时才手动改成 "1 4 8"（会卡住，每个请求 CURL_TIMEOUT 秒后超时）。
CONCURRENCY_LEVELS=""

# 每个场景重复几次取平均
REPEATS="3"

# 预热轮数（不计入统计，让模型/缓存热起来）
WARMUP="1"

# 每个请求的 curl 超时（秒）——请求挂起时不阻塞整个测速，超时记为失败
CURL_TIMEOUT="60"

# 每批返回 top-k 条
TOPK="3"

# ---- 查询池（测速用的 query，按需增删）----
QUERY_POOL=(
    "Who founded the University of Cambridge?"
    "What is the capital of France?"
    "Albert Einstein theory of relativity"
    "Python programming language creator"
    "Great Wall of China history"
    "Mount Everest height and location"
    "Shakespeare most famous plays"
    "Silicon Valley tech companies"
)
# ============================================================

LOG="/tmp/retrieval_test.log"

# 从查询池取前 N 条，生成 JSON 请求体（查询池通过环境变量 QPOOL 注入）
make_payload() {
    python3 - "$1" "$TOPK" <<'EOF'
import json, os, sys
n, topk = int(sys.argv[1]), int(sys.argv[2])
pool = json.loads(os.environ["QPOOL"])
print(json.dumps({"queries": pool[:n], "topk": topk, "return_scores": True}))
EOF
}

export QPOOL="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${QUERY_POOL[@]}")"

echo "==> 启动 CPU 检索服务（无 --faiss_gpu）..."
# OMP_NUM_THREADS=1: 避免 macOS 上 torch+faiss 线程冲突；Linux 无害
# PYTHONUNBUFFERED=1: 服务日志实时落盘
OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "$PYTHON" "$SERVER_DIR/retrieval_server.py" \
    --index_path "$INDEX" \
    --corpus_path "$CORPUS" \
    --topk "$TOPK" \
    --retriever_name e5 \
    --retriever_model "$MODEL" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

echo "==> 等待就绪（每 5 秒刷新服务日志；超时 ${READY_TIMEOUT}s）..."
ready=0
for i in $(seq 1 "$READY_TIMEOUT"); do
    if curl -fsS -o /dev/null "http://localhost:$PORT/status" 2>/dev/null; then
        ready=1
        echo "    就绪（${i}s）"
        break
    fi
    if [ $((i % 5)) -eq 0 ]; then
        last_line="$(tail -1 "$LOG" 2>/dev/null | tr -d '\r' | sed 's/^ *//;s/ *$//')"
        [ -n "$last_line" ] && echo "    [${i}s] $last_line"
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    echo "服务未就绪，日志尾部："
    tail -5 "$LOG"
    exit 1
fi

# 单次请求耗时（毫秒），返回响应体（变量 RESULT_BODY）
request_ms() {
    local payload="$1"
    local start_ns end_ns
    start_ns=$(python3 -c 'import time; print(time.time_ns())')
    RESULT_BODY="$(curl -s --max-time "$CURL_TIMEOUT" -X POST "http://localhost:$PORT/retrieve" \
        -H "Content-Type: application/json" -d "$payload")"
    end_ns=$(python3 -c 'import time; print(time.time_ns())')
    echo $(( (end_ns - start_ns) / 1000000 ))
}

echo "==> 预热（${WARMUP} 轮）..."
for _ in $(seq 1 "${WARMUP:-1}"); do
    request_ms "$(make_payload 1)" > /dev/null
done
echo "    预热完成"

echo
echo "================ 批量测速（每请求 N 条 query）================"
printf "%-14s %-12s %-14s %-14s\n" "每条数" "平均耗时(ms)" "每query(ms)" "吞吐(q/s)"
for size in $QUERY_SIZES; do
    total=0
    for _ in $(seq 1 "$REPEATS"); do
        ms=$(request_ms "$(make_payload "$size")")
        total=$((total + ms))
    done
    avg=$((total / REPEATS))
    per_query=$((avg / size))
    qps=$((size * 1000 / avg))
    printf "%-14s %-12s %-14s %-14s\n" "$size" "$avg" "$per_query" "$qps"
done

echo
echo "================ 并发测速（N 个请求同时发，每请求 1 条）================"
printf "%-14s %-16s %-14s %-16s\n" "并发数" "总耗时(ms)" "每请求(ms)" "吞吐(req/s)"
for n in $CONCURRENCY_LEVELS; do
    payload="$(make_payload 1)"
    start_ns=$(python3 -c 'import time; print(time.time_ns())')
    for _ in $(seq 1 "$n"); do
        curl -s --max-time "$CURL_TIMEOUT" -o /dev/null -X POST "http://localhost:$PORT/retrieve" \
            -H "Content-Type: application/json" -d "$payload" &
    done
    wait
    end_ns=$(python3 -c 'import time; print(time.time_ns())')
    total_ms=$(( (end_ns - start_ns) / 1000000 ))
    per_req=$((total_ms / n))
    rps=$((n * 1000 / total_ms))
    printf "%-14s %-16s %-14s %-16s\n" "$n" "$total_ms" "$per_req" "$rps"
done

echo
echo "==> 完成，服务已停止"
