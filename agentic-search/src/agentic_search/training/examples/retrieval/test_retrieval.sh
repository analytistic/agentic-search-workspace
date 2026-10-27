
#!/usr/bin/env bash
# Search-R1 Retrieval Benchmark
#
# 1. 检查配置及端口
# 2. 启动 CPU Retriever
# 3. 等待 /status 返回 HTTP 200
# 4. 预热
# 5. Batch Size 测速
# 6. 并发测速
# 7. 自动关闭本次启动的服务

set -euo pipefail

# ================= 配置区 =================

PYTHON="/usr/local/python3.12.13/bin/python"

REPO_ROOT="/home/xuwentao/agentic-search-workspace"

SERVER="$REPO_ROOT/agentic-search/src/agentic_search/training/examples/retrieval/retrieval_server.py"

INDEX="${INDEX:-/data/xuwentao_data/wiki-18-e5-index/index}"
CORPUS="${CORPUS:-/data/xuwentao_data/wiki_2018/wiki-18.jsonl}"
MODEL="${MODEL:-/data/models/e5-base-v2/}"

PORT="${PORT:-8000}"
TOPK="${TOPK:-3}"

# faiss search device: cpu (default fallback) | npu | gpu.
# npu uses ALL visible cards (replica/shard auto-sized): only the PHYSICAL
# cards need choosing, via ASCEND_RT_VISIBLE_DEVICES.
FAISS_DEVICE="${FAISS_DEVICE:-npu}"
NPU_VISIBLE_DEVICES="${NPU_VISIBLE_DEVICES:-6,7}"
FAISS_SHARD="${FAISS_SHARD:-1}"   # npu multi-card: 1 = shard base across cards, 0 = replica

READY_TIMEOUT="${READY_TIMEOUT:-1800}"
CURL_TIMEOUT="${CURL_TIMEOUT:-60}"

QUERY_SIZES="${QUERY_SIZES:-1 3 5 8}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-10 30 60}"

REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"

LOG="/tmp/retrieval_benchmark_${PORT}.log"

BASE_URL="http://127.0.0.1:${PORT}"

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

# ================= 环境检查 =================

# 禁止本地请求走代理
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="$NO_PROXY"

# encode 已在服务端串行化；放开 OMP 让单次 encode 吃满全核
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "========== Environment Check =========="

for path in "$PYTHON" "$SERVER" "$INDEX" "$CORPUS" "$MODEL"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: Path not found: $path"
        exit 1
    fi
done

"$PYTHON" -c "import httpx" || {
    echo "ERROR: httpx is not installed"
    exit 1
}

# npu preflight: faiss_ascend installed + at least one visible NPU
if [ "$FAISS_DEVICE" = "npu" ]; then
    export ASCEND_RT_VISIBLE_DEVICES="$NPU_VISIBLE_DEVICES"
    if ! "$PYTHON" <<'PY'
import faiss
import sys

if faiss.get_num_npus() < 1:
    sys.exit(1)
print(f"faiss NPU ready: {faiss.get_num_npus()} visible device(s)")
PY
    then
        echo "ERROR: faiss NPU preflight failed: no NPU visible via"
        echo "       ASCEND_RT_VISIBLE_DEVICES=$NPU_VISIBLE_DEVICES (install faiss_ascend wheel)"
        exit 1
    fi
fi

if [ "${#QUERY_POOL[@]}" -lt 1 ]; then
    echo "ERROR: Empty query pool"
    exit 1
fi

# 防止重复启动 Retriever
if ss -ltn | grep -qE ":${PORT}[[:space:]]"; then
    echo "ERROR: Port ${PORT} is already occupied"
    ss -lntp | grep ":${PORT}" || true
    exit 1
fi

# ================= 服务管理 =================

SERVER_PID=""

cleanup() {
    echo
    echo "========== Cleanup =========="

    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping Retriever PID=$SERVER_PID"

        kill -TERM "$SERVER_PID" 2>/dev/null || true

        for _ in $(seq 1 10); do
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done

        wait "$SERVER_PID" 2>/dev/null || true
    fi

    echo "Cleanup finished"
}

trap cleanup EXIT INT TERM

# ================= 启动服务 =================

echo
echo "========== Starting Retriever =========="

echo "Index:  $INDEX"
echo "Corpus: $CORPUS"
echo "Model:  $MODEL"
echo "Port:   $PORT"
echo "Log:    $LOG"

FAISS_ARGS=(--faiss-device "$FAISS_DEVICE")
if [ "$FAISS_SHARD" = "1" ]; then
    FAISS_ARGS+=(--faiss-shard)
fi

"$PYTHON" "$SERVER" \
    --index_path "$INDEX" \
    --corpus_path "$CORPUS" \
    --topk "$TOPK" \
    --retriever_name e5 \
    --retriever_model "$MODEL" \
    "${FAISS_ARGS[@]}" \
    > "$LOG" 2>&1 &

SERVER_PID=$!

echo "Retriever PID=$SERVER_PID"

# ================= 等待就绪 =================

echo
echo "========== Waiting for /status =========="

ready=0
start_time=$SECONDS

while (( SECONDS - start_time < READY_TIMEOUT )); do

    http_code=$(curl \
        --noproxy '*' \
        -sS \
        --connect-timeout 2 \
        --max-time 5 \
        -o /dev/null \
        -w '%{http_code}' \
        "$BASE_URL/status" 2>/dev/null) || true

    if [ "$http_code" = "200" ]; then
        ready=1
        echo "Retriever ready!"
        break
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Retriever process exited"
        echo "Check log: $LOG"
        exit 1
    fi

    elapsed=$((SECONDS - start_time))

    if (( elapsed % 10 == 0 )); then
        echo "Waiting... ${elapsed}s / ${READY_TIMEOUT}s"
    fi

    sleep 1
done

if [ "$ready" -ne 1 ]; then
    echo "ERROR: Retriever readiness timeout"
    echo "Check log: $LOG"
    exit 1
fi

# ================= 生成请求 =================

export QPOOL="$(
    "$PYTHON" -c \
    'import json,sys; print(json.dumps(sys.argv[1:]))' \
    "${QUERY_POOL[@]}"
)"

make_payload() {
    "$PYTHON" - "$1" "$TOPK" <<'PY'
import json
import os
import sys

n = int(sys.argv[1])
topk = int(sys.argv[2])
pool = json.loads(os.environ["QPOOL"])

queries = [pool[i % len(pool)] for i in range(n)]

print(json.dumps({
    "queries": queries,
    "topk": topk,
    "return_scores": True
}))
PY
}

# ================= 单次请求测速 =================

request_ms() {
    local payload="$1"
    local start_ns end_ns

    start_ns=$("$PYTHON" -c \
        'import time; print(time.perf_counter_ns())')

    if ! curl \
        --noproxy '*' \
        -fsS \
        --connect-timeout 5 \
        --max-time "$CURL_TIMEOUT" \
        -o /dev/null \
        -X POST "$BASE_URL/retrieve" \
        -H "Content-Type: application/json" \
        -d "$payload"; then

        echo "ERROR: Retrieval request failed" >&2
        return 1
    fi

    end_ns=$("$PYTHON" -c \
        'import time; print(time.perf_counter_ns())')

    echo $(((end_ns - start_ns) / 1000000))
}

# ================= 预热 =================

echo
echo "========== Warmup =========="

for ((i=1; i<=WARMUP; i++)); do
    request_ms "$(make_payload 1)" > /dev/null
    echo "Warmup $i/$WARMUP done"
done

# ================= Batch 测速 =================

echo
echo "========== Batch Benchmark =========="

printf "%-10s %-15s %-15s %-15s\n" \
    "Batch" "Avg(ms)" "P95(ms)" "QPS"

for size in $QUERY_SIZES; do

    payload=$(make_payload "$size")
    times=()

    for ((i=1; i<=REPEATS; i++)); do
        ms=$(request_ms "$payload")
        times+=("$ms")
    done

    printf '%s\n' "${times[@]}" | \
        "$PYTHON" -c '
import sys
import math

times = [int(x) for x in sys.stdin if x.strip()]
batch = int(sys.argv[1])

avg = sum(times) / len(times)
sorted_times = sorted(times)
p95 = sorted_times[math.ceil(0.95 * len(times)) - 1]

qps = batch * 1000 / avg if avg > 0 else 0

print(f"{batch:<10} {avg:<15.2f} {p95:<15.2f} {qps:<15.2f}")
' "$size"

done

# ================= 并发测速 =================

echo
echo "========== Concurrency Benchmark =========="

printf "%-12s %-15s %-15s %-15s %-15s\n" \
    "Concurrency" "Avg(ms)" "P95(ms)" "Total(ms)" "QPS"

for level in $CONCURRENCY_LEVELS; do

    "$PYTHON" - \
        "$level" \
        "$REPEATS" \
        "$PORT" \
        "$TOPK" \
        "$CURL_TIMEOUT" <<'PY'

import asyncio
import json
import math
import os
import sys
import time

import httpx

level = int(sys.argv[1])
repeats = int(sys.argv[2])
port = int(sys.argv[3])
topk = int(sys.argv[4])
timeout = int(sys.argv[5])

pool = json.loads(os.environ["QPOOL"])

url = f"http://127.0.0.1:{port}/retrieve"

async def main():

    latencies = []
    total_queries = 0
    total_wall_time = 0.0

    async with httpx.AsyncClient(
        timeout=timeout,
        trust_env=False
    ) as client:

        async def one(query):
            payload = {
                "queries": [query],
                "topk": topk,
                "return_scores": True
            }

            t0 = time.perf_counter()

            response = await client.post(
                url,
                json=payload
            )

            response.raise_for_status()

            return (time.perf_counter() - t0) * 1000

        for repeat in range(repeats):

            queries = [
                pool[(repeat * level + i) % len(pool)]
                for i in range(level)
            ]

            t0 = time.perf_counter()

            results = await asyncio.gather(
                *[one(q) for q in queries]
            )

            wall_time = time.perf_counter() - t0

            latencies.extend(results)
            total_queries += len(queries)
            total_wall_time += wall_time

    avg = sum(latencies) / len(latencies)

    ordered = sorted(latencies)
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]

    qps = total_queries / total_wall_time

    print(
        f"{level:<12} "
        f"{avg:<15.2f} "
        f"{p95:<15.2f} "
        f"{total_wall_time * 1000:<15.2f} "
        f"{qps:<15.2f}"
    )

asyncio.run(main())
PY

done

echo
echo "========== Benchmark Completed =========="
echo "Retriever log: $LOG"
