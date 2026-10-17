#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible inference service.
#
# 1) launch vLLM api_server in background (log to file)
# 2) wait for /v1/models readiness
# 3) print verification curl commands
#
# Usage: edit CONFIG below, then: bash vllm_test.sh
#
# Ascend (vllm-ascend): same CLI; plugin auto-activates, or set
#   export VLLM_TARGET_DEVICE=npu

set -euo pipefail

# ========================== CONFIG ==========================

# Python / vLLM
VLLM_BIN="/usr/local/python3.12.13/bin/vllm"

# 模型路径（修改为你的实际路径）
MODEL_PATH="/data/models/Qwen2.5-7B-Instruct"

# NPU 逻辑设备
NPU_DEVICES="6,7"

# Tensor Parallel
TP_SIZE=2

# 服务配置
HOST="0.0.0.0"
PORT=8001
SERVED_MODEL_NAME="Qwen2.5-7B-Instruct"

# 模型配置
DTYPE="bfloat16"
MAX_MODEL_LEN=8192
GPU_MEMORY_UTILIZATION=0.8


# ============================================================

export ASCEND_VISIBLE_DEVICES="$NPU_DEVICES"
export TOKENIZERS_PARALLELISM=false
export VLLM_PLUGINS=ascend


echo "=========================================="
echo "       vLLM Ascend Server"
echo "=========================================="
echo "Model:      $MODEL_PATH"
echo "NPU:        $NPU_DEVICES"
echo "TP Size:    $TP_SIZE"
echo "Host:       $HOST"
echo "Port:       $PORT"
echo "=========================================="

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "[ERROR] Model config.json not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -x "$VLLM_BIN" ]]; then
    echo "[ERROR] vLLM executable not found: $VLLM_BIN"
    exit 1
fi

"$VLLM_BIN" serve \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
