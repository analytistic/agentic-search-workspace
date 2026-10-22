#!/usr/bin/env bash
# Occupy NPU cards 12,13,14,15 (64 GiB each) with ~60 GiB resident bf16
# tensors + continuous matmul loops.
#
# Usage:
#   bash occupy_npu.sh [start] [end] [gib]
#   bash occupy_npu.sh            # 12..15, 60 GiB each
#   bash occupy_npu.sh 12 15 60
#
# Stop: kill the occupy_npu.py processes (pkill -f occupy_npu.py).

set -euo pipefail

START="${1:-12}"
END="${2:-15}"
GIB="${3:-60}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for npu_id in $(seq "$START" "$END"); do
    rank=$((npu_id - START))
    echo "==> occupying NPU ${npu_id} (local rank ${rank}, ${GIB} GiB)"
    ASCEND_RT_VISIBLE_DEVICES="${npu_id}" nohup python3 "${SCRIPT_DIR}/occupy_npu.py" \
        --rank "$rank" --gib "$GIB" \
        > "/tmp/occupy_npu_${npu_id}.log" 2>&1 &
done

echo "launched; logs: /tmp/occupy_npu_<npu_id>.log"
echo "check:  npu-smi info | grep -E '12|13|14|15'"
echo "stop:   pkill -f occupy_npu.py"
