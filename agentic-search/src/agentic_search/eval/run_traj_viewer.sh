#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Start the trajectory viewer.
#
#   bash run_traj_viewer.sh                       # /root/data/eval, port 8080
#   bash run_traj_viewer.sh /some/other/dir      # CLI 传路径
#   TRAJ_PORT=9000 bash run_traj_viewer.sh
#
# From your laptop, forward the port and open http://127.0.0.1:8080/ :
#   ssh -N -L 8080:127.0.0.1:8080 -p <ssh-port> root@ssh-cn-huabei1.ebcloud.com
# ---------------------------------------------------------------------------
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN_DIR="${PYTHON_BIN_DIR:-/root/miniconda3/bin}"
export PATH="$PYTHON_BIN_DIR:$PATH"

ROOT="${1:-${TRAJ_ROOT:-/root/data/eval}}"
PORT="${TRAJ_PORT:-8080}"
HOST="${TRAJ_HOST:-0.0.0.0}"

echo "trajectory viewer: root=$ROOT  http://$HOST:$PORT/"
exec python3 "$EVAL_DIR/traj_viewer.py" --root "$ROOT" --host "$HOST" --port "$PORT"
