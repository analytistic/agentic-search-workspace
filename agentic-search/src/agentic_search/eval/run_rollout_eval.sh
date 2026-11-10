#!/usr/bin/env bash
# Standalone 三模式 rollout 评估（Search-R1 / SSP 基线）。
# 不起 FSDP trainer，只起 vLLM agent loop workers（verl AgentLoopManager）。
# 用法：MODE=no_tool bash run_rollout_eval.sh   （MODE: agentic | single | no_tool）
#
# 产出：$OUTPUT_DIR/{mode}_trajectories.jsonl（每题每采样完整轨迹 + rm_score + 分量）
#       聚合指标进 mlflow（val-core / val-aux，每个 mode 一个 experiment run）

set -euo pipefail

# ---- paths ----
REPO_ROOT="${REPO_ROOT:-/home/xuwentao/agentic-search-workspace}"
EVAL_DIR="$REPO_ROOT/agentic-search/src/agentic_search/eval"
TRAINING_DIR="$REPO_ROOT/agentic-search/src/agentic_search/training"
export WORKSPACE="$TRAINING_DIR"     # search_r1_config.yaml hydra searchpath / function_tool_path
export PYTHONPATH="$TRAINING_DIR:$TRAINING_DIR/verl:${PYTHONPATH:-}"

# ---- .env（可选：环境变量放这里，git 不追踪；未设置项用下方默认值）----
if [ -f "$EVAL_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$EVAL_DIR/.env"
    set +a
fi

# hermes chat template（与训练 launcher 一致，经 env 传给 config）
export HERMES_JINJA="${HERMES_JINJA:-$(cat "$TRAINING_DIR/quarl/template/qwen3_hermes.jinja")}"

# ---- judge（敏感配置在 .env 的 JUDGE_* 中；未设置时用默认值兜底）----
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://judge_host:5000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-judge_model_name}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-dummy_api_key}"

# ---- trace（verl 原生：worker 内 rollout_trace_op 装饰器逐样本异步记录）----
# 与训练共用 MLFLOW_TRACKING_URI（同一 sqlite），但独立 experiment（rollout_eval）
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////data/mlruns.db}"
TRACE_BACKEND="${TRACE_BACKEND:-mlflow}"    # mlflow | trackio | null（null = 只留 jsonl）

# ---- eval 参数（可覆盖）----
MODE="${MODE:-agentic}"                       # agentic | single | no_tool
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/data/xuwentao_data/ssp/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/rollout_eval}"
N="${N:-5}"                                   # 每题采样数（n=5 与训练一致）
MAX_SAMPLES="${MAX_SAMPLES:--1}"              # 先小批验证：MAX_SAMPLES=50
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-20}"  # 验证样例表格进 mlflow artifact 的数量（0=不记）
TEMPERATURE="${TEMPERATURE:-0.0}"             # greedy
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"

cd "$EVAL_DIR"
# --config-name 选择三种模式的配置（rollout_eval_agentic / _single / _no_tool）
python3 rollout_eval.py \
    --config-name "rollout_eval_${MODE}" \
    model_path="$MODEL_PATH" \
    data.val_files="[\"${DATA_PATH}\"]" \
    output_dir="$OUTPUT_DIR" \
    n="$N" \
    max_samples="$MAX_SAMPLES" \
    log_val_generations="$LOG_VAL_GENERATIONS" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$TEMPERATURE" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    actor_rollout_ref.rollout.trace.backend="$TRACE_BACKEND"
