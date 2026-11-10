#!/usr/bin/env bash
# Standalone 三模式 rollout 评估（Search-R1 / SSP 基线）。
# 不起 FSDP trainer，只起 rollout agent loop workers（verl AgentLoopManager）。
#
# 用法：
#   MODE=no_tool bash run_rollout_eval.sh                 # 用 env 默认值
#   bash run_rollout_eval.sh n=1 max_samples=64           # 临时 CLI 覆盖（原样透传）
#   MODE=agentic bash run_rollout_eval.sh                 # MODE: agentic | single | no_tool
#
# 参数来源优先级：CLI 覆盖（"$@"） > env > yaml 默认值。
# 数值/布尔类 env 由 yaml 的 `${oc.decode:${oc.env:...}}` 还原类型，
# 不会退化成字符串（见 config/rollout_eval_base.yaml 顶部注释）。
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

# ---- 解释器与 CUDA 库 ----
# 不设 PATH 会落到系统 python（没有 hydra / verl）；不设 LD_LIBRARY_PATH 时
# spawn 出的 worker 会先解析到系统 cuda 的 libcudart，而 torch 2.9.1+cu128 报
# "libc10_cuda.so: undefined symbol: cudaGetDriverEntryPointByVersion"。
PYTHON_BIN_DIR="${PYTHON_BIN_DIR:-/root/miniconda3/bin}"
export PATH="$PYTHON_BIN_DIR:$PATH"
NV_LIBS="$(ls -d "$PYTHON_BIN_DIR"/../lib/python3*/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
if [ -n "$NV_LIBS" ]; then
    export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"
fi

# ---- .env（可选：环境变量放这里，git 不追踪；未设置项用下方默认值）----
if [ -f "$EVAL_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$EVAL_DIR/.env"
    set +a
fi

# hermes chat template（与训练 launcher 一致，经 env 传给 config）
export HERMES_JINJA="${HERMES_JINJA:-$(cat "$TRAINING_DIR/quarl/template/qwen3_hermes.jinja")}"

# ---- judge（敏感配置在 .env 的 JUDGE_* 中；USE_JUDGE=false 时不会被调用）----
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://judge_host:5000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-judge_model_name}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-dummy_api_key}"

# ---- trace（verl 原生：worker 内 rollout_trace_op 装饰器逐样本异步记录）----
# 与训练共用 MLFLOW_TRACKING_URI（同一 sqlite），但独立 experiment（rollout_eval）
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////data/mlruns.db}"
export TRACE_BACKEND="${TRACE_BACKEND:-mlflow}"   # mlflow | trackio | null（null = 只留 jsonl）

# ============================================================
# eval 参数：全部 export 给 yaml（yaml 用 oc.decode 保证类型）
# 需要临时改：在命令行追加 hydra 覆盖即可，例如 n=1 max_samples=64
#             data.max_samples_per_source=50（每个 bench 各 50 题）
# ============================================================
# ---- 路径 / 模式 / 后端 ----
export MODE="${MODE:-agentic}"                          # agentic | single | no_tool
export MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3-4B}"
export DATA_PATH="${DATA_PATH:-/data/xuwentao_data/ssp/test.jsonl}"
# 每次 eval 产出一个独立子文件夹（traj_viewer.py 会把它们列成 run 列表），
# 默认落在 /data/eval 下。想复用固定目录就显式传 OUTPUT_DIR=...
export RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_DIR="${OUTPUT_DIR:-/data/eval/${RUN_NAME}}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"       # vllm | sglang
export FORMAT_CHECK="${FORMAT_CHECK:-qwen3}"            # qwen3 | qwen3_5 | search_r1

# ---- 数值（由 yaml 的 oc.decode 还原成 int/float/bool）----
export ROLLOUT_N="${ROLLOUT_N:-5}"                      # 每题采样数
export MAX_SAMPLES="${MAX_SAMPLES:--1}"                 # 只跑前 N 题（-1 = 全部）
export MAX_SAMPLES_PER_SOURCE="${MAX_SAMPLES_PER_SOURCE:-0}"  # 每个 bench 最多 N 题（0 = 不限；同训练侧字段）
export DATA_SEED="${DATA_SEED:-1}"                      # 分层抽样种子（固定 => 每次抽到同一批题）
export LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-20}" # 样例表格进 mlflow 的数量（0 = 不记）
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
# per-tool-call character budget for the observation (verl's default is 256,
# which guts every retrieved passage; training uses 8192)
export TOOL_RESPONSE_LENGTH="${TOOL_RESPONSE_LENGTH:-8192}"
export TEMPERATURE="${TEMPERATURE:-0.0}"                # 0.0 = greedy
export USE_JUDGE="${USE_JUDGE:-true}"                   # false = EM 未命中直接 0

# ---- 并行 / 显存 ----
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
export NNODES="${NNODES:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"                    # 单卡必须 1；需 world_size % tp == 0
export NUM_WORKERS="${NUM_WORKERS:-1}"                  # rollout agent loop worker 数
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
export REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-4}"

cd "$EVAL_DIR"

# CLI 参数规范化：`n=` 同时影响 rollout.n 与 val_kwargs.n 两处，
# 而这两处是各自从 env 插值的（跨 subtree 不能用 ${n}），
# 所以把 `n=X` 折成 ROLLOUT_N=X，其余参数原样透传。
PASSTHRU=()
for _a in "$@"; do
    case "$_a" in
        n=*) ROLLOUT_N="${_a#n=}"; export ROLLOUT_N ;;
        *)   PASSTHRU+=("$_a") ;;
    esac
done

# --config-name 选择三种模式的配置；剩余参数原样透传（后者覆盖前者）
# 例：bash run_rollout_eval.sh n=1 max_samples=64 \
#       reward.custom_reward_function.reward_kwargs.use_judge=false
python3 rollout_eval.py \
    --config-name "rollout_eval_${MODE}" \
    ${PASSTHRU[@]+"${PASSTHRU[@]}"}
