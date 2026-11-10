#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Interactive debug launcher for rollout_eval.
#
# 与 run_rollout_eval.sh 的区别：**前台运行**（不 setsid / 不重定向日志），
# 单卡 + 单 agent worker + 单样本，专门为了打断点调试。
#
# 用法：
#   bash run_eval_debug.sh
#   RAY_DEBUG=1 bash run_eval_debug.sh              # Ray worker 里可用 breakpoint()
#   DEBUGPY=1 bash run_eval_debug.sh                # driver 侧 debugpy，监听 :5678
#   MAX_SAMPLES=4 VAL_BATCH_SIZE=4 bash run_eval_debug.sh
#
# 断点位置（相对 $REPO_ROOT/agentic-search/src/agentic_search/training）：
#   verl/experimental/agent_loop/tool_agent_loop.py:235    <- loop 里真正的 generate
#   verl/experimental/agent_loop/tool_agent_loop.py:225    <- _handle_generating_state
#   verl/experimental/agent_loop/tool_agent_loop.py:169    <- 状态机主循环
#   verl/experimental/agent_loop/agent_loop.py:1225        <- AgentLoopWorker.generate_sequences
#   verl/workers/rollout/sglang_rollout/async_sglang_server.py:526  <- server.generate
#
# 注意：上面这些都在 **AgentLoopWorker（Ray actor）** 进程里，不是 driver。
#   - 想在终端里 pdb：必须 RAY_DEBUG=1（Ray 会把 worker 的 stdio 接到终端）
#   - 想用 IDE 远程调试：DEBUGPY=1 只能 attach 到 driver，断不到 worker
# ---------------------------------------------------------------------------
set -uo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/root/agentic-search-workspace}"
TRAINING_DIR="$REPO_ROOT/agentic-search/src/agentic_search/training"

export PATH="/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export WORKSPACE="$TRAINING_DIR"
export PYTHONPATH="$TRAINING_DIR:$TRAINING_DIR/verl:${PYTHONPATH:-}"

# pip 的 cu128 库必须排在系统 cuda-12.2 之前，否则
# libc10_cuda.so: undefined symbol: cudaGetDriverEntryPointByVersion
NV_LIBS="$(ls -d /root/miniconda3/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# ---- 调试默认值：单卡、单 worker、单样本 ----------------------------------
export MODE="${MODE:-agentic}"                                     # agentic | single | no_tool
export MODEL_PATH="${MODEL_PATH:-/root/data/models/Qwen3-1.7B}"
export DATA_PATH="${DATA_PATH:-/root/data/ssp/test.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/root/data/rollout_eval_debug}"
export TRACE_BACKEND="${TRACE_BACKEND:-null}"                      # mlflow 多进程建表会撞，调试关掉
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-sglang}"                # 本机没 vllm
export FORMAT_CHECK="${FORMAT_CHECK:-qwen3}"
export HERMES_JINJA="${HERMES_JINJA:-$(cat "$TRAINING_DIR/quarl/template/qwen3_hermes.jinja")}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
export NNODES=1
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export USE_JUDGE="${USE_JUDGE:-false}"
export ROLLOUT_N="${ROLLOUT_N:-1}"
export MAX_SAMPLES="${MAX_SAMPLES:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export LOG_VAL_GENERATIONS=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ---- Ray worker 调试开关 ---------------------------------------------------
if [[ "${RAY_DEBUG:-0}" == "1" ]]; then
    export RAY_DEBUG=1
    export RAY_DEBUG_POST_MORTEM="${RAY_DEBUG_POST_MORTEM:-1}"
fi

echo "=========================================================="
echo " rollout_eval DEBUG"
echo "   repo        : $REPO_ROOT"
echo "   model       : $MODEL_PATH"
echo "   data        : $DATA_PATH"
echo "   out         : $OUTPUT_DIR"
echo "   backend     : $ROLLOUT_BACKEND  tp=$ROLLOUT_TP  workers=$NUM_WORKERS  gpus=$N_GPUS_PER_NODE"
echo "   samples     : MAX_SAMPLES=$MAX_SAMPLES  val_batch=$VAL_BATCH_SIZE  n=$ROLLOUT_N"
echo "   RAY_DEBUG   : ${RAY_DEBUG:-0}    DEBUGPY: ${DEBUGPY:-0}"
echo "   breakpoints : training/verl/verl/experimental/agent_loop/tool_agent_loop.py:235"
echo "=========================================================="
echo

cd "$EVAL_DIR" || exit 1

# Hydra CLI 覆盖都写成字面量（走 CLI 才做类型推断，走 yaml 的 oc.env 会变字符串）
ARGS=(
    --config-name "rollout_eval_${MODE}"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "n=${ROLLOUT_N}"
    "max_samples=${MAX_SAMPLES}"
    "log_val_generations=${LOG_VAL_GENERATIONS}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.max_prompt_length=4096"
    "reward.num_workers=2"
    "reward.custom_reward_function.reward_kwargs.use_judge=${USE_JUDGE}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.val_kwargs.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}"
    "actor_rollout_ref.rollout.agent.num_workers=${NUM_WORKERS}"
    "actor_rollout_ref.rollout.max_num_seqs=64"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
    "actor_rollout_ref.rollout.prompt_length=4096"
    "actor_rollout_ref.rollout.response_length=8192"
    "actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE}"
    "actor_rollout_ref.rollout.trace.backend=${TRACE_BACKEND}"
)

# 额外 CLI 覆盖原样追加：bash run_eval_debug.sh actor_rollout_ref.rollout.enforce_eager=true
ARGS+=("$@")

if [[ "${DEBUGPY:-0}" == "1" ]]; then
    echo "[debug] debugpy listening on 0.0.0.0:${DEBUGPY_PORT:-5678} (waiting for client)"
    exec python3 -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT:-5678}" --wait-for-client \
        rollout_eval.py "${ARGS[@]}"
else
    exec python3 rollout_eval.py "${ARGS[@]}"
fi
