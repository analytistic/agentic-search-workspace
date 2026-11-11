#!/usr/bin/env bash
# Search-R1 training with Qwen3-1.7B and Hermes JSON tool calls.
# Uses qwen3_hermes.jinja, the Hermes parser, and sglang on a 2x4090D (CUDA) box.
#
# The retrieval service is NOT started here -- run it once, separately:
#   bash "$TRAINING_DIR/examples/start_retriever.sh"   # dense e5 + HNSW, CPU faiss, :8000
#
# Usage: bash train_searchr1_hermes_qwen3_1.7b.sh  [hydra CLI overrides...]

set -euo pipefail

# ========================== CONFIG ==========================

# ---- paths ----
REPO_ROOT="${REPO_ROOT:-/root/agentic-search-workspace}"
TRAINING_DIR="$REPO_ROOT/agentic-search/src/agentic_search/training"
export WORKSPACE="$TRAINING_DIR"     # used by search_r1_config.yaml hydra searchpath


# python for training (server env; local: uv run --project agentic-search)
# the conda base env holds hydra/verl/torch; the system python3 has no torch
PYTHON="${PYTHON:-/root/miniconda3/bin/python3}"
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="$TRAINING_DIR:$TRAINING_DIR/verl:${PYTHONPATH:-}"
# 4090/4090D are GeForce cards: no CUDA peer access over PCIe, so NCCL must run
# without P2P. Without this, FSDP init dies with
#   "Cuda failure 217 'peer access is not supported between these two devices'"
# (eval never hits it: each sglang replica owns one card and never talks to the
# other one, so nothing crosses the PCIe link).
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"   # no InfiniBand on this box
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"   # CUDA IPC cumem vs. peer access
export NCCL_ALGO="${NCCL_ALGO:-Ring}"           # Ring needs less P2P than Tree
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"         # verl's runtime_env sets WARN; ours wins

# custom chat template: passed via env var (Hydra CLI overrides cannot lex the
# jinja text); consumed in the config as ${oc.env:HERMES_JINJA}
export HERMES_JINJA="${HERMES_JINJA:-$(cat "$TRAINING_DIR/quarl/template/qwen3_hermes.jinja")}"

# ---- model / data ----
MODEL_PATH="${MODEL_PATH:-/root/data/models/Qwen3-1.7B}"   # Qwen3-1.7B (native <think> + hermes JSON <tool_call> tool calling)
DATA_PATH="${DATA_PATH:-/root/data/search_r1/train.parquet}"        # parquet without suffix
TEST_DATA_PATH="${TEST_DATA_PATH:-/root/data/ssp/test.jsonl}"

# ---- output ----
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/train_runs}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-searchr1_hermes_Qwen3-1.7B}"
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/$(date +"%-m_%-d_%-H")"
SAVE_CHECKPOINT_DIR="${OUTPUT_DIR}/ckpt"
# read back by the config as ${oc.env:SAVE_CHECKPOINT_DIR} / ${oc.env:EXPERIMENT_NAME}
export SAVE_CHECKPOINT_DIR
export EXPERIMENT_NAME="searchr1_hermes_qwen3_1.7b_$(date +%m_%d_%H)"
TENSORBOARD_LOG_DIR="${OUTPUT_DIR}/logs/tensorboard"

# ---- judge service (validation scoring: search_eval_score -> llm_as_a_judge) ----
# Required even if the service is down: outcome_score raises ValueError when
# JUDGE_BASE_URL/JUDGE_MODEL are unset; with the vars set, a down service
# simply yields judge score 0 (EM/F1 parts still work).
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://judge_host:5000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-judge_model_name}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-dummy_api_key}"
# no judge service reachable here: EM misses score 0 instead of calling out
export USE_JUDGE="${USE_JUDGE:-false}"

# ---- experiment tracking (unified mlflow: metrics + traces, one UI) ----
# metrics: trainer.logger mlflow backend; traces: rollout.trace mlflow backend

# ---- cluster ----
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6379}"
NODE_IP="${NODE_IP:-127.0.0.1}"
# exported so the search function tool (quarl/tool/search_tool_fn.py) and the
# retrieval service on this node agree on the URL
export SEARCH_IP="${SEARCH_IP:-127.0.0.1}"

# ---- retrieval service (NPU) ----
RETRIEVER_ENABLE="${RETRIEVER_ENABLE:-1}"
RETRIEVER_SERVER="$TRAINING_DIR/examples/retrieval/retrieval_server.py"
RETRIEVER_INDEX_PATH="/data/xuwentao_data/wiki-18-e5-index/index"
RETRIEVER_CORPUS_PATH="/data/xuwentao_data/wiki_2018/wiki-18.jsonl"
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-/root/data/models/e5-base-v2}"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
# NPU-only note (the retriever now runs standalone on CPU, see start_retriever.sh).
RETRIEVER_NPU_DEVICES="${RETRIEVER_NPU_DEVICES:-1}"   # single NPU for retrieval (die 1)
export SEARCH_PORT="${SEARCH_PORT:-8000}"

# ---- devices: 2x 4090D (CUDA). The ASCEND_* vars are kept only so the old
# (now commented-out) NPU retriever block still parses; torch ignores them here. ----
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1}"   # CUDA box: 2x 4090D (die 0,1)
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"   # torch_npu pin (ignored on CUDA)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"

# ---- training hyperparams ----
# 2 cards: train_batch 16 * n=5 = 80 rollouts, mini_batch 8 * n=5 = 40 -> both
# divisible by the 2 GPUs. Raise BATCH_SIZE only if the GPU memory allows it.
BATCH_SIZE="${BATCH_SIZE:-16}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
ROLLOUT_N="${ROLLOUT_N:-5}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"   # 1.7B fits on one 24G card; TP=1 avoids cross-card comm
MAX_TURNS="${MAX_TURNS:-10}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"   # val pass only: 8 * n=5 = 40 rollouts
VAL_ONLY="${VAL_ONLY:-False}"
# Pre-training validation is OFF by default: it costs a full validation
# rollout pass over VAL_BATCH_SIZE before step 1, which is pure overhead when
# the model is still untrained. Set VAL_BEFORE_TRAIN=True to get the baseline
# metrics at step 0.
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
# verl only evaluates `val_only` INSIDE the `val_before_train` branch
# (verl/trainer/ppo/v1/trainer_base.py:421), so VAL_ONLY=True together with
# VAL_BEFORE_TRAIN=False would silently start training instead of exiting
# after the validation pass. Force it on in that case.
if [ "$VAL_ONLY" = "True" ] && [ "$VAL_BEFORE_TRAIN" != "True" ]; then
    echo "[WARN] VAL_ONLY=True requires val_before_train=True; enabling val_before_train."
    VAL_BEFORE_TRAIN=True
fi

# ============================================================

# NOTE: use_dynamic_bsz is OFF for now (see search_r1_config.yaml), so the
# actor's token budget below is NOT read by the trainer. It is kept because
# re-enabling dynamic bsz should start from 2x, and the variable documents the
# value to use then. Fixed micro batches live in the yaml:
#   actor.ppo_micro_batch_size_per_gpu=4
#   rollout/ref.log_prob_micro_batch_size_per_gpu=2
PPO_MAX_TOKEN_LEN_PER_GPU=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
# vLLM keeps its own cap: the actor budget above is for fwd/bwd packing only;
# feeding 2x to the engine just inflates prefill batches and KV-cache peaks.
ROLLOUT_MAX_NUM_BATCHED_TOKENS=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export ASCEND_VISIBLE_DEVICES
# torch_npu (not used on this box) resolves its device ids from
# ASCEND_RT_VISIBLE_DEVICES; the real pin here is CUDA_VISIBLE_DEVICES above.
export ASCEND_RT_VISIBLE_DEVICES

mkdir -p "$OUTPUT_DIR/logs" "$TENSORBOARD_LOG_DIR" "$SAVE_CHECKPOINT_DIR"

# tool URL is resolved from SEARCH_IP/SEARCH_PORT by quarl/tool/search_tool_fn.py

# NOTE(2026-11-11): the retrieval service is NOT started by this launcher any more.
#   Start it once, separately, BEFORE training:
#       bash "$TRAINING_DIR/examples/start_retriever.sh"   # dense e5 + HNSW, CPU faiss, :8000
#   The old in-launcher (NPU) startup is kept below for reference but is disabled:
#   every training run restarted the service, which reloads the ~70G index and
#   drops the warm page cache. The agent talks to it over
#   http://$SEARCH_IP:$SEARCH_PORT/ via quarl/tool/search_tool_fn.py.
# # ---- retrieval service (NPU: single-card faiss) ----
# start_retriever() {
#     # Kill stale retriever instances (SIGKILL; SIGTERM may not die fast enough).
#     if command -v lsof >/dev/null 2>&1; then
#         lsof -ti:"$SEARCH_PORT" | xargs -r kill -9 2>/dev/null || true
#     else
#         pkill -9 -f "retrieval_server.py" 2>/dev/null || true
#     fi
#     # Wait until the port is actually released (max 10s).
#     for _ in $(seq 1 10); do
#         if command -v lsof >/dev/null 2>&1 && ! lsof -ti:"$SEARCH_PORT" >/dev/null 2>&1; then
#             break
#         fi
#         sleep 1
#     done
#     echo "==> starting retrieval service (npu mode on $RETRIEVER_NPU_DEVICES, logs to $OUTPUT_DIR/logs/retriever_rank0.log)"
#     # Pin BOTH visibility vars (faiss_ascend reads ASCEND_VISIBLE_DEVICES,
#     # torch_npu prefers ASCEND_RT_VISIBLE_DEVICES). Command-prefix env (no
#     # export) so the training launch below keeps its own visibility.
#     ASCEND_VISIBLE_DEVICES="${RETRIEVER_NPU_DEVICES}" ASCEND_RT_VISIBLE_DEVICES="${RETRIEVER_NPU_DEVICES}" \
#         OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "$PYTHON" "$RETRIEVER_SERVER" \
#         --index_path "$RETRIEVER_INDEX_PATH" \
#         --corpus_path "$RETRIEVER_CORPUS_PATH" \
#         --topk "$RETRIEVER_TOPK" \
#         --retriever_name e5 \
#         --retriever_model "$RETRIEVER_MODEL_PATH" \
#         --cache-entries 10000 \
#         --faiss-device npu \
#         > "$OUTPUT_DIR/logs/retriever_rank0.log" 2>&1 &
#     return 0
# }
#
# if [ "$RANK" -eq 0 ] && [ "$RETRIEVER_ENABLE" = "1" ]; then
#     start_retriever
# fi

# ---- mlflow trace db (pre-init so AgentLoopWorkers don't race the alembic migration) ----
# mlflow creates a TrackingServiceClient (and runs alembic upgrade) in every new
# process; upgrade is idempotent (driven by the alembic_version table), so a
# fully-migrated DB makes all later worker inits a no-op. Do the whole migration
# in this single process and verify the version stamp before proceeding.
# NOTE: workers read MLFLOW_TRACKING_URI (rollout_trace.py) and default to
# "sqlite:////tmp/mlruns.db" (4 slashes = ABSOLUTE path). Keep the db under
# /data (big, survives reboots) and pre-init the exact same URI: 3 slashes
# ("sqlite:///tmp/mlruns.db") points at ${CWD}/tmp/mlruns.db -- a DIFFERENT
# file, so workers would race on a fresh DB again. Export the variable so
# every process agrees.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////root/data/mlruns.db}"
export MLFLOW_DB_PATH="${MLFLOW_DB_PATH:-/root/data/mlruns.db}"
if [ "$RANK" -eq 0 ]; then
    mkdir -p "$(dirname "$MLFLOW_DB_PATH")"
    # Do NOT rm the DB: once migrated, every new process's TrackingServiceClient
    # runs alembic upgrade which is a no-op (driven by the alembic_version stamp),
    # so all worker inits just connect. Only a MISSING db needs the single-process
    # migration below (idempotent; retries until the version stamp exists).
    "$PYTHON" - <<'EOF'
import mlflow, os, sqlite3, sys

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
db_path = os.environ["MLFLOW_DB_PATH"]
for attempt in range(5):
    try:
        # A deleted experiment cannot be set active; restore it first so the
        # existing history (traces/metrics) stays under the same name.
        exp = mlflow.get_experiment_by_name('searchr1_grpo')
        if exp is not None and exp.lifecycle_stage == 'deleted':
            mlflow.restore_experiment(exp.experiment_id)
            print(f'restored deleted experiment searchr1_grpo (id={exp.experiment_id})')
        mlflow.set_experiment('searchr1_grpo')  # triggers TrackingServiceClient -> alembic upgrade to heads
        con = sqlite3.connect(db_path)
        version = con.execute('SELECT version_num FROM alembic_version').fetchone()
        con.close()
        if version is not None:
            print(f'mlflow db ready ({db_path}, alembic version={version[0]})')
            break
    except Exception as e:
        print(f'mlflow init attempt {attempt}: {e}', file=sys.stderr)
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
else:
    sys.exit('mlflow db init failed after retries')
EOF
fi

# ---- Ray cluster ----
ray stop || true
# Ray natively detects Ascend NPUs as "NPU" resources; verl requests them
# when trainer.device=npu (see launch args). No --num-gpus needed.
if [ "$RANK" -eq 0 ]; then
    ray start --block --head --node-ip-address="$NODE_IP" --port="$MASTER_PORT" &
    if [ "$NNODES" -gt 1 ]; then sleep 120; fi
else
    ray start --block --node-ip-address="$NODE_IP" --address="${MASTER_ADDR}:${MASTER_PORT}"
fi
export PYTHONPATH="$TRAINING_DIR:$TRAINING_DIR/verl:${PYTHONPATH:-}"
# ---- launch training ----
if [ "$RANK" -eq 0 ]; then
    # All settings live in search_r1_config_cuda_1.7b.yaml: hydra CLI overrides
    # for trainer.* / actor_rollout_ref.model.override_config.* do NOT survive
    # the trip into the trainer on this verl build (they silently revert to the
    # yaml defaults, i.e. the Ascend ones). The launcher therefore only exports
    # env knobs that the config interpolates.
    "$PYTHON" -m quarl.main_rl \
        --config-name search_r1_config_cuda_1.7b
fi
