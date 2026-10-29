#!/usr/bin/env bash
# Search-R1 training launcher, hermes protocol variant (JSON <tool_call> tool calls).
# Same pipeline as train_searchr1_qwen2.5_7b_instruct.sh, but uses
# search_r1_config.yaml (hermes overrides injected by this launcher): format=hermes (Qwen-native <tool_call>{json}<//tool_call>), qwen3_5_hermes.jinja (unified system prompt with think/answer rules), and data.clean_question=True (strips the Search-R1 long instructions from prompts).
#
# 1) optional: start retrieval service (CPU mode)
# 2) start Ray cluster
# 3) run quarl.main_rl with search_r1_config.yaml (vllm backend, hermes
#    tool parser, search_r1 EM reward)
#
# Usage: edit CONFIG below, then: bash train_searchr1_qwen2.5_7b_instruct.sh

set -euo pipefail


# ========================== CONFIG ==========================

# ---- paths ----
REPO_ROOT="/home/xuwentao/agentic-search-workspace"
TRAINING_DIR="$REPO_ROOT/agentic-search/src/agentic_search/training"
export WORKSPACE="$TRAINING_DIR"     # used by search_r1_config.yaml hydra searchpath


# python for training (server env; local: uv run --project agentic-search)
PYTHON="python3"
export PYTHONPATH="$TRAINING_DIR:$TRAINING_DIR/verl:${PYTHONPATH:-}"

# custom chat template: passed via env var (Hydra CLI overrides cannot lex the
# jinja text); consumed in the config as ${oc.env:SEARCH_R1_JINJA}
export HERMES_JINJA="${HERMES_JINJA:-$(cat "$TRAINING_DIR/quarl/template/qwen3_5_hermes.jinja")}"

# ---- model / data ----
MODEL_PATH="/data/models/Qwen3.5-4B"   # Qwen3.5-4B (native <think> + <tool_call> JSON tool calling)
DATA_PATH="/data/xuwentao_data/search_r1/train.parquet"        # parquet without suffix
TEST_DATA_PATH="/data/xuwentao_data/ssp/test.jsonl"

# ---- output (checkpoints are large: 7B bf16 + fp32 optimizer ~40-50GB each; keep on /data) ----
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-searchr1_Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/$(date +"%-m_%-d_%-H")"
SAVE_CHECKPOINT_DIR="${OUTPUT_DIR}/ckpt"
TENSORBOARD_LOG_DIR="${OUTPUT_DIR}/logs/tensorboard"

# ---- judge service (validation scoring: search_eval_score -> llm_as_a_judge) ----
# Required even if the service is down: outcome_score raises ValueError when
# JUDGE_BASE_URL/JUDGE_MODEL are unset; with the vars set, a down service
# simply yields judge score 0 (EM/F1 parts still work).
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://judge_host:5000/v1}"
export JUDGE_MODEL="${JUDGE_MODEL:-judge_model_name}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-dummy_api_key}"

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
RETRIEVER_MODEL_PATH="/data/models/e5-base-v2/"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
# Pin both visibility vars: faiss_ascend reads ASCEND_VISIBLE_DEVICES, torch_npu prefers ASCEND_RT_VISIBLE_DEVICES.
RETRIEVER_NPU_DEVICES="${RETRIEVER_NPU_DEVICES:-1}"   # single NPU for retrieval (die 1)
export SEARCH_PORT="${SEARCH_PORT:-8000}"

# ---- Ascend (8 training cards die 8-15, retriever die 1) ----
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"   # 8 training cards (die 8-15)
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"   # torch_npu pin, must match ASCEND_VISIBLE_DEVICES
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ---- training hyperparams ----
# 8 training cards: 256*5=1280, 128*5=640, 64*5=320 all divisible by 8.
BATCH_SIZE="${BATCH_SIZE:-256}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
ROLLOUT_N="${ROLLOUT_N:-5}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"   # 7B fits one 64GB NPU; TP=1 avoids cross-card TP comm
MAX_TURNS="${MAX_TURNS:-10}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"   # 64*5=320 % 10 = 0
VAL_ONLY="${VAL_ONLY:-False}"

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
# torch_npu resolves its device ids from ASCEND_RT_VISIBLE_DEVICES; without it
# the pin above is silently ignored by torch_npu (faiss_ascend only reads the
# other one), so both must be exported.
export ASCEND_RT_VISIBLE_DEVICES

mkdir -p "$OUTPUT_DIR/logs" "$TENSORBOARD_LOG_DIR" "$SAVE_CHECKPOINT_DIR"

# tool URL is resolved from SEARCH_IP/SEARCH_PORT by quarl/tool/search_tool_fn.py

# ---- retrieval service (NPU: single-card faiss) ----
start_retriever() {
    # Kill stale retriever instances (SIGKILL; SIGTERM may not die fast enough).
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:"$SEARCH_PORT" | xargs -r kill -9 2>/dev/null || true
    else
        pkill -9 -f "retrieval_server.py" 2>/dev/null || true
    fi
    # Wait until the port is actually released (max 10s).
    for _ in $(seq 1 10); do
        if command -v lsof >/dev/null 2>&1 && ! lsof -ti:"$SEARCH_PORT" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    echo "==> starting retrieval service (npu mode on $RETRIEVER_NPU_DEVICES, logs to $OUTPUT_DIR/logs/retriever_rank0.log)"
    # Pin BOTH visibility vars (faiss_ascend reads ASCEND_VISIBLE_DEVICES,
    # torch_npu prefers ASCEND_RT_VISIBLE_DEVICES). Command-prefix env (no
    # export) so the training launch below keeps its own visibility.
    ASCEND_VISIBLE_DEVICES="${RETRIEVER_NPU_DEVICES}" ASCEND_RT_VISIBLE_DEVICES="${RETRIEVER_NPU_DEVICES}" \
        OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "$PYTHON" "$RETRIEVER_SERVER" \
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL_PATH" \
        --cache-entries 10000 \
        --faiss-device npu \
        > "$OUTPUT_DIR/logs/retriever_rank0.log" 2>&1 &
    return 0
}

if [ "$RANK" -eq 0 ] && [ "$RETRIEVER_ENABLE" = "1" ]; then
    start_retriever
fi

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
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////data/mlruns.db}"
export MLFLOW_DB_PATH="${MLFLOW_DB_PATH:-/data/mlruns.db}"
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
    "$PYTHON" -m quarl.main_rl \
        --config-name search_r1_config \
        actor_rollout_ref.model.custom_chat_template='${oc.env:HERMES_JINJA,null}' \
        data.clean_question=true \
        data.train_files="[\"${DATA_PATH}\"]" \
        data.val_files="[\"${TEST_DATA_PATH}\"]" \
        data.train_batch_size="$BATCH_SIZE" \
        data.val_batch_size="$VAL_BATCH_SIZE" \
        data.max_prompt_length="$MAX_PROMPT_LENGTH" \
        data.max_response_length="$MAX_RESPONSE_LENGTH" \
        data.filter_overlong_prompts=True \
        data.truncation='left' \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr="$LR" \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
        actor_rollout_ref.actor.clip_ratio_high=0.285 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
        actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.n="$ROLLOUT_N" \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
        actor_rollout_ref.rollout.agent.num_workers=10 \
        actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
        actor_rollout_ref.hybrid_engine=True \
        diff_val_reward_fn_config.custom_reward_function.path="$TRAINING_DIR/quarl/reward/score/search_eval_score.py" \
        trainer.project_name=searchr1_hermes \
        trainer.experiment_name="searchr1_hermes_qwen3_5_4b_$(date +%m_%d_%H)" \
        trainer.logger='["console","mlflow"]' \
        trainer.val_before_train=True \
        trainer.val_only="$VAL_ONLY" \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        trainer.nnodes="$NNODES" \
        trainer.save_freq="$SAVE_FREQ" \
        trainer.test_freq="$TEST_FREQ" \
        trainer.default_local_dir="$SAVE_CHECKPOINT_DIR" \
        trainer.total_epochs="$TOTAL_EPOCHS"
fi
