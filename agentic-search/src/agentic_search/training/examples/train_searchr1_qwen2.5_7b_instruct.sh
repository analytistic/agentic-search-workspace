#!/usr/bin/env bash
# Search-R1 training launcher (self-contained, no run_common dependency).
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

# ---- model / data ----
MODEL_PATH="/data/models/Qwen2.5-7B-Instruct"
DATA_PATH="/data/xuwentao_data/search_r1/train.parquet"        # parquet without suffix
TEST_DATA_PATH="/data/xuwentao_data/ssp/test.jsonl"

# ---- output ----
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-searchr1_Qwen2.5-7B-Instruct}"
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

# ---- experiment tracking (offline wandb; sync later with `wandb sync`) ----
export WANDB_MODE="${WANDB_MODE:-offline}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$TENSORBOARD_LOG_DIR}"   # verl tensorboard backend writes here

# ---- cluster ----
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6379}"
NODE_IP="${NODE_IP:-127.0.0.1}"
# exported so the search function tool (quarl/tool/search_tool_fn.py) and the
# retrieval service on this node agree on the URL
export SEARCH_IP="${SEARCH_IP:-127.0.0.1}"

# ---- retrieval service (CPU mode on Ascend) ----
RETRIEVER_ENABLE="${RETRIEVER_ENABLE:-1}"
RETRIEVER_SERVER="$TRAINING_DIR/examples/retrieval/retrieval_server.py"
RETRIEVER_INDEX_PATH="/data/xuwentao_data/wiki_2018/index/wiki-18-e5-index/e5_Flat.index"
RETRIEVER_CORPUS_PATH="/data/xuwentao_data/wiki_2018/corpus/wiki_dump.jsonl"
RETRIEVER_MODEL_PATH="/data/models/e5-base-v2/"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
export SEARCH_PORT="${SEARCH_PORT:-8000}"

# ---- Ascend ----
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-12,13,14,15}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"

# ---- training hyperparams ----
BATCH_SIZE="${BATCH_SIZE:-256}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
ROLLOUT_N="${ROLLOUT_N:-5}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
MAX_TURNS="${MAX_TURNS:-10}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
LR="${LR:-1e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
VAL_ONLY="${VAL_ONLY:-False}"

# ============================================================

PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export ASCEND_VISIBLE_DEVICES

mkdir -p "$OUTPUT_DIR/logs" "$TENSORBOARD_LOG_DIR" "$SAVE_CHECKPOINT_DIR"

# tool URL is resolved from SEARCH_IP/SEARCH_PORT by quarl/tool/search_tool_fn.py

# ---- retrieval service (CPU: no --faiss_gpu) ----
start_retriever() {
    local log="$OUTPUT_DIR/logs/retriever_rank0.log"
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
    echo "==> starting retrieval service (CPU mode)"
    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "$PYTHON" "$RETRIEVER_SERVER" \
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL_PATH" > "$log" 2>&1 &
    return 0
}

if [ "$RANK" -eq 0 ] && [ "$RETRIEVER_ENABLE" = "1" ]; then
    start_retriever
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
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
        actor_rollout_ref.actor.clip_ratio_high=0.285 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
        actor_rollout_ref.rollout.max_num_batched_tokens="$PPO_MAX_TOKEN_LEN_PER_GPU" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.n="$ROLLOUT_N" \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
        actor_rollout_ref.hybrid_engine=True \
        diff_val_reward_fn_config.custom_reward_function.path="$TRAINING_DIR/quarl/reward/score/search_eval_score.py" \
        trainer.project_name=searchr1_grpo \
        trainer.experiment_name=searchr1_grpo \
        trainer.logger='["console","tensorboard"]' \
        trainer.val_before_train=True \
        trainer.val_only="$VAL_ONLY" \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        trainer.nnodes="$NNODES" \
        trainer.save_freq="$SAVE_FREQ" \
        trainer.test_freq="$TEST_FREQ" \
        trainer.default_local_dir="$SAVE_CHECKPOINT_DIR" \
        trainer.total_epochs="$TOTAL_EPOCHS"
fi
