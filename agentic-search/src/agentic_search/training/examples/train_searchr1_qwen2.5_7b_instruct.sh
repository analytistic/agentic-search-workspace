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
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
TRAINING_DIR="$REPO_ROOT/agentic-search/src/agentic_search/training"
export WORKSPACE="$TRAINING_DIR"     # used by search_r1_config.yaml hydra searchpath

# python for training (server env; local: uv run --project agentic-search)
PYTHON="python3"

# tool config template (class_name: quarl SSPSearchTool)
TOOL_CONFIG_TEMPLATE="$REPO_ROOT/third_party/Knowledge-Graph-Paths-as-Intermediate-Supervision-for-Self-Evolving-Search-Agents/training/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml"

# ---- model / data ----
MODEL_PATH="/data/models/Qwen2.5-7B-Instruct"
DATA_PATH="/data/search-r1/train"        # parquet without suffix
TEST_DATA_PATH="/data/search-r1/test"

# ---- output ----
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-searchr1_Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/$(date +"%-m_%-d_%-H")"
SAVE_CHECKPOINT_DIR="${OUTPUT_DIR}/ckpt"
TENSORBOARD_LOG_DIR="${OUTPUT_DIR}/logs/tensorboard"

# ---- cluster ----
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6379}"
NODE_IP="${NODE_IP:-$(hostname -I | awk '{print $1}')}"
SEARCH_IP="${SEARCH_IP:-$NODE_IP}"

# ---- retrieval service (CPU mode on Ascend) ----
RETRIEVER_ENABLE="${RETRIEVER_ENABLE:-1}"
RETRIEVER_SERVER="$TRAINING_DIR/examples/retrieval/retrieval_server.py"
RETRIEVER_INDEX_PATH="/data/wiki/index/e5_HNSW64.index"
RETRIEVER_CORPUS_PATH="/data/wiki/corpus/wiki-18.jsonl"
RETRIEVER_MODEL_PATH="/data/wiki/model/intfloat/e5-base-v2"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
SEARCH_PORT="${SEARCH_PORT:-8000}"

# ---- Ascend ----
ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-6,7}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"

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
export PYTHONPATH="$TRAINING_DIR/verl:$TRAINING_DIR/quarl"

mkdir -p "$OUTPUT_DIR/logs" "$TENSORBOARD_LOG_DIR" "$SAVE_CHECKPOINT_DIR"

# tool config: copy template and point the wiki URL at the retrieval service
TOOL_CONFIG="$OUTPUT_DIR/logs/search_tool_config_rank${RANK}.yaml"
cp "$TOOL_CONFIG_TEMPLATE" "$TOOL_CONFIG"
sed -i "/wiki:/s|http://[0-9.]\+:${SEARCH_PORT}/retrieve|http://${SEARCH_IP}:${SEARCH_PORT}/retrieve|" "$TOOL_CONFIG"
echo "Tool config: $TOOL_CONFIG (wiki -> http://${SEARCH_IP}:${SEARCH_PORT}/retrieve)"

# ---- retrieval service (CPU: no --faiss_gpu) ----
start_retriever() {
    local log="$OUTPUT_DIR/logs/retriever_rank0.log"
    echo "==> starting retrieval service (CPU mode)"
    OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "$PYTHON" "$RETRIEVER_SERVER" \
        --index_path "$RETRIEVER_INDEX_PATH" \
        --corpus_path "$RETRIEVER_CORPUS_PATH" \
        --topk "$RETRIEVER_TOPK" \
        --retriever_name e5 \
        --retriever_model "$RETRIEVER_MODEL_PATH" > "$log" 2>&1 &
    for _ in $(seq 1 600); do
        if curl -fsS -o /dev/null "http://localhost:${SEARCH_PORT}/status" 2>/dev/null; then
            echo "    retrieval ready"
            return 0
        fi
        sleep 1
    done
    echo "    retrieval NOT ready; log tail:"; tail -5 "$log"
    return 1
}

if [ "$RANK" -eq 0 ] && [ "$RETRIEVER_ENABLE" = "1" ]; then
    start_retriever
fi

# ---- Ray cluster ----
if [ "$RANK" -eq 0 ]; then
    ray start --block --head --node-ip-address="$NODE_IP" --port="$MASTER_PORT" &
    if [ "$NNODES" -gt 1 ]; then sleep 120; fi
else
    ray start --block --node-ip-address="$NODE_IP" --address="${MASTER_ADDR}:${MASTER_PORT}"
fi

# ---- launch training ----
if [ "$RANK" -eq 0 ]; then
    "$PYTHON" -m quarl.main_rl \
        --config-name search_r1_config \
        data.train_files="[\"${DATA_PATH}.parquet\"]" \
        data.val_files="[\"${TEST_DATA_PATH}.parquet\"]" \
        data.train_batch_size="$BATCH_SIZE" \
        data.val_batch_size="$VAL_BATCH_SIZE" \
        data.max_prompt_length="$MAX_PROMPT_LENGTH" \
        data.max_response_length="$MAX_RESPONSE_LENGTH" \
        data.return_raw_chat=True \
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
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.format=hermes \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
        actor_rollout_ref.hybrid_engine=True \
        reward_model.reward_manager=search_r1 \
        trainer.project_name=searchr1_grpo \
        trainer.experiment_name=searchr1_grpo \
        trainer.val_before_train=True \
        trainer.val_only="$VAL_ONLY" \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        trainer.nnodes="$NNODES" \
        trainer.save_freq="$SAVE_FREQ" \
        trainer.test_freq="$TEST_FREQ" \
        trainer.default_local_dir="$SAVE_CHECKPOINT_DIR" \
        trainer.total_epochs="$TOTAL_EPOCHS"
fi
