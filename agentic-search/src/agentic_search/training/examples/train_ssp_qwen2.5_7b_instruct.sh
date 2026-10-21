#!/usr/bin/env bash
# SSP (Search Self-Play) training launcher (self-contained, no run_common dependency).
#
# 1) optional: start retrieval service (CPU mode)
# 2) start Ray cluster
# 3) run quarl.main_rl with rl_config.yaml (sglang backend, quark multi-turn
#    tool format, quark_score reward / naive-with-prompt val reward)
#
# Notes:
# - sglang backend goes through verl FSDPSGLangShardingManager: requires
#   sglang>=0.4.9 (weight_sync) or the 0.4.6-compat patch on verl.
# - judge service: quark_score reads QUARK_BASE_URL / QUARK_MODEL.
# - Usage: edit CONFIG below, then: bash train_ssp_qwen2.5_7b_instruct.sh

set -euo pipefail

# ========================== CONFIG ==========================

# ---- paths ----
REPO_ROOT="/home/xuwentao/agentic-search-workspace"
TRAINING_DIR="$REPO_ROOT/agentic-search/src/agentic_search/training"
export WORKSPACE="$TRAINING_DIR"     # used by rl_config.yaml hydra searchpath


# python for training (server env; local: uv run --project agentic-search)
PYTHON="python3"
export PYTHONPATH="$TRAINING_DIR:$TRAINING_DIR/verl:${PYTHONPATH:-}"

# ---- model / data ----
MODEL_PATH="/data/models/Qwen2.5-7B-Instruct"
DATA_PATH="/data/xuwentao_data/ssp/train.jsonl"
TEST_DATA_PATH="/data/xuwentao_data/ssp/test.jsonl"

# ---- judge service (SSP quark_score) ----
export QUARK_BASE_URL="${QUARK_BASE_URL:-http://judge_host:5000/v1}"  # LLM-as-a-Judge URL
export QUARK_MODEL="${QUARK_MODEL:-judge_model_name}"

# ---- tool config (quark multi-turn format) ----
TOOL_CONFIG="$TRAINING_DIR/examples/tool_config/search_tool_config.yaml"

# ---- task / reward ----
# ssp (origin self-play) or quark_deep_search (search-r1-like on SSP trainer)
TASK_TYPE="${TASK_TYPE:-origin}"
if [ "${TASK_TYPE}" = "quark_deep_search" ]; then
    ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
    ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-True}"
    ROLLOUT_N="${ROLLOUT_N:-5}"
else
    ADV_ESTIMATOR="${ADV_ESTIMATOR:-gae}"
    ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-False}"
    ROLLOUT_N="${ROLLOUT_N:-1}"
fi

# ---- output ----
OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-ssp_Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/$(date +"%-m_%-d_%-H")"
SAVE_CHECKPOINT_DIR="${OUTPUT_DIR}/ckpt"
TENSORBOARD_LOG_DIR="${OUTPUT_DIR}/logs/tensorboard"

# ---- cluster ----
NNODES="${NNODES:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6379}"
NODE_IP="${NODE_IP:-127.0.0.1}"
SEARCH_IP="${SEARCH_IP:-127.0.0.1}"
SEARCH_PORT="${SEARCH_PORT:-8000}"

# ---- retrieval service (CPU mode on Ascend) ----
RETRIEVER_ENABLE="${RETRIEVER_ENABLE:-1}"
RETRIEVER_SERVER="$TRAINING_DIR/examples/retrieval/retrieval_server.py"
RETRIEVER_INDEX_PATH="/data/xuwentao_data/wiki_2018/index/wiki-18-e5-index/e5_Flat.index"
RETRIEVER_CORPUS_PATH="/data/xuwentao_data/wiki_2018/corpus/wiki_dump.jsonl"
RETRIEVER_MODEL_PATH="/data/models/e5-base-v2/"
RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"

# ---- training hyper-params ----
FSDP_TYPE="${FSDP_TYPE:-fsdp}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-128}"
PPO_MICRO_BATCH_PER_GPU="${PPO_MICRO_BATCH_PER_GPU:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
MAX_TURNS="${MAX_TURNS:-10}"
LR="${LR:-1e-6}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"
VAL_N="${VAL_N:-1}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"
VAL_ONLY="${VAL_ONLY:-False}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# ============================================================

PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
export ASCEND_VISIBLE_DEVICES

mkdir -p "$OUTPUT_DIR/logs" "$TENSORBOARD_LOG_DIR" "$SAVE_CHECKPOINT_DIR" \
    "$OUTPUT_DIR/rollout" "$OUTPUT_DIR/val"

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
        --config-name rl_config \
        quark.task_type="$TASK_TYPE" \
        algorithm.adv_estimator="$ADV_ESTIMATOR" \
        data.train_files="[\"${DATA_PATH}\"]" \
        data.val_files="[\"${TEST_DATA_PATH}\"]" \
        data.train_batch_size="$BATCH_SIZE" \
        data.val_batch_size="$VAL_BATCH_SIZE" \
        data.max_prompt_length="$MAX_PROMPT_LENGTH" \
        data.max_response_length="$MAX_RESPONSE_LENGTH" \
        data.return_raw_chat=True \
        data.filter_overlong_prompts=True \
        data.truncation='left' \
        data.shuffle=True \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.actor.optim.lr="$LR" \
        actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.strategy="$FSDP_TYPE" \
        actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH_SIZE" \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_PER_GPU" \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.use_kl_loss="$ACTOR_USE_KL_LOSS" \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_high="$CLIP_RATIO_HIGH" \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.offload_policy=True \
        actor_rollout_ref.rollout.name=sglang_async \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
        actor_rollout_ref.rollout.max_num_batched_tokens="$PPO_MAX_TOKEN_LEN_PER_GPU" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
        actor_rollout_ref.rollout.n="$ROLLOUT_N" \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_PER_GPU" \
        actor_rollout_ref.rollout.multi_turn.format=quark \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
        actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
        actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
        actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
        actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.hybrid_engine=True \
        reward_model.reward_manager=ssp \
        +custom_reward_functions.quark_score.labels=['unknown'] \
        +custom_reward_functions.quark_score.integration=sum \
        quark.diff_val_reward_fn_config.reward_model.reward_manager=naive_with_prompt \
        quark.diff_val_reward_fn_config.custom_reward_function.path="$TRAINING_DIR/quarl/reward/score/search_eval_score.py" \
        quark.diff_val_reward_fn_config.custom_reward_function.name=compute_score \
        self_play.enable=True \
        self_play.lang="${SELF_PLAY_LANG:-zh}" \
        self_play.save_freq=5 \
        self_play.use_rag_filter=True \
        self_play.use_search_terms_filter=False \
        retriever.url="http://${SEARCH_IP}:${SEARCH_PORT}/retrieve" \
        retriever.topk="$RETRIEVER_TOPK" \
        trainer.project_name=ssp \
        trainer.experiment_name=ssp_qwen2.5_7b \
        trainer.val_before_train=True \
        trainer.val_only="$VAL_ONLY" \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        trainer.nnodes="$NNODES" \
        trainer.save_freq="$SAVE_FREQ" \
        trainer.test_freq="$TEST_FREQ" \
        trainer.default_local_dir="$SAVE_CHECKPOINT_DIR" \
        trainer.total_epochs="$TOTAL_EPOCHS" \
        trainer.rollout_data_dir="$OUTPUT_DIR/rollout" \
        trainer.validation_data_dir="$OUTPUT_DIR/val"
fi
