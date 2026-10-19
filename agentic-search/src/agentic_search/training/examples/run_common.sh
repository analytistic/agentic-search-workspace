#!/bin/bash

set -xeuo pipefail
ulimit -n 65535
ulimit -u 1048576

# Raise vm.max_map_count for CUDA + Triton memory mappings (needs root)
if [ "$(cat /proc/sys/vm/max_map_count)" -lt 1048576 ]; then
    echo 1048576 > /proc/sys/vm/max_map_count 2>/dev/null || echo "WARNING: cannot raise max_map_count, may need root"
fi

# Limit Triton JIT concurrent compilation to avoid thread explosion
export TRITON_NUM_COMPILER_WORKERS=${TRITON_NUM_COMPILER_WORKERS:-1}

# Cap common CPU-side thread pools to avoid exhausting pid/thread quotas when
# Ray, tokenizers, BLAS, and Triton workers are active at the same time.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

export WANDB_API_KEY=${WANDB_API_KEY:-"YOUR_WANDB_API_KEY"}
export WANDB_ENTITY=${WANDB_ENTITY:-"YOUR_WANDB_ENTITY"}

: "${SOURCE_ACTOR_CHECKPOINT_DIR:?SOURCE_ACTOR_CHECKPOINT_DIR must be set}"
: "${SOURCE_REWARD_CHECKPOINT_DIR:?SOURCE_REWARD_CHECKPOINT_DIR must be set}"
: "${SOURCE_CRITIC_CHECKPOINT_DIR:?SOURCE_CRITIC_CHECKPOINT_DIR must be set}"
: "${DATA_PATH:?DATA_PATH must be set}"

export SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME=${SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME:-""}
export SOURCE_REWARD_CHECKPOINT_ITERATION_DIRNAME=${SOURCE_REWARD_CHECKPOINT_ITERATION_DIRNAME:-checkpoint_name}
export SOURCE_CRITIC_CHECKPOINT_ITERATION_DIRNAME=${SOURCE_CRITIC_CHECKPOINT_ITERATION_DIRNAME:-checkpoint_name}
export TEST_DATA_PATH=${TEST_DATA_PATH:-/path/to/test/data}

export OUTPUT_ROOT=${OUTPUT_ROOT:-./output}
export MODEL_NAME=${MODEL_NAME:-$(basename "$SOURCE_ACTOR_CHECKPOINT_DIR")}
export RUN_DATE=${RUN_DATE:-$(date +"%-m_%-d_%-H")}
: "${OUTPUT_SUBDIR:?OUTPUT_SUBDIR must be set}"
export OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/${RUN_DATE}"
export SAVE_CHECKPOINT_DIR="${OUTPUT_DIR}/ckpt"
export TENSORBOARD_LOG_DIR="${OUTPUT_DIR}/logs/tensorboard"
export TRAJECTORY_DIR="${OUTPUT_DIR}/trajectories"
export EXTRACTION_STATS_DIR="${OUTPUT_DIR}/extraction_stats"

export NNODES=${NNODES:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-${DEFAULT_MASTER_ADDR:-localhost}}
export MASTER_PORT=${MASTER_PORT:-6379}
export SEARCH_IP=${SEARCH_IP:-${MASTER_ADDR}}
if [ "${RANK}" -eq 0 ]; then
    DEFAULT_NODE_IP=${DEFAULT_NODE_IP_RANK0:-$(hostname -I | awk '{print $1}')}
else
    DEFAULT_NODE_IP=${DEFAULT_NODE_IP_RANK1:-$(hostname -I | awk '{print $1}')}
fi
export NODE_IP=${NODE_IP:-${DEFAULT_NODE_IP}}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/ssp_triton_cache_rank${RANK}}

normalize_proxy_scheme() {
    local proxy_var="$1"
    local proxy_value="${!proxy_var:-}"

    if [[ -n "${proxy_value}" && "${proxy_value}" != *"://"* ]]; then
        printf -v "${proxy_var}" '%s' "http://${proxy_value}"
        export "${proxy_var}"
        echo "Normalized ${proxy_var} to ${!proxy_var}"
    fi
}

for proxy_var in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; do
    normalize_proxy_scheme "$proxy_var"
done

export JUDGE_TYPE=${JUDGE_TYPE:-api}
export JUDGE_BASE_URL=${JUDGE_BASE_URL:-"http://localhost:5000/v1"}
export JUDGE_API_KEY=${JUDGE_API_KEY:-"YOUR_JUDGE_API_KEY"}
export JUDGE_MODEL=${JUDGE_MODEL:-"YOUR_JUDGE_MODEL"}
export JUDGE_HEADER_TYPE=${JUDGE_HEADER_TYPE:-default}

echo "Judge config: model=${JUDGE_MODEL}, url=${JUDGE_BASE_URL}"

export SEARCH_CHAT_TEMPLATE=${SEARCH_CHAT_TEMPLATE:-qwen2p5}
export WORKSPACE=${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}

export LOG_LEVEL=${LOG_LEVEL:-DEBUG}
export SELF_PLAY_DEBUG=${SELF_PLAY_DEBUG:-True}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-72000000}
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-72000}
export NCCL_WORK_FIFO_DEPTH=${NCCL_WORK_FIFO_DEPTH:-4194304}
export LANG=${LANG:-C.UTF-8}
export LANGUAGE=${LANGUAGE:-C.UTF-8}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}

export SEARCH_PORT=${SEARCH_PORT:-8000}
export RETRIEVER_ENABLE_ON_MASTER=${RETRIEVER_ENABLE_ON_MASTER:-1}
export RETRIEVER_GPU_IDS=${RETRIEVER_GPU_IDS:-0,1,2,3,4,5,6,7}
export RETRIEVER_SCRIPT=${RETRIEVER_SCRIPT:-${WORKSPACE}/examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py}
export RETRIEVER_INDEX_PATH=${RETRIEVER_INDEX_PATH:-/path/to/e5_Flat.index}
export RETRIEVER_CORPUS_PATH=${RETRIEVER_CORPUS_PATH:-/path/to/wiki-18.jsonl}
export RETRIEVER_MODEL_PATH=${RETRIEVER_MODEL_PATH:-/path/to/intfloat/e5-base-v2}
export RETRIEVER_TOPK=${RETRIEVER_TOPK:-3}

echo "Log level set to: $LOG_LEVEL"
echo "NNODES=${NNODES} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} NODE_IP=${NODE_IP} SEARCH_IP=${SEARCH_IP}"
echo "Retriever GPUs on rank0: ${RETRIEVER_GPU_IDS}"
echo "Triton cache dir: ${TRITON_CACHE_DIR}"

mkdir -p "$WORKSPACE"
mkdir -p "$TRITON_CACHE_DIR"

if [ ! -d "${WORKSPACE}/verl" ]; then
    echo "ERROR: verl directory not found at ${WORKSPACE}/verl"
    echo "Please clone verl: git clone https://github.com/volcengine/verl ${WORKSPACE}/verl"
    exit 1
fi

export PYTHONPATH=$WORKSPACE/verl
export REPO_DIR=$WORKSPACE/quarl
export PYTHONPATH=$PYTHONPATH:$REPO_DIR
export TENSORBOARD_DIR=${TENSORBOARD_LOG_DIR}

if [ -z "$SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME" ]; then
    ACTOR_PATH=${SOURCE_ACTOR_CHECKPOINT_DIR}
else
    ACTOR_PATH=${SOURCE_ACTOR_CHECKPOINT_DIR}/${SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME}
fi

if [ -z "$SOURCE_REWARD_CHECKPOINT_ITERATION_DIRNAME" ]; then
    REWARD_PATH=${SOURCE_REWARD_CHECKPOINT_DIR}
else
    REWARD_PATH=${SOURCE_REWARD_CHECKPOINT_DIR}/${SOURCE_REWARD_CHECKPOINT_ITERATION_DIRNAME}
fi

if [ -z "$SOURCE_CRITIC_CHECKPOINT_ITERATION_DIRNAME" ]; then
    CRITIC_PATH=${SOURCE_CRITIC_CHECKPOINT_DIR}
else
    CRITIC_PATH=${SOURCE_CRITIC_CHECKPOINT_DIR}/${SOURCE_CRITIC_CHECKPOINT_ITERATION_DIRNAME}
fi

read -ra TRAIN_PATHS <<< "$DATA_PATH"
read -ra TEST_PATHS <<< "$TEST_DATA_PATH"

TRAIN_FILES=()
for path in "${TRAIN_PATHS[@]}"; do
    TRAIN_FILES+=("${path}.parquet")
done

TEST_FILES=()
for path in "${TEST_PATHS[@]}"; do
    TEST_FILES+=("${path}.parquet")
done

TRAIN_FILES_STR="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"
TEST_FILES_STR="[$(IFS=,; echo "${TEST_FILES[*]}")]"

if [[ "${SEARCH_CHAT_TEMPLATE}" == "R-Search" ]]; then
    echo "Detected SEARCH_CHAT_TEMPLATE=${SEARCH_CHAT_TEMPLATE}, processing test files..."
    PROCESSED_DIR="./processed_test_files"
    mkdir -p "$PROCESSED_DIR"

    PROCESSED_TEST_FILES=()
    for path in "${TEST_PATHS[@]}"; do
        input_file="${path}.parquet"
        filename=$(basename "$path")
        output_file="${PROCESSED_DIR}/${filename}_${SEARCH_CHAT_TEMPLATE}.parquet"

        python3 "$WORKSPACE/examples/sglang_multiturn/search_r1_like/reprocess_data.py" \
            -i "$input_file" \
            -o "$output_file" \
            -t "${SEARCH_CHAT_TEMPLATE}"

        if [ $? -eq 0 ]; then
            PROCESSED_TEST_FILES+=("$output_file")
        else
            PROCESSED_TEST_FILES+=("$input_file")
        fi
    done

    TEST_FILES_STR="[$(IFS=,; echo "${PROCESSED_TEST_FILES[*]}")]"
    LANG=${SEARCH_CHAT_TEMPLATE}
elif [[ "${SEARCH_CHAT_TEMPLATE}" == "qwen2p5" ]] || [[ "${SEARCH_CHAT_TEMPLATE}" == "llama3p1" ]]; then
    LANG=en
else
    LANG=en
fi

export NODE_RANK=${RANK}

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$TENSORBOARD_LOG_DIR"
mkdir -p "$OUTPUT_DIR/val"
mkdir -p "$OUTPUT_DIR/rollout"
mkdir -p "$TRAJECTORY_DIR"
mkdir -p "$EXTRACTION_STATS_DIR"
mkdir -p "$SAVE_CHECKPOINT_DIR"

TOOL_CONFIG_TEMPLATE=${WORKSPACE}/examples/sglang_multiturn/config/tool_config/search_tool_config.yaml
TOOL_CONFIG=${OUTPUT_DIR}/logs/search_tool_config_rank${NODE_RANK}.yaml
cp "$TOOL_CONFIG_TEMPLATE" "$TOOL_CONFIG"
sed -i "/wiki:/s|http://[0-9.]\+:${SEARCH_PORT}/retrieve|http://${SEARCH_IP}:${SEARCH_PORT}/retrieve|" "$TOOL_CONFIG"
echo "Updated wiki search service address to: http://${SEARCH_IP}:${SEARCH_PORT}/retrieve"

TRAIN_METHOD=${TRAIN_METHOD:-grpo}
PROCESS_REWARD_ALPHA=${PROCESS_REWARD_ALPHA:-0.3}

BATCH_SIZE=${BATCH_SIZE:-256}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
PPO_MICRO_BATCH_PER_GPU=${PPO_MICRO_BATCH_PER_GPU:-1}
LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU=${LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU:-1}
REWARD_MIRCO_BATCH_SIZE_PER_GPU=${REWARD_MIRCO_BATCH_SIZE_PER_GPU:-1}
CRITIC_MIRCO_BATCH_SIZE_PER_GPU=${CRITIC_MIRCO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ACTOR_SP=${ACTOR_SP:-1}
REWARD_SP=${REWARD_SP:-1}
CRITIC_SP=${CRITIC_SP:-1}
ACTOR_FSDP_SIZE=${ACTOR_FSDP_SIZE:--1}
FSDP_TYPE=${FSDP_TYPE:-fsdp}
REWARD_MODEL_ENABLE=${REWARD_MODEL_ENABLE:-False}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.285}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
VAL_N=${VAL_N:-1}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-${DEFAULT_VAL_BATCH_SIZE:-64}}
VAL_ONLY=${VAL_ONLY:-False}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-triton}
MAX_TURNS=${MAX_TURNS:-10}

PROCESS_REWARD_MAX_SEARCH_CALLS=${PROCESS_REWARD_MAX_SEARCH_CALLS:-${MAX_TURNS}}
PROCESS_REWARD_MAX_RESPONSE_TOKENS=${PROCESS_REWARD_MAX_RESPONSE_TOKENS:-${MAX_RESPONSE_LENGTH}}
PROCESS_REWARD_NODE_FILTER_MODE=${PROCESS_REWARD_NODE_FILTER_MODE:-named_entity_only}

if [[ "${TRAIN_METHOD}" == "grpo" ]]; then
    ROLLOUT_N=5
    ADV_ESTIMATOR=grpo
    ACTOR_USE_KL_LOSS=True
    USE_KL_IN_REWARD=False
else
    ROLLOUT_N=1
    ADV_ESTIMATOR=gae
    ACTOR_USE_KL_LOSS=False
    USE_KL_IN_REWARD=True
fi

TASK_TYPE=ssp_kg_deep_search

if [[ "${TASK_TYPE}" == "ssp_kg_deep_search" ]]; then
    RM_TYPE=default
    DATASET_CLASS=null
    RM_MANAGER=ssp_kg
else
    RM_TYPE=default
    DATASET_CLASS=null
    RM_MANAGER=naive
fi

if [[ "${RM_MANAGER}" == "ssp_kg" ]]; then
    USE_PROCESS_REWARD=True
    CUSTOM_RM_ARGS="reward_model.reward_manager=${RM_MANAGER}"

    if [[ "${USE_PROCESS_REWARD}" == "True" ]]; then
        CUSTOM_RM_ARGS="${CUSTOM_RM_ARGS} \
            +custom_reward_functions.process_score.labels=['unknown'] \
            +custom_reward_functions.process_score.integration=sum \
            +custom_reward_functions.process_score.kwargs.alpha=${PROCESS_REWARD_ALPHA} \
            +custom_reward_functions.process_score.kwargs.think_only=True \
            +custom_reward_functions.process_score.kwargs.case_sensitive=False \
            +custom_reward_functions.process_score.kwargs.max_search_calls=${PROCESS_REWARD_MAX_SEARCH_CALLS} \
            +custom_reward_functions.process_score.kwargs.max_response_tokens=${PROCESS_REWARD_MAX_RESPONSE_TOKENS} \
            +custom_reward_functions.process_score.kwargs.node_filter_mode=${PROCESS_REWARD_NODE_FILTER_MODE} \
            diff_val_reward_fn_config.reward_model.reward_manager=naive_with_prompt \
            diff_val_reward_fn_config.custom_reward_function.path=$WORKSPACE/quarl/reward/score/search_eval_score.py \
            diff_val_reward_fn_config.custom_reward_function.name=compute_score"
    fi
else
    CUSTOM_RM_ARGS="reward_model.reward_manager=${RM_MANAGER}"
fi

start_retriever() {
    local retriever_log="${OUTPUT_DIR}/logs/retriever_rank0.log"
    echo "Starting GPU retriever on rank0 with CUDA_VISIBLE_DEVICES=${RETRIEVER_GPU_IDS}"

    CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU_IDS}" \
        python3 "${RETRIEVER_SCRIPT}" \
        --index_path "${RETRIEVER_INDEX_PATH}" \
        --corpus_path "${RETRIEVER_CORPUS_PATH}" \
        --topk "${RETRIEVER_TOPK}" \
        --retriever_model "${RETRIEVER_MODEL_PATH}" \
        --faiss_gpu > "${retriever_log}" 2>&1 &

    for _ in $(seq 1 90); do
        if curl -fsS "http://${SEARCH_IP}:${SEARCH_PORT}/status" >/dev/null 2>&1; then
            echo "Retriever is ready at http://${SEARCH_IP}:${SEARCH_PORT}/retrieve"
            return 0
        fi
        sleep 2
    done

    echo "Retriever failed to become ready within timeout."
    echo "Check log: ${retriever_log}"
    return 1
}

if [ "$NODE_RANK" -eq 0 ]; then
    if [ "${RETRIEVER_ENABLE_ON_MASTER}" = "1" ]; then
        start_retriever
    fi

    ray start --block --head --node-ip-address="${NODE_IP}" --port="${MASTER_PORT}" &

    if [ ! "$NNODES" -eq 1 ]; then
        sleep 120
    fi

    python3 -m quarl.main_rl \
        task_type=$TASK_TYPE \
        algorithm.adv_estimator=${ADV_ESTIMATOR} \
        data.train_files=${TRAIN_FILES_STR} \
        data.val_files=${TEST_FILES_STR} \
        data.train_batch_size=${BATCH_SIZE} \
        data.val_batch_size=${VAL_BATCH_SIZE} \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.return_raw_chat=True \
        data.filter_overlong_prompts=True \
        data.truncation='left' \
        data.shuffle=${DATA_SHUFFLE:-False} \
        data.custom_cls.name=${DATASET_CLASS} \
        actor_rollout_ref.model.path=${ACTOR_PATH} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.strategy=${FSDP_TYPE} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_PER_GPU} \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS} \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_SP} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.offload_policy=True \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=${ACTOR_FSDP_SIZE} \
        actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
        actor_rollout_ref.rollout.name=sglang_async \
        actor_rollout_ref.rollout.multi_turn.format=ssp_kg \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
        actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False \
        actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
        actor_rollout_ref.hybrid_engine=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION} \
        actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=${SGLANG_ATTENTION_BACKEND} \
        actor_rollout_ref.rollout.n=${ROLLOUT_N} \
        actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
        actor_rollout_ref.ref.strategy=${FSDP_TYPE} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
        algorithm.kl_ctrl.kl_coef=0.001 \
        algorithm.kl_penalty='low_var_kl' \
        algorithm.gamma=1.0 \
        critic.optim.lr=1e-5 \
        critic.ppo_micro_batch_size_per_gpu=${CRITIC_MIRCO_BATCH_SIZE_PER_GPU} \
        critic.strategy=${FSDP_TYPE} \
        critic.model.use_remove_padding=True \
        critic.model.path=${CRITIC_PATH} \
        critic.model.enable_gradient_checkpointing=True \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        critic.ulysses_sequence_parallel_size=${CRITIC_SP} \
        reward_model.enable=${REWARD_MODEL_ENABLE} \
        reward_model.strategy=${FSDP_TYPE} \
        reward_model.model.type=${RM_TYPE} \
        reward_model.model.path=${REWARD_PATH} \
        reward_model.model.use_remove_padding=True \
        reward_model.max_length=${PPO_MAX_TOKEN_LEN_PER_GPU} \
        reward_model.micro_batch_size_per_gpu=${REWARD_MIRCO_BATCH_SIZE_PER_GPU} \
        reward_model.model.fsdp_config.param_offload=True \
        reward_model.ulysses_sequence_parallel_size=${REWARD_SP} \
        ${CUSTOM_RM_ARGS} \
        trainer.critic_warmup=0 \
        trainer.logger=['console','tensorboard','wandb'] \
        trainer.project_name=ssp_kg_${TRAIN_METHOD} \
        trainer.experiment_name=ssp_kg_${TRAIN_METHOD} \
        trainer.val_before_train=True \
        trainer.val_only=${VAL_ONLY} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=20 \
        trainer.test_freq=20 \
        trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
        trainer.rollout_data_dir=${OUTPUT_DIR}/rollout \
        trainer.validation_data_dir=${OUTPUT_DIR}/val \
        trainer.total_epochs=${TOTAL_EPOCHS} \
        actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
        actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE} \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE} \
        +actor_rollout_ref.rollout.val_kwargs.frequency_penalty=0 \
        +actor_rollout_ref.rollout.val_kwargs.repetition_penalty=1 \
        +actor_rollout_ref.rollout.repetition_penalty=1 \
        self_play.enable=True \
        self_play.lang=${LANG} \
        self_play.save_freq=20 \
        self_play.use_rag_filter=True \
        self_play.noisy_RAG_materials=4 \
        self_play.proposer.enable=True \
        self_play.proposer.warm_up_steps=-1 \
        self_play.proposer.format_penalty=0 \
        self_play.proposer.n=1 \
        self_play.proposer.reward_type=1-acc \
        self_play.proposer.adv_estimator=grpo \
        self_play.proposer.right=1.0 \
        self_play.proposer.left=0.0 \
        self_play.extraction_failure.strategy=reuse \
        self_play.solver.enable=True \
        self_play.dynamic_sampling.enable=False \
        self_play.reward_dynamic_sampling.enable=False \
        self_play.use_search_terms_filter=False \
        self_play.extraction_failure.reuse_success_rate_threshold=1.0 \
        self_play.combine_update=False \
        self_play.mini_epochs=1 \
        self_play.extraction_failure.pool_clear_interval=10 \
        self_play.answer_pattern=question \
        self_play.validate_config=True \
        self_play.extraction_failure.keep_ratio=0
else
    ray start --block --node-ip-address="${NODE_IP}" --address="${MASTER_ADDR}:${MASTER_PORT}"
fi
