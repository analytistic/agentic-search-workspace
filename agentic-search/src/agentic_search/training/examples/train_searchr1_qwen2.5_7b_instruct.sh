#!/bin/bash

set -xeuo pipefail
ulimit -n 65535

export SOURCE_ACTOR_CHECKPOINT_DIR=/path/to/Qwen2.5-7B-Instruct
export SOURCE_REWARD_CHECKPOINT_DIR=/path/to/Qwen2.5-7B-Instruct
export SOURCE_CRITIC_CHECKPOINT_DIR=/path/to/Qwen2.5-7B-Instruct

export DATA_PATH=/path/to/processed    # path to .parquet (without .parquet suffix)
export OUTPUT_SUBDIR=ssp_kg_${MODEL_NAME:-Qwen2.5-7B-Instruct}

export DEFAULT_MASTER_ADDR=${DEFAULT_MASTER_ADDR:-localhost}
export DEFAULT_NODE_IP_RANK0=${DEFAULT_NODE_IP_RANK0:-$(hostname -I | awk '{print $1}')}
export DEFAULT_NODE_IP_RANK1=${DEFAULT_NODE_IP_RANK1:-$(hostname -I | awk '{print $1}')}
export DEFAULT_VAL_BATCH_SIZE=256

export PROCESS_REWARD_ALPHA=${PROCESS_REWARD_ALPHA:-0.3}
export PROCESS_REWARD_MAX_SEARCH_CALLS=${PROCESS_REWARD_MAX_SEARCH_CALLS:-10}
export PROCESS_REWARD_MAX_RESPONSE_TOKENS=${PROCESS_REWARD_MAX_RESPONSE_TOKENS:-8192}
export PROCESS_REWARD_NODE_FILTER_MODE=${PROCESS_REWARD_NODE_FILTER_MODE:-named_entity_only}

source "$(dirname "${BASH_SOURCE[0]}")/run_common.sh"
