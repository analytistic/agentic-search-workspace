#!/usr/bin/env bash

# KG self-play launcher. Set MODEL_PATH, TRAIN_FILES and VAL_FILES before running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_NAME="${CONFIG_NAME:-kg_self_play_config}"
export PROJECT_NAME="${PROJECT_NAME:-quarl-kg-self-play}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-kg-self-play}"

exec "${SCRIPT_DIR}/run_common.sh" "$@"
