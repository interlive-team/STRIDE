#!/bin/bash
#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
# STRIDE — ET-Bench trigger-detection evaluation launcher.
#
# Runs streaming trigger detection over ET-Bench (data-parallel across GPUs),
# then scores the temporal-grounding metrics.
#
# Usage:
#   bash scripts/etbench.sh <MODEL_PATH>
#
#   <MODEL_PATH>  HF repo id or local path of the STRIDE checkpoint to evaluate.
#
# Required env (dataset locations):
#   ANNO_PATH    ET-Bench annotation JSON (e.g. etbench_txt_v1.0.json)
#   DATA_PATH    Root directory containing the ET-Bench videos
#
# Optional env:
#   PRED_PATH    Output dir for the prediction DB   (default: outputs/etbench)
#   DB_NAME      Shelve DB name                      (default: etbench)
#   TASKS        Comma-separated tasks               (default: tvg,epm,tal,dvc,slc)
#   NUM_GPUS     Data-parallel world size            (default: 8)
#
set -e

MODEL_PATH="${1:?Usage: bash scripts/etbench.sh <MODEL_PATH>}"

ANNO_PATH="${ANNO_PATH:?Set ANNO_PATH to the ET-Bench annotation JSON}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the ET-Bench video root}"
PRED_PATH="${PRED_PATH:-outputs/etbench}"
DB_NAME="${DB_NAME:-etbench}"
TASKS="${TASKS:-tvg,epm,tal,dvc,slc}"
NUM_GPUS="${NUM_GPUS:-8}"

export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-102400}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

declare -a ARGS=(
    # Model
    --model_type qwen3_stride
    --model_path "$MODEL_PATH"

    # Data
    --anno_path "$ANNO_PATH"
    --data_path "$DATA_PATH"
    --pred_path "$PRED_PATH"
    --db_name "$DB_NAME"
    --tasks "$TASKS"
    --event_mode default

    # Trigger (window-shift masked diffusion)
    --chunk_size 32
    --max_window_size 128
    --stride 32
    --unmasking_steps 8
    --confidence_threshold 0.75
)

# 1) Trigger detection (data-parallel over GPUs)
uv run torchrun --nproc_per_node "$NUM_GPUS" -m eval.task.etbench.bench "${ARGS[@]}"

# 2) Scoring (temporal-grounding F1 per task)
uv run python -m eval.task.etbench.score \
    --anno_path "$ANNO_PATH" \
    --db_name "$PRED_PATH/$DB_NAME" \
    --tasks "$TASKS"
