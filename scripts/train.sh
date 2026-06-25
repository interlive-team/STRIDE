#!/bin/bash
#
# Copyright (C) 2025 InterLive Team. All Rights Reserved.
#
# STRIDE — activation model training launcher.
#
# Usage:
#   bash scripts/train.sh <DATA_PATH>
#
#   <DATA_PATH>  Path to the activation training data (.jsonl) produced by
#                scripts/prepare_activation_dataset.py
#
set -e

DATA_PATH="${1:?Usage: bash scripts/train.sh <DATA_PATH>}"

export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-102400}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NUM_GPUS="${NUM_GPUS:-8}"

declare -a ARGS=(
    # Model Configuration
    --model_name_or_path Qwen/Qwen3-VL-2B-Instruct
    --model_type qwen3_vl
    --model_max_length 4096000
    --tune_embed True
    --tune_lang True
    --tune_proj False
    --tune_vis False

    # Trigger Window
    --trigger_window_past 256

    # Data
    --data_path "$DATA_PATH"
    --video_min_frames_per_clip 2
    --video_max_total_frames 512
    --video_max_fps 1.0
    --video_frame_multiple 2
    --dataloader_num_workers 4
    --dataloader_prefetch_factor 2
    --dataloader_drop_last True

    # Training Hyperparameters
    --num_train_epochs 1
    --per_device_train_batch_size 8
    --gradient_accumulation_steps 4
    --learning_rate 1e-5
    --lr_embed 3e-5
    --weight_decay 0
    --max_grad_norm 1
    --warmup_steps 512
    --lr_scheduler_type cosine
    --mask_modes 1 2 3

    # Runtime & Optimization
    --bf16 True
    --deepspeed configs/deepspeed_zero2.json
    --gradient_checkpointing True

    # Logging & Saving
    --report_to none
    --eval_strategy no
    --logging_steps 1
    --save_strategy steps
    --save_steps 1000
    --save_total_limit 2
)

# Execution
uv run deepspeed --num_gpus "$NUM_GPUS" stride/train/run.py \
    "${ARGS[@]}"
