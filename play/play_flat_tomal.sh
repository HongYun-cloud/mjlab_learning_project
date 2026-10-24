#!/bin/bash

# ==============================
# Go2 Flat Track Play
# ==============================

MODEL=$(find -L wandb/latest-run/files \
  -maxdepth 1 -type f -name 'model_*.pt' \
  | sort -V | tail -n 1)

echo "加载模型：$MODEL"

TASK="Mjlab-Tamols-Tracking-Flat-Unitree-Go2"
RUN_NAME="$1"

echo "======================================"
echo "       Go2 Flat Track Play"
echo "======================================"
echo "Task:      $TASK"
echo "Run name:  $RUN_NAME"
echo "======================================"

uv run play "$TASK" \
    --checkpoint-file "$MODEL" \
    --viewer viser
