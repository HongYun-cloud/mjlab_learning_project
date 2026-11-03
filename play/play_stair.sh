#!/bin/bash

# ==============================
# Go2 Stair Play
# ==============================

# 切到仓库根目录，保证从任意目录执行时相对路径都正确
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")" || exit 1

source "$SCRIPT_DIR/lib_play_args.sh"
play_parse_args "$@"

TASK="Mjlab-Velocity-StairFromFlat-Unitree-Go2"

echo "加载模型：$MODEL"

echo "======================================"
echo "       Go2 Stair Play"
echo "======================================"
echo "Task:      $TASK"
echo "Run name:  $RUN_NAME"
echo "======================================"

uv run play "$TASK" \
    --checkpoint-file "$MODEL" \
    --viewer viser
