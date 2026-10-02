
#!/bin/bash

# ==============================
# Go2 mjlab play
# ==============================

# 必须提供运行名称
if [ -z "$1" ]; then
    echo "错误：必须指定要播放的实验名称"
    echo ""
    echo "用法："
    echo "  ./play.sh <run_name>"
    echo ""
    echo "例如："
    echo "  ./play.sh go2_baseline"
    echo "  ./play.sh go2_action_rate_02"
    exit 1
fi

TASK="Mjlab-Velocity-Flat-Unitree-Go2"

RUN_NAME="$1"

echo "======================================"
echo "       Go2 Play"
echo "======================================"
echo "Task:      $TASK"
echo "Run name:  $RUN_NAME"
echo "======================================"

uv run play "$TASK" \
    --wandb-run-path "$RUN_NAME"

