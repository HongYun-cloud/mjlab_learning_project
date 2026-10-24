
#!/bin/bash

# ==============================
# Go2 mjlab play
# ==============================

# 必须提供运行名称
MODEL=$(find -L wandb/latest-run/files \
  -maxdepth 1 -type f -name 'model_*.pt' \
  | sort -V | tail -n 1)

echo "加载模型：$MODEL"

TASK="Mjlab-Velocity-TireArray-Unitree-Go2"



echo "======================================"
echo "       Go2 Play"
echo "======================================"
echo "Task:      $TASK"
echo "Run name:  $RUN_NAME"
echo "======================================"

uv run play "$TASK" \
    --checkpoint-file "$MODEL"

