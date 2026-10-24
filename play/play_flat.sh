
#!/bin/bash

# ==============================
# Go2 mjlab play
# ==============================

# 必须提供运行名称
MODEL="src/myrobot_mjlab/model/flat_999.pt"

echo "加载模型：$MODEL"

TASK="Mjlab-Velocity-Flat-Unitree-Go2"



echo "======================================"
echo "       Go2 Play"
echo "======================================"
echo "Task:      $TASK"
echo "Run name:  $RUN_NAME"
echo "======================================"

uv run play "$TASK" \
    --checkpoint-file "$MODEL"

