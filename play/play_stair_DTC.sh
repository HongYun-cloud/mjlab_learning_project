
#!/bin/bash

# ==============================
# Go2 mjlab play
# ==============================


MODEL=$(find -L wandb/latest-run/files \
  -maxdepth 1 -type f -name 'model_*.pt' \
  | sort -V | tail -n 1)

# MODEL="wandb/latest-run/files/model_600.pt"


echo "加载模型：$MODEL"



TASK="Mjlab-DTC-Velocity-StairFromFlat-Unitree-Go2"
RUN_NAME="$1"

echo "======================================"
echo "       Go2 Play"
echo "======================================"
echo "Task:      $TASK"
echo "Run name:  $RUN_NAME"
echo "======================================"



uv run play "$TASK" \
    --checkpoint-file "$MODEL"   \
    --viewer viser

