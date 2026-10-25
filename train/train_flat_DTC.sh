#!/bin/bash

# ==============================
# Go2 Flat Tracking Training
# ==============================

if [ -z "$1" ]; then
    echo "错误：必须指定本次运行的名称"
    echo ""
    echo "用法："
    echo "  ./train_flat_track.sh <run_name>"
    echo ""
    echo "例如："
    echo "  ./train_flat_track.sh go2_flat_track_baseline"
    exit 1
fi



TASK="Mjlab-DTC-Velocity-Flat-Unitree-Go2"
PROJECT="track_flat"
RUN_NAME="$1"
BASEPATH="2026-10-21_20-04-20_tamols"

RUN_DIR="logs/rsl_rl/go2_velocity/$BASEPATH"
MODEL=""
LOAD_ARGS=()
if [ -d "$RUN_DIR" ]; then
  MODEL=$(find -L "$RUN_DIR" \
    -maxdepth 1 -type f -name 'model_*.pt' \
    | sort -V | tail -n 1| xargs -r basename)
fi

if [ -n "$MODEL" ]; then
  echo "加载模型：$BASEPATH/$MODEL"
  LOAD_ARGS=(--agent.load-run "$BASEPATH" --agent.load-checkpoint "$MODEL")
else
  echo "未找到可加载的模型 ($RUN_DIR)，将从零开始训练。"
fi


NUM_ENVS=2048
MAX_ITERATIONS=1000

echo "======================================"
echo "       Go2 Flat Track Training"
echo "======================================"
echo "Task:          $TASK"
echo "Project:       $PROJECT"
echo "Run name:      $RUN_NAME"
echo "Num envs:      $NUM_ENVS"
echo "Iterations:    $MAX_ITERATIONS"
echo "======================================"

uv run train "$TASK" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.max-iterations "$MAX_ITERATIONS" \
    --agent.wandb-project "$PROJECT" \
    --agent.run-name "$RUN_NAME" \
    "${LOAD_ARGS[@]}" \
    --agent.resume False   \
    --env.viewer.env-idx 0
