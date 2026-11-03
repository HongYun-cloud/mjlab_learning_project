#!/bin/bash

# ==============================
# Go2 Stair Training
# ==============================

# 切到仓库根目录，保证从任意目录执行时相对路径都正确
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")" || exit 1

source "$SCRIPT_DIR/lib_train_args.sh"
train_parse_args "$@"

TASK="Mjlab-Velocity-StairFromFlat-Unitree-Go2"
PROJECT="go2_stair"
NUM_ENVS=1024
MAX_ITERATIONS=300

echo "======================================"
echo "       Go2 Stair Training"
echo "======================================"
echo "Task:          $TASK"
echo "Project:       $PROJECT"
echo "Run name:      $RUN_NAME"
echo "Num envs:      $NUM_ENVS"
echo "Iterations:    $MAX_ITERATIONS"
echo "Resume:        $RESUME"
if [[ -n "$MODEL" ]]; then
    echo "Load run:      $BASEPATH"
    echo "Load model:    $MODEL"
fi
echo "======================================"

uv run train "$TASK" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.max-iterations "$MAX_ITERATIONS" \
    --agent.wandb-project "$PROJECT" \
    --agent.run-name "$RUN_NAME" \
    "${LOAD_ARGS[@]}" \
    --agent.resume "$RESUME" \
    --env.viewer.env-idx 0
