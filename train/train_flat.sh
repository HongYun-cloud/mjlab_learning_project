
#!/bin/bash

# ==============================
# Go2 mjlab training
# ==============================

# 必须提供运行名称
if [ -z "$1" ]; then
    echo "错误：必须指定本次运行的名称"
    echo ""
    echo "用法："
    echo "  ./run.sh <run_name>"
    echo ""
    echo "例如："
    echo "  ./run.sh go2_baseline"
    echo "  ./run.sh go2_action_rate_02"
    exit 1
fi

TASK="Mjlab-Velocity-Flat-Unitree-Go2"
PROJECT="go2"
RUN_NAME="$1"

NUM_ENVS=2048
MAX_ITERATIONS=1000

echo "======================================"
echo "       Go2 Training"
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
    --agent.run-name "$RUN_NAME"

