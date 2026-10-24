
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

TASK="Mjlab-Tracking-Platform-Unitree-Go2"
PROJECT="track_stair"
RUN_NAME="$1"
BASEPATH="2026-10-18_17-59-40_fromstairfirst"

MODEL=$(find -L "logs/rsl_rl/go2_velocity/$BASEPATH" \
  -maxdepth 1 -type f -name 'model_*.pt' \
  | sort -V | tail -n 1| xargs -r basename)

echo "加载模型：$MODEL"

NUM_ENVS=1024
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
    --agent.run-name "$RUN_NAME" \
    --agent.load-run "$BASEPATH" \
    --agent.load-checkpoint "$MODEL" \
    --agent.resume False \
    --env.viewer.env-idx 0
    

