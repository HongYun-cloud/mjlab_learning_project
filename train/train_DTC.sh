#!/bin/bash

# ==============================
# DTC Velocity Training (all terrains)
# ==============================
# 用法:
#   ./train/train_DTC.sh <run_name> -e <terrain> [-p <run_dir>] [-m <model>]
#
# terrain:
#   flat | stair | tire | platform

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")" || exit 1

AVAILABLE_TERRAINS="flat stair tire platform"
TERRAIN="flat"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--env)
            if [[ -z "${2:-}" ]]; then
                echo "错误：-e 后面需要跟地形名称"
                echo "可使用的地形: $AVAILABLE_TERRAINS"
                exit 1
            fi
            TERRAIN="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: ./train/$(basename "$0") <run_name> -e <terrain> [-p <run_dir>] [-m <model>]"
            echo "可使用的地形: $AVAILABLE_TERRAINS"
            exit 0
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

TERRAIN="${TERRAIN,,}"
TERRAIN="${TERRAIN//-/_}"

case "$TERRAIN" in
    flat)
        SUFFIX="Flat"; PROJECT="track_flat"; NUM_ENVS=1024; MAX_ITERATIONS=1000 ;;
    stair|stair_up|stair_down|stairs)
        SUFFIX="StairFromFlat"; PROJECT="track_stair"; NUM_ENVS=1024; MAX_ITERATIONS=1000 ;;
    tire|tire_array|tirearray)
        SUFFIX="TireArray"; PROJECT="track_tire"; NUM_ENVS=1024; MAX_ITERATIONS=1000 ;;
    platform)
        SUFFIX="Platform"; PROJECT="track_platform"; NUM_ENVS=1024; MAX_ITERATIONS=1000 ;;
    *)
        echo "错误：未知地形 '$TERRAIN'"
        echo "可使用的地形: $AVAILABLE_TERRAINS"
        exit 1
        ;;
esac

source "$SCRIPT_DIR/lib_train_args.sh"
train_parse_args "${ARGS[@]}"

TASK="Mjlab-DTC-Velocity-${SUFFIX}-Unitree-Go2"

echo "======================================"
echo "       Go2 DTC Velocity Training"
echo "======================================"
echo "Terrain:       $TERRAIN ($SUFFIX)"
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
