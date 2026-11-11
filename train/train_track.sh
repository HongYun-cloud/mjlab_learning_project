#!/bin/bash

# ==============================
# Track Training (all terrains)
# ==============================
# 用法:
#   ./train/train_track.sh <run_name> -e <terrain> [-p <run_dir>] [-m <model>]
#
# terrain:
#   flat | stair | stair_up | stair_down | platform | plum | stones | tire | low_clearance_bar

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")" || exit 1

AVAILABLE_TERRAINS="flat stair stair_up stair_down platform plum stones tire low_clearance_bar"
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
    stair_up|stairup|up)
        SUFFIX="StairUp"; PROJECT="track_stair"; NUM_ENVS=512; MAX_ITERATIONS=5000 ;;
    stair|stairs|stair_combined|combined_stair|combined_stairs)
        SUFFIX="Stair"; PROJECT="track_stair"; NUM_ENVS=512; MAX_ITERATIONS=5000 ;;
    stair_down|stairdown|down)
        SUFFIX="StairDown"; PROJECT="track_stair"; NUM_ENVS=512; MAX_ITERATIONS=5000 ;;
    platform)
        SUFFIX="Platform"; PROJECT="track_stair"; NUM_ENVS=1024; MAX_ITERATIONS=6000 ;;
    plum|plum_blossom|plumblossom)
        SUFFIX="PlumBlossom"; PROJECT="track_plum"; NUM_ENVS=512; MAX_ITERATIONS=10000 ;;
    stones|stone2row|stones2row|stones2rows|stone_2rows|stones_2rows)
        SUFFIX="Stones2Rows"; PROJECT="track_stones_2rows"; NUM_ENVS=512; MAX_ITERATIONS=2000 ;;
    tire|tire_array|tirearray)
        SUFFIX="TireArray"; PROJECT="go2_tair"; NUM_ENVS=512; MAX_ITERATIONS=500 ;;
    low_clearance_bar|low_clearance|lowclearancebar)
        SUFFIX="LowClearanceBar"; PROJECT="track_low_clearance_bar"; NUM_ENVS=512; MAX_ITERATIONS=5000 ;;
    *)
        echo "错误：未知地形 '$TERRAIN'"
        echo "可使用的地形: $AVAILABLE_TERRAINS"
        exit 1
        ;;
esac

source "$SCRIPT_DIR/lib_train_args.sh"
train_parse_args "${ARGS[@]}"

TASK="Mjlab-Tracking-${SUFFIX}-Unitree-Go2"

echo "======================================"
echo "       Go2 Track Training"
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
