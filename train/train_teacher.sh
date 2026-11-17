#!/bin/bash

# ==============================
# Teacher Training (privileged observations)
# ==============================
# 用法:
#   ./train/train_teacher.sh <run_name> -e <terrain> [-p <run_dir>] [-m <model>]
#
# terrain:
#   flat | flat_stair_down | flat_stair_down_stones_2rows | flat_platform_stair_down | flat_platform_stair_down_plum | halfcompetitionmap | stair | stair_up | stair_down | platform | plum | stones | tire | low_clearance_bar

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")" || exit 1

AVAILABLE_TERRAINS="flat flat_stair_down flat_stair_down_stones_2rows flat_platform_stair_down flat_platform_stair_down_plum halfcompetitionmap stair stair_up stair_down platform plum stones tire low_clearance_bar"
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
    flat) SUFFIX="Flat" ;;
    flat_stair_down|flatstairdown) SUFFIX="FlatStairDown" ;;
    flat_stair_down_stones_2rows|flatstairdownstones2rows|flat_stair_down_stone2row|flatstairdownstone2row) SUFFIX="FlatStairDownStones2Rows" ;;
    flat_platform_stair_down|flatplatformstairdown) SUFFIX="FlatPlatformStairDown" ;;
    flat_platform_stair_down_plum|flatplatformstairdownplum|flat_stair_down_platform_plum|flatstairdownplatformplum) SUFFIX="FlatPlatformStairDownPlum" ;;
    halfcompetitionmap|half_competition_map|half_competition) SUFFIX="HalfCompetitionMap" ;;
    stair_up|stairup|up) SUFFIX="StairUp" ;;
    stair|stairs|stair_combined|combined_stair|combined_stairs) SUFFIX="Stair" ;;
    stair_down|stairdown|down) SUFFIX="StairDown" ;;
    platform) SUFFIX="Platform" ;;
    plum|plum_blossom|plumblossom) SUFFIX="PlumBlossom" ;;
    stones|stone2row|stones2row|stones2rows|stone_2rows|stones_2rows) SUFFIX="Stones2Rows" ;;
    tire|tire_array|tirearray) SUFFIX="TireArray" ;;
    low_clearance_bar|low_clearance|lowclearancebar) SUFFIX="LowClearanceBar" ;;
    *)
        echo "错误：未知地形 '$TERRAIN'"
        echo "可使用的地形: $AVAILABLE_TERRAINS"
        exit 1
        ;;
esac

source "$SCRIPT_DIR/lib_train_args.sh"
train_parse_args "${ARGS[@]}"

TASK="Mjlab-Tracking-${SUFFIX}-Teacher-Unitree-Go2"
PROJECT="track_teacher"
# 可通过环境变量覆盖，便于先跑小规模验证：
#   NUM_ENVS=256 MAX_ITERATIONS=300 ./train/train_teacher.sh quick -e flat_stair_down
NUM_ENVS="${NUM_ENVS:-512}"
if [[ -z "${MAX_ITERATIONS:-}" ]]; then
    # 混合地形比单地形难，默认训练更久。
    if [[ "$SUFFIX" == "FlatPlatformStairDown" || "$SUFFIX" == "FlatPlatformStairDownPlum" || "$SUFFIX" == "FlatStairDownStones2Rows" || "$SUFFIX" == "HalfCompetitionMap" ]]; then
        MAX_ITERATIONS=6000
    elif [[ "$SUFFIX" == "FlatStairDown" ]]; then
        MAX_ITERATIONS=4000
    else
        MAX_ITERATIONS=2000
    fi
fi

echo "======================================"
echo "       Go2 Teacher Training"
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
