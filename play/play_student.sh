#!/bin/bash

# ==============================
# Student Play (distilled policy)
# ==============================
# 用法:
#   ./play/play_student.sh -e <terrain> -c <student_checkpoint>
#   ./play/play_student.sh -e <terrain> [-p <run_dir>] [-m <model>]
#
# -c 可以直接指定 student checkpoint 文件或目录（目录取最新 *.pt）。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")" || exit 1

AVAILABLE_TERRAINS="flat stair_up stair_down platform plum stones tire low_clearance_bar"
TERRAIN="flat"
CHECKPOINT=""
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
        -c|--checkpoint)
            if [[ -z "${2:-}" ]]; then
                echo "错误：-c 后面需要跟 checkpoint 路径"
                exit 1
            fi
            CHECKPOINT="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: ./play/$(basename "$0") -e <terrain> -c <checkpoint>"
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
    stair_up|stairup|up) SUFFIX="StairUp" ;;
    stair_down|stairdown|down|stair) SUFFIX="StairDown" ;;
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

if [[ -n "$CHECKPOINT" ]]; then
    MODEL="$CHECKPOINT"
    RUN_NAME="${ARGS[0]:-student}"
else
    source "$SCRIPT_DIR/lib_play_args.sh"
    play_parse_args "${ARGS[@]}"
fi

if [[ -d "$MODEL" ]]; then
    MODEL=$(find -L "$MODEL" -maxdepth 1 -type f -name "*.pt" \
        | sort -V | tail -1)
fi
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
    echo "错误：找不到学生 checkpoint：${MODEL:-<empty>}"
    exit 1
fi

TASK="Mjlab-Tracking-${SUFFIX}-Student-Unitree-Go2"

echo "======================================"
echo "       Go2 Student Play"
echo "======================================"
echo "Terrain:   $TERRAIN ($SUFFIX)"
echo "Task:      $TASK"
echo "Model:     $MODEL"
echo "Run name:  $RUN_NAME"
echo "======================================"

uv run play "$TASK" \
    --checkpoint-file "$MODEL" \
    --viewer viser
