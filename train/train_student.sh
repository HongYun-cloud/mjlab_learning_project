#!/bin/bash

# ==============================
# Student Training (multi-teacher distillation)
# ==============================
# 用法:
#   ./train/train_student.sh <run_name> -e <terrain> [-t <teacher_checkpoint>] [-p <student_run>] [-m <student_model>]
#
# terrain:
#   flat | stair | stair_up | stair_down | platform | plum | stones | tire | low_clearance_bar
#
# -t: 教师 PPO checkpoint（文件或目录；目录时取最新的 *.pt）
#     不指定时使用 task/student/config/teachers.py 里的注册路径。
# -p/-m: 恢复 student 蒸馏训练，不是教师 checkpoint。

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")" || exit 1

AVAILABLE_TERRAINS="flat stair stair_up stair_down platform plum stones tire low_clearance_bar"
TERRAIN="flat"
TEACHER_CHECKPOINT=""
TEACHER_ARG_GIVEN=0
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
        -t|--teacher)
            if [[ -z "${2:-}" ]]; then
                echo "错误：-t 后面需要跟教师 checkpoint 路径"
                exit 1
            fi
            TEACHER_CHECKPOINT="$2"
            TEACHER_ARG_GIVEN=1
            shift 2
            ;;
        -h|--help)
            echo "用法: ./train/$(basename "$0") <run_name> -e <terrain> [-t <teacher_checkpoint>] [-p <run_dir>] [-m <model>]"
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

if [[ -n "$TEACHER_CHECKPOINT" && -d "$TEACHER_CHECKPOINT" ]]; then
    TEACHER_CHECKPOINT=$(find -L "$TEACHER_CHECKPOINT" \
        -maxdepth 1 -type f -name "*.pt" \
        | sort -V | tail -1)
fi
if [[ "$TEACHER_ARG_GIVEN" -eq 1 && -z "$TEACHER_CHECKPOINT" ]]; then
    echo "错误：教师目录下没有 *.pt 文件"
    exit 1
fi
if [[ -n "$TEACHER_CHECKPOINT" && ! -f "$TEACHER_CHECKPOINT" ]]; then
    echo "错误：找不到教师 checkpoint：$TEACHER_CHECKPOINT"
    exit 1
fi

TEACHER_ARGS=()
if [[ -n "$TEACHER_CHECKPOINT" ]]; then
    TEACHER_ARGS=(--agent.teacher-checkpoint "$TEACHER_CHECKPOINT")
else
    echo "提示：未指定 -t，将使用 task/student/config/teachers.py 中的教师路径。"
fi

TASK="Mjlab-Tracking-${SUFFIX}-Student-Unitree-Go2"
PROJECT="track_student"
NUM_ENVS=512
MAX_ITERATIONS=5000

echo "======================================"
echo "       Go2 Student Distillation"
echo "======================================"
echo "Terrain:       $TERRAIN ($SUFFIX)"
echo "Task:          $TASK"
echo "Project:       $PROJECT"
echo "Run name:      $RUN_NAME"
echo "Num envs:      $NUM_ENVS"
echo "Iterations:    $MAX_ITERATIONS"
echo "Teacher:       ${TEACHER_CHECKPOINT:-<teachers.py>}"
echo "Student resume:$RESUME"
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
    "${TEACHER_ARGS[@]}" \
    "${LOAD_ARGS[@]}" \
    --agent.resume "$RESUME" \
    --env.viewer.env-idx 0
