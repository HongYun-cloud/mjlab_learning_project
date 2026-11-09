#!/bin/bash
# 被 train/*.sh source 的通用参数解析。
#
# 解析后设置以下全局变量:
#   RUN_NAME   本次运行的名称
#   BASEPATH   继续训练时指定的 run 目录名 (空表示重新开始)
#   MODEL      继续训练时加载的 checkpoint 文件名
#   LOAD_ARGS  传给 uv run train 的 --agent.load-run / --agent.load-checkpoint
#   RESUME     "True" 表示继续训练, "False" 表示重新开始
#   CURRICULUM_MODE  "curriculum" 或 "random"，并通过
#                    MYROBOT_CURRICULUM_MODE 传给 Track env_cfg

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "该文件用于被 train/*.sh source，请不要直接执行。"
    exit 1
fi

TRAIN_SCRIPT_NAME="$(basename "$0")"

train_usage() {
    cat <<EOF
用法:
  ./train/$TRAIN_SCRIPT_NAME <run_name> [-p <run_dir>] [-m <iteration>] [-c <mode>]

参数:
  <run_name>    本次训练的运行名称 (必填)
  -p <run_dir>  从 logs/rsl_rl/go2_velocity/<run_dir> 继续训练
                不写 -p 时重新开始训练
  -m <iter>     从该 run 下的 model_<iter>.pt 继续训练
                不写 -m 时默认使用该 run 下最新的 model_*.pt
                仅在与 -p 一起使用时有效
  -c <mode>     课程模式 (仅 Track 任务生效):
                curriculum  按到达次数升级/降级，到最高难度后随机 (默认)
                random      每次 reset 直接从 0~max_level 随机难度

示例:
  ./train/$TRAIN_SCRIPT_NAME first
  ./train/$TRAIN_SCRIPT_NAME first -p 2026-10-27_20-25-32_baseline
  ./train/$TRAIN_SCRIPT_NAME first -p 2026-10-27_20-25-32_baseline -m 1000
  ./train/$TRAIN_SCRIPT_NAME first -c random
EOF
}

train_parse_args() {
    RUN_NAME=""
    BASEPATH=""
    MODEL_ARG=""
    MODEL=""
    LOAD_ARGS=()
    RESUME="False"
    CURRICULUM_MODE="curriculum"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--path)
                if [[ -z "${2:-}" ]]; then
                    echo "错误：-p 后面需要跟 run 目录名"
                    exit 1
                fi
                BASEPATH="$2"
                shift 2
                ;;
            -c|--curriculum)
                if [[ -z "${2:-}" ]]; then
                    echo "错误：-c 后面需要跟 curriculum 或 random"
                    exit 1
                fi
                CURRICULUM_MODE="$2"
                shift 2
                ;;
            -m|--model)
                if [[ -z "${2:-}" ]]; then
                    echo "错误：-m 后面需要跟迭代数，例如 -m 1000"
                    exit 1
                fi
                MODEL_ARG="$2"
                shift 2
                ;;
            -h|--help)
                train_usage
                exit 0
                ;;
            -*)
                echo "错误：未知参数 $1"
                train_usage
                exit 1
                ;;
            *)
                if [[ -z "$RUN_NAME" ]]; then
                    RUN_NAME="$1"
                else
                    echo "错误：多余的参数 $1"
                    train_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    case "$CURRICULUM_MODE" in
        curriculum|normal|0)
            CURRICULUM_MODE="curriculum"
            ;;
        random|direct|1)
            CURRICULUM_MODE="random"
            ;;
        *)
            echo "错误：-c 只支持 curriculum 或 random"
            exit 1
            ;;
    esac
    export MYROBOT_CURRICULUM_MODE="$CURRICULUM_MODE"

    if [[ -z "$RUN_NAME" ]]; then
        echo "错误：必须指定本次运行的名称"
        train_usage
        exit 1
    fi

    case "$TRAIN_SCRIPT_NAME" in
        *_track.sh)
            echo "课程模式：$CURRICULUM_MODE"
            ;;
        *)
            if [[ "$CURRICULUM_MODE" == "random" ]]; then
                echo "警告：-c 只对 *_track.sh 任务生效，当前脚本会忽略。"
            fi
            ;;
    esac

    # 去掉 -p 末尾可能带上的斜杠
    BASEPATH="${BASEPATH%/}"

    if [[ -n "$BASEPATH" ]]; then
        RUN_DIR="logs/rsl_rl/go2_velocity/$BASEPATH"

        if [[ ! -d "$RUN_DIR" ]]; then
            echo "错误：找不到 run 目录：$RUN_DIR"
            exit 1
        fi

        if [[ -n "$MODEL_ARG" ]]; then
            # 支持 -m 1000、-m 1000.pt 或 -m model_1000.pt
            case "$MODEL_ARG" in
                model_*.pt) MODEL="$MODEL_ARG" ;;
                model_*)    MODEL="${MODEL_ARG}.pt" ;;
                *.pt)       MODEL="model_${MODEL_ARG}" ;;
                *)          MODEL="model_${MODEL_ARG}.pt" ;;
            esac

            if [[ ! -f "$RUN_DIR/$MODEL" ]]; then
                echo "错误：找不到模型：$RUN_DIR/$MODEL"
                echo "该 run 下最近的模型："
                find -L "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' \
                    -printf '%f\n' | sort -V | tail -n 10
                exit 1
            fi
        else
            MODEL=$(find -L "$RUN_DIR" \
                -maxdepth 1 -type f -name 'model_*.pt' \
                | sort -V | tail -n 1 | xargs -r basename)

            if [[ -z "$MODEL" ]]; then
                echo "错误：$RUN_DIR 下没有 model_*.pt"
                exit 1
            fi
        fi

        echo "继续训练：$BASEPATH/$MODEL"
        LOAD_ARGS=(--agent.load-run "$BASEPATH" --agent.load-checkpoint "$MODEL")
        RESUME="True"
    else
        if [[ -n "$MODEL_ARG" ]]; then
            echo "警告：指定了 -m 但没有指定 -p，忽略 -m，重新开始训练。"
        fi
        echo "重新开始训练：$RUN_NAME"
    fi
}
