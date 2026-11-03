#!/bin/bash
# 被 play/*.sh source 的通用参数解析。
#
# 解析后设置以下全局变量:
#   MODEL      要加载的 checkpoint 路径
#   RUN_NAME   仅用于打印的标签
#
# 默认: 从 wandb/latest-run/files 取最新的 model_*.pt
# -p <run_dir>: 从 logs/rsl_rl/go2_velocity/<run_dir> 取
# -m <value>:
#   - 数字 (如 1000)          -> 取 model_1000.pt
#   - run 目录名 (如 2026-..) -> 从 logs/rsl_rl/go2_velocity/<run_dir> 取最新
#   - model_1000.pt           -> 取该文件
#   - 不指定                   -> 取当前来源下最新的 model_*.pt

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "该文件用于被 play/*.sh source，请不要直接执行。"
    exit 1
fi

PLAY_SCRIPT_NAME="$(basename "$0")"

play_usage() {
    cat <<EOF
用法:
  ./play/$PLAY_SCRIPT_NAME [run_name] [-p <run_dir>] [-m <value>]

参数:
  [run_name]    仅用于打印的标签 (可选)
  -p <run_dir>  从 logs/rsl_rl/go2_velocity/<run_dir> 找模型
  -m <value>    数字: 加载 model_<value>.pt
                run 目录名: 从该目录加载最新模型
                model_xxx.pt: 直接加载该文件
                不写 -m: 加载当前来源下最新的 model_*.pt

默认来源:
  wandb/latest-run/files

示例:
  ./play/$PLAY_SCRIPT_NAME
  ./play/$PLAY_SCRIPT_NAME -m 1000
  ./play/$PLAY_SCRIPT_NAME -m 2026-10-27_20-25-32_baseline
  ./play/$PLAY_SCRIPT_NAME -p 2026-10-27_20-25-32_baseline -m 100
EOF
}

play_parse_args() {
    MODEL=""
    RUN_NAME=""
    PATH_ARG=""
    MODEL_ARG=""
    SOURCE_DIR="wandb/latest-run/files"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--path)
                if [[ -z "${2:-}" ]]; then
                    echo "错误：-p 后面需要跟 run 目录名"
                    exit 1
                fi
                PATH_ARG="$2"
                shift 2
                ;;
            -m|--model)
                if [[ -z "${2:-}" ]]; then
                    echo "错误：-m 后面需要跟 run 目录名或迭代数，例如 -m 1000"
                    exit 1
                fi
                MODEL_ARG="$2"
                shift 2
                ;;
            -h|--help)
                play_usage
                exit 0
                ;;
            -*)
                echo "错误：未知参数 $1"
                play_usage
                exit 1
                ;;
            *)
                if [[ -z "$RUN_NAME" ]]; then
                    RUN_NAME="$1"
                else
                    echo "错误：多余的参数 $1"
                    play_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    # -p: 显式指定本地 run 目录
    if [[ -n "$PATH_ARG" ]]; then
        if [[ -d "$PATH_ARG" ]]; then
            SOURCE_DIR="$PATH_ARG"
        else
            SOURCE_DIR="logs/rsl_rl/go2_velocity/$PATH_ARG"
        fi
        if [[ ! -d "$SOURCE_DIR" ]]; then
            echo "错误：找不到 run 目录：$SOURCE_DIR"
            exit 1
        fi
    fi

    # -m: 可能是 run 目录名、数字或模型文件名
    if [[ -n "$MODEL_ARG" ]]; then
        if [[ -d "$MODEL_ARG" ]] || [[ -d "logs/rsl_rl/go2_velocity/$MODEL_ARG" ]]; then
            if [[ -n "$PATH_ARG" ]]; then
                echo "错误：-p 和 -m 不能同时指定 run 目录"
                exit 1
            fi
            if [[ -d "$MODEL_ARG" ]]; then
                SOURCE_DIR="$MODEL_ARG"
            else
                SOURCE_DIR="logs/rsl_rl/go2_velocity/$MODEL_ARG"
            fi
            MODEL_ARG=""
        fi
    fi

    # 默认来源 wandb/latest-run/files 可能刚创建、还没有保存模型；
    # 此时退回到最近一个真正包含 model_*.pt 的 wandb run。
    if [[ "$SOURCE_DIR" == "wandb/latest-run/files" ]]; then
        if ! compgen -G "$SOURCE_DIR/model_*.pt" >/dev/null 2>&1; then
            for _dir in $(ls -dt wandb/run-*/files 2>/dev/null); do
                if compgen -G "$_dir/model_*.pt" >/dev/null 2>&1; then
                    echo "[WARN] $SOURCE_DIR 下还没有 model_*.pt，改用最近的 $_dir" >&2
                    SOURCE_DIR="$_dir"
                    break
                fi
            done
        fi
    fi

    if [[ -n "$MODEL_ARG" ]]; then
        case "$MODEL_ARG" in
            model_*.pt) MODEL="$MODEL_ARG" ;;
            model_*)    MODEL="${MODEL_ARG}.pt" ;;
            *.pt)       MODEL="model_${MODEL_ARG}" ;;
            *)          MODEL="model_${MODEL_ARG}.pt" ;;
        esac

        if [[ ! -f "$SOURCE_DIR/$MODEL" ]]; then
            echo "错误：找不到模型：$SOURCE_DIR/$MODEL"
            echo "该来源下最近的模型："
            find -L "$SOURCE_DIR" -maxdepth 1 -type f -name 'model_*.pt' \
                -printf '%f\n' 2>/dev/null | sort -V | tail -n 10
            echo "提示：-m 可以是 run 目录名、数字 (如 1000) 或 model_xxx.pt。"
            exit 1
        fi
        MODEL="$SOURCE_DIR/$MODEL"
    else
        if [[ ! -d "$SOURCE_DIR" ]]; then
            echo "错误：找不到模型来源目录：$SOURCE_DIR"
            exit 1
        fi
        MODEL=$(find -L "$SOURCE_DIR" \
            -maxdepth 1 -type f -name 'model_*.pt' \
            | sort -V | tail -n 1)
        if [[ -z "$MODEL" ]]; then
            echo "错误：$SOURCE_DIR 下没有 model_*.pt"
            exit 1
        fi
    fi

    if [[ -z "$RUN_NAME" ]]; then
        RUN_NAME="play"
    fi
}
