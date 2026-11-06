#!/usr/bin/env bash
# 打开 TensorBoard 查看 logs/rsl_rl/go2_velocity 下的训练曲线。
#
# 用法:
#   ./tensorboard.sh                         # 查看全部 run
#   ./tensorboard.sh --latest                # 只看最近修改的 run
#   ./tensorboard.sh <run_name_or_path>      # 只看指定 run
#   ./tensorboard.sh --port 6007             # 指定端口
#   ./tensorboard.sh --list                  # 列出所有 run
#   ./tensorboard.sh --no-browser            # 不自动打开浏览器

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="$SCRIPT_DIR/logs/rsl_rl/go2_velocity"
LOGDIR="$LOG_ROOT"
PORT=6006
HOST=127.0.0.1
OPEN_BROWSER=1
TARGET=""
LIST_ONLY=0

usage() {
  cat <<EOF
用法:
  ./tensorboard.sh [run_name_or_path] [选项]

选项:
  -p, --port <port>    TensorBoard 端口，默认 6006
      --host <host>    监听地址，默认 127.0.0.1
  -l, --logdir <dir>   日志根目录，默认 logs/rsl_rl/go2_velocity
      --latest         只打开最近修改的 run
      --list           列出可用 run 后退出
      --no-browser     不自动打开浏览器
  -h, --help           显示帮助

示例:
  ./tensorboard.sh
  ./tensorboard.sh --latest
  ./tensorboard.sh 2026-11-05_17-32-17_use:LALRT,stair,addjointpower,aggressmotion
  ./tensorboard.sh --latest --port 6007
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)
      [[ $# -ge 2 ]] || { echo "错误：--port 需要参数"; exit 1; }
      PORT="$2"
      shift 2
      ;;
    --host)
      [[ $# -ge 2 ]] || { echo "错误：--host 需要参数"; exit 1; }
      HOST="$2"
      shift 2
      ;;
    -l|--logdir)
      [[ $# -ge 2 ]] || { echo "错误：--logdir 需要参数"; exit 1; }
      LOG_ROOT="$2"
      LOGDIR="$LOG_ROOT"
      shift 2
      ;;
    --latest)
      TARGET="__LATEST__"
      shift
      ;;
    --list)
      LIST_ONLY=1
      shift
      ;;
    --no-browser)
      OPEN_BROWSER=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "错误：未知参数 $1"
      usage
      exit 1
      ;;
    *)
      TARGET="$1"
      shift
      ;;
  esac
done

if [[ ! -d "$LOG_ROOT" ]]; then
  echo "错误：日志目录不存在：$LOG_ROOT"
  exit 1
fi

list_runs() {
  find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %f\n' | sort -nr | cut -d' ' -f2-
}

if [[ "$LIST_ONLY" == "1" ]]; then
  list_runs
  exit 0
fi

if [[ "$TARGET" == "__LATEST__" ]]; then
  LOGDIR="$(find "$LOG_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -z "$LOGDIR" ]]; then
    echo "错误：$LOG_ROOT 下没有 run"
    exit 1
  fi
elif [[ -n "$TARGET" ]]; then
  if [[ -d "$TARGET" ]]; then
    LOGDIR="$TARGET"
  elif [[ -d "$LOG_ROOT/$TARGET" ]]; then
    LOGDIR="$LOG_ROOT/$TARGET"
  else
    echo "错误：找不到 run：$TARGET"
    echo
    echo "可用 run："
    list_runs | sed 's/^/  /'
    exit 1
  fi
fi

if ! find "$LOGDIR" -maxdepth 1 -name 'events.out.tfevents*' | grep -q .; then
  echo "警告：$LOGDIR 下没有 events.out.tfevents*，TensorBoard 可能没有数据。"
fi

BROWSER_HOST="$HOST"
if [[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]]; then
  BROWSER_HOST="127.0.0.1"
fi
URL="http://$BROWSER_HOST:$PORT"

echo "======================================"
echo "            TensorBoard"
echo "======================================"
echo "Log dir: $LOGDIR"
echo "URL:     $URL"
echo "======================================"

if [[ "$OPEN_BROWSER" == "1" ]]; then
  (
    sleep 2
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$URL" >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
      open "$URL" >/dev/null 2>&1 || true
    fi
  ) &
fi

if command -v uvx >/dev/null 2>&1; then
  TB_CMD=(uvx --from tensorboard tensorboard)
elif command -v uv >/dev/null 2>&1; then
  TB_CMD=(uv run --with tensorboard tensorboard)
else
  echo "错误：找不到 uvx 或 uv，无法启动 TensorBoard"
  exit 1
fi

exec "${TB_CMD[@]}" \
  --logdir "$LOGDIR" \
  --host "$HOST" \
  --port "$PORT" \
  --reload_multifile True
