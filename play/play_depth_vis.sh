#!/bin/bash

# ==============================
# Track Depth Visualization (Viser)
# ==============================
# 在 Viser 中实时查看深度相机：
#   - 原始深度 270x480 (HxW)
#   - 处理后的 18x32 (HxW) 深度图
#
# 用法:
#   ./play/play_depth_vis.sh -e <terrain> [-c <checkpoint>] [--num-envs N]
#
# 不传 -c 时使用零动作策略，只观察深度相机画面。
# 可用地形:
#   flat | stair | stair_up | stair_down | platform | plum | stones | tire | low_clearance_bar

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname -- "$SCRIPT_DIR")" || exit 1

exec uv run python "$SCRIPT_DIR/play_depth_vis.py" "$@"
