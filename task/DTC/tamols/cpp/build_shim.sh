#!/usr/bin/env bash
# 编译 QuadProg++ C++ shim 为共享库. 用法:
#   bash build_shim.sh [QUADPROPP_SRC_DIR]
# 默认 QUADPROPP_SRC_DIR=/home/hong/myrobot_mjlab/src/gowrobot_mjlab/QuadProgpp/src
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-/home/hong/myrobot_mjlab/src/gowrobot_mjlab/QuadProgpp/src}"
# MATRIX_DIM 必须 >= max(n, m): 当前 NLP 规模 n<=105, m<=~300; 留余量 1024.
g++ -O2 -fPIC -shared -DMATRIX_DIM=1024 -I"$SRC" \
    "$HERE/quadprog_shim.cc" "$SRC/QuadProg++.cc" "$SRC/Array.cc" \
    -o "$HERE/libquadprog_shim.so"
echo "built $HERE/libquadprog_shim.so"
