#!/usr/bin/env bash
# 构建 C++ NLP 核心共享库.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/setup_cppad.sh"
CPPAD_INC="/usr/include"
if [ ! -d /usr/include/cppad ]; then
    CPPAD_INC="$HERE/third_party/usr/include"
fi
EIGEN_INC="$(pkg-config --cflags-only-I eigen3 2>/dev/null | sed 's/^-I//' | awk '{print $1}')"
if [ -z "${EIGEN_INC:-}" ]; then
    EIGEN_INC="/usr/include/eigen3"
fi

QP_SRC="${QUADPROG_SRC:-/home/hong/myrobot_mjlab/src/gowrobot_mjlab/QuadProgpp/src}"
g++ -O2 -std=c++17 -fPIC -shared -DMATRIX_DIM=1024 \
    -I"$CPPAD_INC" -I"$EIGEN_INC" -I"$QP_SRC" -I"$HERE/.." \
    "$HERE/tamols_solver.cpp" \
    "$HERE/../quadprog_shim.cc" "$QP_SRC/QuadProg++.cc" "$QP_SRC/Array.cc" \
    -o "$HERE/libtamols_core.so"
echo "built $HERE/libtamols_core.so"
