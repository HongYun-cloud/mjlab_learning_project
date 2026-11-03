#!/usr/bin/env bash
# 准备 CppAD 头文件: 优先系统安装, 否则从 Ubuntu 包下载并解压到 third_party.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/third_party"

if [ -d /usr/include/cppad ]; then
    echo "使用系统 CppAD: /usr/include/cppad"
    exit 0
fi
if [ -d "$DEST/usr/include/cppad" ]; then
    echo "使用本地 CppAD: $DEST/usr/include/cppad"
    exit 0
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "下载 libcppad-dev ..."
(
    cd "$WORK"
    apt-get download libcppad-dev
)
mkdir -p "$DEST"
dpkg-deb -x "$WORK"/*.deb "$DEST"
echo "CppAD 已解压到 $DEST/usr/include/cppad"
