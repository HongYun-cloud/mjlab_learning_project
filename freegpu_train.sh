#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

mkdir -p logs/rsl_rl/go2_velocity/model1
mv ./model.pt ./logs/rsl_rl/go2_velocity/model1/