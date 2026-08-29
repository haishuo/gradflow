#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
build="$here/build"
cuda_root=/home/haishuo/cuda-13.0

mkdir -p "$build"
"$cuda_root/bin/nvcc" \
  -O3 \
  --use_fast_math \
  --fmad=true \
  -lineinfo \
  -Xptxas=-v \
  -std=c++17 \
  -arch=sm_120 \
  "$here/u0.cu" \
  -o "$build/gradflow_u0"

printf '%s\n' "$build/gradflow_u0"
