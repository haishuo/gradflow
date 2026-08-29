#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
build="$here/build"
cuda_root=/home/haishuo/cuda-13.0

if [[ $# -ne 1 || ( $1 != r1 && $1 != r2 && $1 != r3 && $1 != r4 ) ]]; then
  printf 'usage: %s r1|r2|r3|r4\n' "$0" >&2
  exit 2
fi

level=${1#r}
mkdir -p "$build"
"$cuda_root/bin/nvcc" \
  -O3 \
  --fmad=false \
  --prec-div=true \
  --prec-sqrt=true \
  -lineinfo \
  -Xptxas=-v \
  -std=c++17 \
  -arch=sm_120 \
  -DGRADFLOW_RECOVERY_LEVEL="$level" \
  "$here/u0.cu" \
  -o "$build/gradflow_$1"

printf '%s\n' "$build/gradflow_$1"
