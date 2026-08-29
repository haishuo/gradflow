#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
build="$here/build"
cuda_root=/home/haishuo/cuda-13.0

if [[ $# -ne 1 || ( $1 != r1 && $1 != r2 && $1 != r3 && $1 != r4 && $1 != r5 && $1 != r6 && $1 != r6q ) ]]; then
  printf 'usage: %s r1|r2|r3|r4|r5|r6|r6q\n' "$0" >&2
  exit 2
fi

if [[ $1 == r6q ]]; then
  level=7
else
  level=${1#r}
fi
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
