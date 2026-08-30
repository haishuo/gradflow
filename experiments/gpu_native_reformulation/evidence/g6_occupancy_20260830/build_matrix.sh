#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
source_file="$here/../native_u0/u0.cu"
build="$here/build"
logs="$here/compiler_logs"
cuda_root=/home/haishuo/cuda-13.0

mkdir -p "$build" "$logs"
: > "$logs/commands.txt"

for block in 64 128 256; do
  for policy in u r112 r96; do
    if [[ $policy == u ]]; then
      limit=0
      register_flag=()
    else
      limit=${policy#r}
      register_flag=("--maxrregcount=$limit")
    fi
    id="b${block}_${policy}"
    contract="g6_r6q_${id}_v1"
    executable="$build/gradflow_g6_$id"
    command=(
      "$cuda_root/bin/nvcc"
      -O3
      --fmad=false
      --prec-div=true
      --prec-sqrt=true
      -lineinfo
      -Xptxas=-v
      -std=c++17
      -arch=sm_120
      -DGRADFLOW_RECOVERY_LEVEL=7
      "-DGRADFLOW_FACE_THREADS=$block"
      "-DGRADFLOW_G6_REGISTER_LIMIT=$limit"
      "-DGRADFLOW_G6_CONTRACT=\"$contract\""
      "${register_flag[@]}"
      "$source_file"
      -o "$executable"
    )
    printf '%q' "${command[0]}" >> "$logs/commands.txt"
    for argument in "${command[@]:1}"; do
      printf ' %q' "$argument" >> "$logs/commands.txt"
    done
    printf '\n' >> "$logs/commands.txt"
    "${command[@]}" > "$logs/$id.log" 2>&1
    printf '%s\n' "$executable"
  done
done
