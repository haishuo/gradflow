#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "$0")" && pwd)
build="$here/build"
cuda=/home/haishuo/cuda-13.0
mkdir -p "$build"

g++ -O3 -march=native -std=c++17 -fopenmp -ffp-contract=fast \
  -I "$here" -I "$cuda/include" -c "$here/cpu.cpp" -o "$build/cpu.o"
g++ -O3 -march=native -std=c++17 -fopenmp -ffp-contract=fast \
  -I "$here" -I "$cuda/include" -c "$here/main.cpp" -o "$build/main.o"
"$cuda/bin/nvcc" -O3 -std=c++17 -arch=sm_120 --fmad=true \
  -I "$here" -c "$here/cuda.cu" -o "$build/cuda.o"
g++ -fopenmp "$build/main.o" "$build/cpu.o" "$build/cuda.o" \
  -L "$cuda/lib64" -lcudart -Wl,-rpath,"$cuda/lib64" -o "$build/shu_ceiling"

printf '%s\n' "$build/shu_ceiling"

