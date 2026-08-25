#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 3 )); then
  echo "usage: $0 GRID_SIZE [TIME_STEPS=1] [TOUCH_ALL=0]" >&2
  exit 2
fi

grid_size=$1
time_steps=${2:-1}
touch_all=${3:-0}

if [[ ! $grid_size =~ ^[1-9][0-9]*$ ]]; then
  echo "GRID_SIZE must be a positive integer" >&2
  exit 2
fi
if [[ ! $time_steps =~ ^[0-9]+$ ]]; then
  echo "TIME_STEPS must be a nonnegative integer" >&2
  exit 2
fi
if [[ $touch_all != 0 && $touch_all != 1 ]]; then
  echo "TOUCH_ALL must be 0 or 1" >&2
  exit 2
fi

experiment_dir=$(cd "$(dirname "$0")" && pwd)
binary="$experiment_dir/build/weno_dynamic"

if [[ ! -x $binary ]]; then
  echo "build the experiment first with: make -C $experiment_dir" >&2
  exit 2
fi

printf '3\n%s %s\n0.1\n%s\n0.001\n1\n0\n' \
  "$grid_size" "$grid_size" "$time_steps" |
  env WENO_WRITE_SOLUTION=0 WENO_TOUCH_ALL="$touch_all" \
    /usr/bin/time -f \
      'wall=%e s user=%U s system=%S s peak_rss=%M KiB swaps=%W' \
      "$binary"
