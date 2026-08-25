#!/usr/bin/env bash
set -euo pipefail

experiment_dir=$(cd "$(dirname "$0")" && pwd)
input_file="$experiment_dir/inputs/10x10_one_step.txt"
scratch_dir=$(mktemp -d)
trap 'rm -rf -- "$scratch_dir"' EXIT

mkdir \
  "$scratch_dir/original" \
  "$scratch_dir/storage_only" \
  "$scratch_dir/original_repaired" \
  "$scratch_dir/dynamic"

(
  cd "$scratch_dir/original"
  "$experiment_dir/build/weno_original" < "$input_file" >/dev/null
)
(
  cd "$scratch_dir/storage_only"
  "$experiment_dir/build/weno_storage_only" < "$input_file" >/dev/null
)
(
  cd "$scratch_dir/original_repaired"
  "$experiment_dir/build/weno_original_repaired" < "$input_file" >/dev/null
)
(
  cd "$scratch_dir/dynamic"
  "$experiment_dir/build/weno_dynamic" < "$input_file" >/dev/null 2>stderr.txt
)

cmp "$scratch_dir/original/fort.8" "$scratch_dir/storage_only/fort.8"
cmp "$scratch_dir/original/fort.9" "$scratch_dir/storage_only/fort.9"
cmp "$scratch_dir/original_repaired/fort.8" "$scratch_dir/dynamic/fort.8"
cmp "$scratch_dir/original_repaired/fort.9" "$scratch_dir/dynamic/fort.9"

if grep -Eiq 'nan|inf' "$scratch_dir/dynamic/fort.8" "$scratch_dir/dynamic/fort.9"; then
  echo "FAIL: repaired dynamic output contains NaN or infinity" >&2
  exit 1
fi

if grep -Eq 'IEEE_(DIVIDE_BY_ZERO|INVALID|OVERFLOW)' "$scratch_dir/dynamic/stderr.txt"; then
  echo "FAIL: repaired dynamic run signalled a serious floating-point exception" >&2
  exit 1
fi

echo "PASS: storage-only dynamic output is byte-identical to the frozen original"
sha256sum "$scratch_dir/original/fort.8" "$scratch_dir/original/fort.9"
echo "PASS: repaired dynamic output is byte-identical to the repaired static program"
sha256sum "$scratch_dir/original_repaired/fort.8" "$scratch_dir/original_repaired/fort.9"
