#!/bin/bash
set -euo pipefail

source_repo=$(git rev-parse --show-toplevel)
tag=academic-v0.1.0-rc2
commit=c5e8ab81ef5b33a2138b2db33afc538398b6f57f
workspace=/mnt/projects
run_dir=$workspace/gradflow-a4-moody-20260831
controller=$source_repo/experiments/academic_a4

test "$(git -C "$source_repo" rev-parse "$tag^{commit}")" = "$commit"
test -z "$(git -C "$source_repo" status --porcelain=v1)"
test ! -e "$run_dir"
mkdir -m 700 "$run_dir" "$run_dir/controller"
git -C "$source_repo" rev-parse HEAD > "$run_dir/CONTROLLER_COMMIT.txt"
git -C "$source_repo" bundle create "$run_dir/gradflow-rc2.bundle" "$tag"
git -C "$source_repo" bundle verify "$run_dir/gradflow-rc2.bundle"
git clone "$run_dir/gradflow-rc2.bundle" "$run_dir/repo"
git -C "$run_dir/repo" checkout --detach "$tag"
git -C "$run_dir/repo" status --porcelain=v1 > "$run_dir/INITIAL_STATUS.txt"
test ! -s "$run_dir/INITIAL_STATUS.txt"

cp "$controller/unity/run_second_machine.py" "$run_dir/controller/"
cp "$controller/moody/verify_moody.py" "$run_dir/controller/"
cp "$controller/moody/setup_environment.sh" "$run_dir/controller/"
cp "$controller/moody/run_moody.sh" "$run_dir/controller/"
cp "$source_repo/docs/ACADEMIC_A4_MOODY_PROTOCOL.md" "$run_dir/controller/"
sha256sum "$run_dir/gradflow-rc2.bundle" "$run_dir/controller/"* \
    > "$run_dir/STAGED_SHA256SUMS"

printf 'staged_run=%s\n' "$run_dir"
