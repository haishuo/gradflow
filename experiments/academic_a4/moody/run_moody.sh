#!/bin/bash
set -euo pipefail

run_dir=/mnt/projects/gradflow-a4-moody-20260831
repo=$run_dir/repo
python=$run_dir/venv-pytorch-2.13-cu126/bin/python
output=$run_dir/evidence

test -x "$python"
test -f "$run_dir/venv-pytorch-2.13-cu126/.gradflow-environment-complete"
test ! -e "$output"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"$python" "$run_dir/controller/run_second_machine.py" \
    --repo "$repo" \
    --output "$output" \
    --execution-context standalone \
    --workspace-contract /mnt/projects \
    --protocol controller/ACADEMIC_A4_MOODY_PROTOCOL.md \
    2>&1 | tee "$run_dir/replication.log"
