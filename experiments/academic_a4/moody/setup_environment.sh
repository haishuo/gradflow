#!/bin/bash
set -euo pipefail

run_dir=/mnt/projects/gradflow-a4-moody-20260831
repo=$run_dir/repo
venv=$run_dir/venv-pytorch-2.13-cu126
marker=$venv/.gradflow-environment-complete
log=$run_dir/environment-setup.log

test -d "$repo"
test ! -e "$venv"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
start_seconds=$SECONDS
{
    printf 'started_utc=%s\n' "$started"
    python3 --version
    python3 -m venv "$venv"
    "$venv/bin/python" -m pip install --upgrade pip setuptools wheel
    "$venv/bin/python" -m pip install 'torch==2.13.0' \
        --index-url https://download.pytorch.org/whl/cu126
    "$venv/bin/python" -m pip install -e "$repo[test]"
    "$venv/bin/python" -m pip freeze > "$run_dir/environment-freeze.txt"
    "$venv/bin/python" -c \
        'import torch; print(f"torch={torch.__version__}\ncuda={torch.version.cuda}")' \
        > "$marker"
    cat "$marker"
    printf 'duration_seconds=%s\n' "$((SECONDS - start_seconds))"
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee "$log"
