#!/bin/bash
#SBATCH --job-name=gf-a4-env
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=slurm-env-%j.out
#SBATCH --error=slurm-env-%j.err

set -euo pipefail

: "${GRADFLOW_UNITY_RUN:?GRADFLOW_UNITY_RUN must name the staged run directory}"

workspace=/work/pi_zchen2_umassd_edu/hshu
case "$GRADFLOW_UNITY_RUN" in
    "$workspace"/*) ;;
    *) echo "run directory is outside the designated workspace" >&2; exit 2 ;;
esac

repo="$GRADFLOW_UNITY_RUN/repo"
venv="$GRADFLOW_UNITY_RUN/venv-pytorch-2.13-cu126"
marker="$venv/.gradflow-environment-complete"

cd "$GRADFLOW_UNITY_RUN"
if test -f "$marker"; then
    "$venv/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda)'
    exit 0
fi
if test -e "$venv"; then
    echo "refusing incomplete pre-existing environment: $venv" >&2
    exit 3
fi

python3 -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$venv/bin/python" -m pip install 'torch==2.13.0' \
    --index-url https://download.pytorch.org/whl/cu126
"$venv/bin/python" -m pip install -e "$repo[test]"
"$venv/bin/python" -m pip freeze > "$GRADFLOW_UNITY_RUN/environment-freeze.txt"
"$venv/bin/python" - <<'PY'
from pathlib import Path
import os
import torch

run = Path(os.environ["GRADFLOW_UNITY_RUN"])
marker = run / "venv-pytorch-2.13-cu126" / ".gradflow-environment-complete"
marker.write_text(f"torch={torch.__version__}\ncuda={torch.version.cuda}\n")
print(marker.read_text(), end="")
PY

