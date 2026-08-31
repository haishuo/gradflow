#!/bin/bash
#SBATCH --job-name=gf-a4-repl
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --output=slurm-replication-%j.out
#SBATCH --error=slurm-replication-%j.err

set -euo pipefail

: "${GRADFLOW_UNITY_RUN:?GRADFLOW_UNITY_RUN must name the staged run directory}"

workspace=/work/pi_zchen2_umassd_edu/hshu
case "$GRADFLOW_UNITY_RUN" in
    "$workspace"/*) ;;
    *) echo "run directory is outside the designated workspace" >&2; exit 2 ;;
esac

repo="$GRADFLOW_UNITY_RUN/repo"
python="$GRADFLOW_UNITY_RUN/venv-pytorch-2.13-cu126/bin/python"
output="$GRADFLOW_UNITY_RUN/evidence"

test -x "$python"
test -f "$GRADFLOW_UNITY_RUN/venv-pytorch-2.13-cu126/.gradflow-environment-complete"
cd "$GRADFLOW_UNITY_RUN"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"$python" "$GRADFLOW_UNITY_RUN/controller/run_second_machine.py" \
    --repo "$repo" \
    --output "$output"

