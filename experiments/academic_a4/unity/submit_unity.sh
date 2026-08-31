#!/bin/bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
run_dir=$(dirname "$script_dir")
workspace=/work/pi_zchen2_umassd_edu/hshu

case "$run_dir" in
    "$workspace"/*) ;;
    *) echo "controller is outside the designated workspace: $run_dir" >&2; exit 2 ;;
esac

export GRADFLOW_UNITY_RUN="$run_dir"
env_job=$(sbatch --parsable --export=ALL,GRADFLOW_UNITY_RUN="$run_dir" \
    "$script_dir/setup_environment.sbatch")
replication_job=$(sbatch --parsable --dependency="afterok:$env_job" \
    --export=ALL,GRADFLOW_UNITY_RUN="$run_dir" \
    "$script_dir/replicate.sbatch")

cat > "$run_dir/SUBMITTED_JOBS.txt" <<EOF
environment_job=$env_job
replication_job=$replication_job
submitted_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
run_dir=$run_dir
EOF

cat "$run_dir/SUBMITTED_JOBS.txt"

