#!/bin/bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
tag=academic-v0.1.0-rc2
workspace=/work/pi_zchen2_umassd_edu/hshu
target=${UNITY_SSH_TARGET:-hshu_umassd_edu@login.unityhpc.org}
run_name=${UNITY_RUN_NAME:-gradflow-a4-$(date -u +%Y%m%dT%H%M%SZ)}
remote_run="$workspace/$run_name"
script_dir="$root/experiments/academic_a4/unity"
temporary=$(mktemp -d /tmp/gradflow-unity-stage.XXXXXX)
bundle="$temporary/gradflow-rc2.bundle"

cleanup() {
    rm -f "$bundle"
    rmdir "$temporary" 2>/dev/null || true
}
trap cleanup EXIT

case "$run_name" in
    ""|*[!A-Za-z0-9._-]*)
        echo "UNITY_RUN_NAME may contain only letters, digits, dot, underscore, and hyphen" >&2
        exit 2
        ;;
esac

git -C "$root" rev-parse --verify "$tag^{commit}" >/dev/null
git -C "$root" bundle create "$bundle" "$tag"
git bundle verify "$bundle"

ssh_options=(-o BatchMode=yes)
if test -n "${UNITY_IDENTITY_FILE:-}"; then
    ssh_options+=(-i "$UNITY_IDENTITY_FILE")
fi

ssh "${ssh_options[@]}" "$target" \
    "test -d '$workspace' && test ! -e '$remote_run' && mkdir -m 700 '$remote_run' '$remote_run/controller'"
scp "${ssh_options[@]}" "$bundle" "$target:$remote_run/gradflow-rc2.bundle"
scp "${ssh_options[@]}" \
    "$script_dir/run_second_machine.py" \
    "$script_dir/setup_environment.sbatch" \
    "$script_dir/replicate.sbatch" \
    "$script_dir/submit_unity.sh" \
    "$target:$remote_run/controller/"
ssh "${ssh_options[@]}" "$target" \
    "git clone '$remote_run/gradflow-rc2.bundle' '$remote_run/repo' >/dev/null && git -C '$remote_run/repo' checkout --detach '$tag' >/dev/null && git -C '$remote_run/repo' status --porcelain=v1 > '$remote_run/INITIAL_STATUS.txt' && test ! -s '$remote_run/INITIAL_STATUS.txt' && sha256sum '$remote_run/gradflow-rc2.bundle' > '$remote_run/STAGED_SHA256SUMS'"

printf 'staged_run=%s\n' "$remote_run"
printf 'submit_command=cd %s/controller && bash submit_unity.sh\n' "$remote_run"
