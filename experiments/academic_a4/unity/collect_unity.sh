#!/bin/bash
set -euo pipefail

if test "$#" -ne 1; then
    echo "usage: $0 RUN_NAME" >&2
    exit 2
fi

root=$(git rev-parse --show-toplevel)
workspace=/work/pi_zchen2_umassd_edu/hshu
target=${UNITY_SSH_TARGET:-hshu_umassd_edu@login.unityhpc.org}
run_name=$1
destination="$root/experiments/academic_a4/evidence/$run_name"

case "$run_name" in
    ""|*[!A-Za-z0-9._-]*) echo "invalid run name" >&2; exit 2 ;;
esac
if test -e "$destination"; then
    echo "refusing existing destination: $destination" >&2
    exit 3
fi

ssh_options=(-o BatchMode=yes)
if test -n "${UNITY_IDENTITY_FILE:-}"; then
    ssh_options+=(-i "$UNITY_IDENTITY_FILE")
fi

mkdir -m 700 "$destination"
rsync -a --protect-args -e "ssh ${ssh_options[*]}" \
    "$target:$workspace/$run_name/evidence/" "$destination/"
python3 "$root/experiments/academic_a4/unity/verify_second_machine.py" \
    "$destination"
printf 'collected=%s\n' "$destination"

