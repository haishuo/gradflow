# Unity second-machine replication

This directory adapts the frozen Academic A4 second-machine contract to the
UMass Unity SLURM cluster. It is orchestration only: numerical source is run
from a detached clean checkout of tag `academic-v0.1.0-rc2`.

The designated persistent root is:

```text
/work/pi_zchen2_umassd_edu/hshu
```

The Unity login directory is used only to submit and inspect jobs. Neither
environment creation nor scientific execution runs on a login node.

## Frozen Unity choices

- one physically distinct Unity compute node;
- one NVIDIA GPU in the non-preemptible `gpu` partition;
- six CPU cores and 64 GiB host memory;
- stable PyTorch `2.13.0` from the CUDA 12.6 wheel index recommended by Unity;
- all rc2 clean-checkout sentinels;
- the complete A1 numerical-limit campaign;
- the complete A3 CPU/CUDA inverse and gradient campaign; and
- three fresh A2 workers for every combination of order 5/11/15,
  binary32/binary64, and CPU/CUDA at `64^3`.

The GPU model is not constrained by default. Its identity is evidence, not a
selection made after observing results. To add a prospective constraint such
as A100, pass it to `sbatch` before execution and record it in the command
ledger. Exact Forge timings are not a replication target.

## Stage from Forge

The staging helper creates a git bundle from the frozen tag, makes a unique
run directory under the designated Unity workspace, copies the controller,
and checks out the tag remotely. It never pushes either repository.

```bash
UNITY_SSH_TARGET=hshu_umassd_edu@login.unityhpc.org \
  bash experiments/academic_a4/unity/stage_unity.sh
```

If Unity uses a different registered SSH key, set `UNITY_IDENTITY_FILE` or use
an SSH agent/configured alias. The helper prints the resulting run directory.

## Submit on Unity

From the staged controller directory on the Unity login node:

```bash
bash submit_unity.sh
```

This submits a CPU environment-build job and a dependent GPU replication job.
The latter writes only beneath the staged run directory and the node-local
temporary directory. Use `squeue -j JOB_ID` and the recorded SLURM logs to
monitor it.

## Result status

`second_machine.json` records one of:

- `pass`;
- `pass_with_limitations`; or
- `fail_needs_investigation`.

Failures, exclusions, stdout, stderr, exit codes, durations, environment
identity, and SHA-256 hashes are retained. The controller never relaxes a
tolerance or falls back from a failed compiled lane.

