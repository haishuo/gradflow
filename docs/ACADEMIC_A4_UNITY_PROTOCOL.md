# Academic A4 Unity replication protocol

Status: **frozen prospectively before Unity scientific execution**.

Date: 2026-08-31 (UTC)

## Purpose

This protocol instantiates the already frozen A4 second-machine contract on
the UMass Unity HPC cluster. Unity is physically distinct from Forge and uses
SLURM compute nodes. A successful run can close the second-machine scientific
gate; it cannot close the independent CFD/prior-art audit or redistribution
questions.

The governing order remains:

> correctness > performance > convenience

## Location and scheduler contract

All persistent files live below
`/work/pi_zchen2_umassd_edu/hshu`. The Unity login directory is used only for
staging, submission, and monitoring. Scientific work runs in SLURM jobs.

Environment construction uses a CPU job. Replication uses one node in the
non-preemptible `gpu` partition with one GPU, six CPU cores, 64 GiB memory,
and a 12-hour limit. No GPU model is selected after observing results. Any
prospective model constraint must be present in the submitted command ledger.

## Source and environment

The scientific checkout is detached at annotated tag
`academic-v0.1.0-rc2`, commit
`c5e8ab81ef5b33a2138b2db33afc538398b6f57f`, and must be clean. Source is
transported as a verified git bundle; no GitHub push is required.

The environment uses Python 3.10 or newer and stable PyTorch 2.13.0 from
Unity's recommended CUDA 12.6 wheel index. Exact Python, PyTorch commit, CUDA
runtime, driver, GPU, CPU, operating system, packages, SLURM allocation,
memory, and thread variables are recorded. Environment setup failure is
retained and investigated; another wheel is not silently substituted.

## Frozen execution surface

The replication runs:

1. the complete rc2 test suite and A1/A2/A3/U5/A4 offline sentinels;
2. the complete A1 numerical-limit campaign;
3. the complete A3 inverse/gradient campaign on CPU and CUDA; and
4. three fresh isolated A2 workers for every combination of scalar WENO-JS
   order 5/11/15, binary32/binary64, and CPU/CUDA at `64^3`.

Each A2 worker retains five warmups, 20 randomized eager/compiled pairs,
correctness admission, graph records, compiler failures, CUDA telemetry,
resident timings, and the existing transfer slice. Every fresh worker receives
an empty node-local TorchInductor cache. CPU thread observations remain the
existing one- and six-thread worker contract.

No tolerance, numerical source, warmup count, repetition count, compiler
option, shape, or dtype may change after execution begins.

## Decision contract

Exact Forge timing is not a target. A `pass` requires:

- all clean-checkout sentinels pass;
- A1 completes under its frozen health rules;
- A3 autograd, centered differences, derivative-free recovery, and compiled
  CPU/CUDA lanes pass their existing tolerances;
- all 36 A2 fresh workers are present;
- every admitted compiled worker reports one graph and zero graph breaks;
- at least one binary32 `64^3` cell shows CUDA below 0.95 times the fastest
  CPU median; and
- no A2 correctness admission failure occurs.

If the core qualitative contract passes but one or more A2 lanes fail their
frozen numerical admission, status is `pass_with_limitations`. Any missing
core requirement is `fail_needs_investigation`. All outcomes are evidence.
Binary64 is reported as a property of the allocated Unity accelerator; no
cross-hardware FP64 generalization follows from one run.

## Evidence and stop condition

The controller preserves raw stdout/stderr, commands, return codes, timeouts,
durations, parsed records, environment identity, SLURM identity, qualification
decisions, and SHA-256 sums. The offline verifier checks the source identity,
machine independence, record cardinality, status consistency, and every file
hash.

Stop after one frozen run is complete or retained as a failure. Do not tune,
rerun selectively, or change GPU constraints in response to timings. A wholly
new replication requires a new prospective run identity and written reason.

