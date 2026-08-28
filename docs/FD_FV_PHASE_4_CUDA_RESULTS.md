# FD/FV Phase-4R Forge CUDA replication result

Status: **fresh CUDA admission passed; resident RTX 5070 Ti matrix complete**.

Measurement date: 2026-08-28 UTC.

Measurement source commit:
`ba646aa757d2fee63a9c7369ed106571c3f699b9`.

The immutable aggregate is
`experiments/fd_fv_bakeoff/results/phase_4r_cuda_20260828/replication_cuda.json`,
SHA-256
`37c616695fd89b088dd301a21575116fd10ac0caf12180b2c0751f277cd99dcf`.
The independent verifier recomputes all 2,400 CUDA-event samples and verifies
the fresh six-case admission, 24 raw workers, aggregate, and 26-entry manifest.

## Visibility correction

Forge physically contains a working NVIDIA GeForce RTX 5070 Ti. The default
Codex execution sandbox did not expose `/dev/nvidia*`, so `nvidia-smi` could
not communicate with the driver and PyTorch reported zero CUDA devices inside
that sandbox. An explicitly permitted host execution saw the GPU immediately.

The earlier CPU Phase-4R record remains immutable and correctly records what
its process could observe, but `untested_unavailable` must not be interpreted
as “Forge has no CUDA.” This linked supplement performs the conditional CUDA
stratum on the same host outside that device-isolation boundary.

## Admission and environment

Fresh admission passed for both formulations at the largest frozen 1-D, 2-D,
and 3-D Phase-4A sizes. Every case retained one graph with zero graph breaks,
resident CUDA outputs, finite float64 values, and CPU/GPU plus compiled/eager
maximum absolute differences no larger than `2.22e-16` during admission.

The measured environment was:

- NVIDIA GeForce RTX 5070 Ti, 15.47 GiB visible memory;
- compute capability 12.0, 70 streaming multiprocessors;
- NVIDIA driver 580.173.02;
- PyTorch 2.13.0+cu130 with CUDA runtime 13.0; and
- float64 state and arithmetic under the frozen formulation policy.

Consumer-GPU FP64 throughput limits remain relevant to absolute CPU/GPU
interpretation. They do not invalidate the matched FD/FV comparison because
both methods ran on the same device under the same dtype policy.

## Resident execution result

Each value is the median of three independent worker medians. Every worker
used a fresh TorchInductor cache, ten warmups, fifty CUDA-event samples, and
device-resident state. Transfers and compilation are excluded from these
latencies; first-call compilation was recorded separately. `FV/FD > 1`
favors FD.

| N | Cells | FD compiled ms | FV compiled ms | FV/FD | Decision |
|---:|---:|---:|---:|---:|---|
| 18 | 5,832 | 0.238 | 0.247 | 1.040 | unresolved |
| 27 | 19,683 | 0.446 | 0.493 | 1.105 | FD faster |
| 40 | 64,000 | 1.075 | 1.177 | 1.095 | FD faster |
| 64 | 262,144 | 3.678 | 4.125 | 1.122 | FD faster |

The paired ratios were tightly replicated. At `N=64`, for example, they were
`1.1214`, `1.1218`, and `1.1216`. Compiled coefficients of variation fell from
approximately 2--4% at the smallest points to about 0.2--0.3% at `N=64`.
Nothing resembles the multimodal CPU `N=27` FV timing.

Eager CUDA was unresolved at every size. Its approximately `6.3 ms` launch-
dominated latency through `N=40` shows why “PyTorch on a GPU” is not itself a
performance result. TorchInductor's compiled path was essential: the compiled
FD speedup over eager ranged from approximately `26.3x` at `N=18` to `2.19x`
at `N=64`; FV ranged from approximately `25.7x` to `1.93x`.

## CPU relationship

At the three sizes shared with the CPU replication, compiled resident CUDA FD
was approximately `4.65x`, `6.82x`, and `9.39x` faster than the corresponding
six-thread CPU process medians at `N=18`, `27`, and `40`. For FV the stable
endpoints were approximately `5.09x` and `10.00x` faster at `N=18` and `40`;
the `N=27` CPU denominator is deliberately not promoted because Phase 4R
showed that it was non-stationary.

These are warm, device-resident one-step comparisons. They exclude host/device
transfers and roughly 18--20 seconds of first-call compilation, so they do not
answer cold “press run and receive a host answer” latency. They do establish
that the RTX materially accelerates this 3-D float64 scalar step once state is
resident and compilation has been paid.

## Compiler and memory evidence

All FD workers emitted 54 generated kernels and 229 pre-fusion IR nodes. All
FV workers emitted 57 kernels and 187 pre-fusion IR nodes. These identities
were invariant across size and replicate, and the tightly replicated timing
contains no `N=27` compiler cliff.

Compiled peak allocated CUDA memory was consistently higher for FV:

| N | FD MiB | FV MiB | FV/FD |
|---:|---:|---:|---:|
| 18 | 2.02 | 2.56 | 1.268 |
| 27 | 6.77 | 8.57 | 1.267 |
| 40 | 21.97 | 27.83 | 1.267 |
| 64 | 90.00 | 114.00 | 1.267 |

This is compiler-observed peak allocation for the timed step, not a general
proof that all FV implementations require 26.7% more memory. It is a stable
property of these matched ordinary-PyTorch expressions on this compiler.

## Scientific interpretation

The CUDA replication resolves the original surprise in two ways:

1. the CPU `N=27` FV mode does not transfer to the GPU backend; and
2. the GPU distributions are stable enough to support a bounded conclusion.

For this smooth periodic 3-D linear-advection problem, float64 policy, RTX
5070 Ti, and TorchInductor version, compiled classical FD is approximately
9--12% faster than the matched dimension-by-dimension FV formulation at
`N=27--64`, while the smallest point is unresolved. FV also uses about 26.7%
more peak allocated device memory. This is useful evidence for a conditional
phase diagram, not a universal FD-superiority claim.

Compilation and transfer-inclusive endpoints, nonlinear scalar flow, Euler,
other FV classes, other GPUs, and data-center FP64 hardware remain separate
questions.
