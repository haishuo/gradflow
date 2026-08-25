# Automatic DVEB matched bakeoff results

All values below are medians of 30 randomized-order fresh processes, in
milliseconds. The timed endpoint ends after complete state materialization
in pageable host memory. AOT build/calibration are excluded preparation
costs and are reported separately in the artifact manifest.

| N | Steps | Cells | Fortran | DVEB auto | PyTorch eager | PyTorch AOT | DVEB target | Winner | DVEB vs next best |
|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|
| 8 | 1 | 729 | 2.463 | 2.777 | 1154.685 | 2609.929 | cpu_simd[6] | fortran | 0.89x |
| 16 | 1 | 4,913 | 9.011 | 6.972 | 1159.174 | 2608.361 | cpu_simd[6] | dveb-auto | 1.29x |
| 32 | 1 | 35,937 | 55.640 | 51.206 | 1171.928 | 2636.620 | cpu_simd[6] | dveb-auto | 1.09x |
| 64 | 1 | 274,625 | 410.929 | 202.779 | 1228.164 | 2686.338 | cuda | dveb-auto | 2.03x |
| 96 | 1 | 912,673 | 1339.173 | 204.973 | 1316.977 | 2683.243 | cuda | dveb-auto | 6.43x |
| 128 | 1 | 2,146,689 | 3153.075 | 252.587 | 1612.691 | 2741.766 | cuda | dveb-auto | 6.38x |
| 32 | 10 | 35,937 | 517.043 | 174.894 | 1244.417 | 2611.619 | cuda | dveb-auto | 2.96x |
| 64 | 10 | 274,625 | 3875.661 | 197.349 | 1547.473 | 2696.431 | cuda | dveb-auto | 7.84x |
| 128 | 10 | 2,146,689 | 30582.411 | 343.178 | 5431.874 | 3376.829 | cuda | dveb-auto | 9.84x |

## Decision

DVEB wins 8 of the 9 declared regions. For one-step work it selects
`cpu_simd[6]` at N=8, 16, and 32, then CUDA at N=64, 96, and 128.
For ten-step work it selects CUDA at every tested size, beginning at
N=32. Thus placement depends on both grid size and timestep count.

Fortran wins only at N=8 / one step: 2.463 ms versus DVEB's 2.777 ms,
with Fortran ahead in 27 of 30 paired repetitions. DVEB wins at N=16
in 26 of 30 pairs and at every larger or longer point in 30 of 30.
The N=32 / one-step medians differ by 8.7%, so both qualify as
competitive under the frozen 10% rule even though DVEB wins all pairs.

At N=128 / ten steps, DVEB's 0.343 s median is 9.84x faster than AOT PyTorch, 15.83x faster than eager PyTorch, and 89.12x faster than Fortran at the complete start-to-finish endpoint.

This validates a bounded reason for DVEB to exist in WENO: one
target-neutral source produced competitive native CPU code and
ceiling-class CUDA code, and application-specific calibrated dispatch
selected the winning family at these points. It is not evidence that
DVEB wins other formulations or machines, or that its general selector
is production-qualified.

## Preparation and limitations

The fixed-shape AOT packages took 30.23–36.69 s each to build and 5.66–6.13 s for the recorded first extraction/preparation runs. DVEB calibration
also ran before timing; its per-observation raw records are committed,
but this first harness did not record one enclosing wall-clock duration
for calibration including warmups.

The selector was calibrated at the same grid/step points with separate
observations. That is valid application-specific profile-guided
deployment evidence, not a held-out generalization test. DVEB's later
generic disjoint-point campaign at commit `2f1f3ab` recorded NO-GO for
the initial automatic selector because fresh-process maximum regret and
CPU-schedule proximity missed their frozen bands. This WENO result does
not override that decision.

The campaign covers one float32 3-D Shu
Euler WENO-5 workload, one vortex, and one Ryzen 7600X / RTX 5070 Ti
machine. The frozen DVEB executable came from an uncommitted compiler
worktree state. DVEB subsequently committed and requalified a different
final artifact; therefore this exact run is auditable by binary hash but
is not a clean-source reproduction of that later artifact.

Full-array correctness maximum: `7.152557373e-07` (bound `2.0e-05`).

The paired-win counts and SHA-256 identities for every raw result are in
`dveb_bakeoff_20260825.json`. That record also hashes the exact
GradFlow PyTorch source, Fortran source/executable, protocol, harness,
DVEB executable, and placement model used by the campaign.
