# G5 shared-pencil memory-recovery results

Date: 2026-08-29 (UTC)

## Decision

The frozen P1 candidate **does not establish a successful speed-memory Pareto
result**.

P1 removes R6Q's three whole-grid directional face arrays and produces
bitwise-identical output on every pre-timing specimen. At `N=128`, declared
peak allocation falls from 336,134,148 bytes to 210,305,028 bytes: P1 uses
62.57% of R6Q memory, a 37.43% reduction.

The speed price is decisive. At the two primary points, paired median resident
`P1 / R6Q` ratios are 2.619 for one step and 2.701 for ten steps. Their
bootstrap 95% intervals are `[2.615, 2.622]` and `[2.699, 2.706]`. Both are far
above the frozen 1.10 median and 1.15 upper-bound limits. Cell-recompute also
beats P1 at every resident-time point.

This is useful negative evidence about one execution schedule. It is not a
claim that all shared-memory tiling is inferior, and it does not qualify or
disqualify any GradFlow backend.

## Correctness and memory gate

P1 and frozen R6Q were run from exact shared FP32 input bytes. All comparisons
were bitwise identical:

| Specimen | Mode | Result |
| --- | --- | --- |
| periodic vortex, `32^3`, one step | SSP-RK3 | exact |
| periodic vortex, `32^3`, ten steps | SSP-RK3 | exact |
| perturbed vortex, `32^3`, one step | SSP-RK3 | exact |
| dual-interface Shu--Osher type, `32^3`, ten steps | SSP-RK3 | exact |
| smooth entropy wave, `40^3` | RHS | exact |

Every stepped state remained finite with positive density and pressure. This
establishes schedule parity with R6Q only. R6Q's earlier G3 RHS relative-RMS,
zero-budget conservation, and missing-autograd failures remain unchanged.

The gate runner recorded its first (`N=32`) allocation metadata before timing,
where P1 used 62.76% of R6Q. The protocol named `N=128`; the subsequent first
`N=128` observations explicitly recorded 210,305,028 and 336,134,148 bytes,
or 62.57%. The earlier check was slightly more conservative, so this recording
deviation could not turn a memory failure into a pass, but it is disclosed.

## Resident performance

Times are medians of 30 randomized fresh-process triplets after three separate
warm-up processes per lane and configuration. The table reports numerical-loop
CUDA-event time.

| Grid | Steps | P1 pencil (ms) | R6Q global faces (ms) | Cell recompute (ms) | P1 / R6Q | Cell / P1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `32^3` | 1 | 0.804 | 0.457 | 0.529 | 1.756 | 0.658 |
| `32^3` | 10 | 4.370 | 1.569 | 2.911 | 2.787 | 0.666 |
| `64^3` | 1 | 2.011 | 0.886 | 1.337 | 2.275 | 0.665 |
| `64^3` | 10 | 16.650 | 6.048 | 12.369 | 2.753 | 0.743 |
| `128^3` | 1 | 13.143 | 5.018 | 9.608 | 2.619 | 0.731 |
| `128^3` | 10 | 127.913 | 47.363 | 94.374 | 2.701 | 0.738 |
| `192^3` | 1 | 53.533 | 16.546 | 31.483 | 3.234 | 0.588 |
| `192^3` | 10 | 532.525 | 162.520 | 313.695 | 3.278 | 0.589 |
| `256^3` | 1 | 114.059 | 38.126 | 73.580 | 2.992 | 0.645 |
| `256^3` | 10 | 1,141.558 | 379.644 | 737.336 | 3.007 | 0.646 |

The full P1/R6Q range is 1.756--3.278. Increasing the line length does not
rescue the schedule: even at `N=256`, where every thread in the frozen
256-thread block owns a coordinate, P1 takes about three times R6Q.

## Button-to-answer endpoint

Fresh-process startup and input handling dominate the smallest cases. At the
primary one-step point the medians are P1 243.8 ms, R6Q 235.8 ms, and
cell-recompute 268.0 ms. At ten steps they are 355.5, 279.4, and 349.2 ms.
At `256^3`, the corresponding one-step values are 756.5, 679.9, and 925.1 ms;
ten-step values are 1,787.9, 1,027.8, and 1,594.6 ms.

P1 can therefore appear better than cell-recompute for an isolated process at
some one-step points because executable startup and allocation overwhelm the
numerical schedule. It never beats R6Q, and it loses to cell-recompute once
enough numerical work is performed. This does not alter the resident Pareto
decision.

## Causal profiler characterization

Nsight Systems 2025.3.2 recorded exactly the expected 20 numerical launches:
one CFL scan, one CFL finish, nine line-alpha reductions, and nine pencil
kernels. At `128^3`, one step:

| Kernel class | Launches | Total (ms) | Share |
| --- | ---: | ---: | ---: |
| P1 pencil | 9 | 11.433 | 89.6% |
| line alpha | 9 | 1.287 | 10.1% |
| CFL scan/finish | 2 | 0.036 | 0.3% |

The pencil kernel uses 128 registers per thread, one barrier, 2.5 KiB dynamic
shared memory at `N=128`, and has zero reported spills. Thus spilling is not
the observed cause.

The per-launch trace gives a stronger explanation. Across the three RK stages,
x-pencil kernels total 1.340 ms, y pencils 5.757 ms, and z pencils 4.336 ms.
P1 maps neighboring threads along the chosen pencil, so y and z state accesses
are strided in the component-major `(z,y,x)` layout. R6Q instead maps linear
threads over cells, keeping corresponding stencil loads coalesced while each
thread computes all three axes.

For comparison, frozen G4 `N=128` traces total 4.680 ms for R6Q kernels and
9.338 ms for cell-recompute kernels. P1 totals 12.756 ms.

The initial unprivileged Nsight Compute 2025.3.1 attempt returned
`ERR_NVGPUCTRPERM`. A subsequent explicitly authorized one-time `sudo` run
collected the frozen Basic set without changing Forge's persistent driver
permissions. Its pencil-kernel medians and three-stage duration totals are:

| Axis | Total replay duration (ms) | L2 throughput | DRAM throughput | SM compute | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: |
| x | 1.660 | 6.30% | 13.53% | 48.25% | 32.29% |
| y | 6.570 | 64.53% | 5.70% | 12.27% | 27.33% |
| z | 5.050 | 91.81% | 10.25% | 18.74% | 27.71% |

The 128-register kernel is limited to two resident blocks per SM and 33.33%
theoretical occupancy; shared memory would permit four blocks and warps six.
The counters therefore establish two interacting costs: register pressure
limits latency hiding, while the y/z mappings move the bottleneck away from
arithmetic and onto L2/cache-side memory service. Low DRAM utilization shows
that raw off-card bandwidth is not the limiting resource. This strongly
corroborates the known direction-strided access pattern as the cause, although
the Basic counter set does not directly measure every memory transaction.

The 3.522-second application time printed under Nsight Compute is deliberately
excluded from performance results: each of 20 kernels was replayed for nine
counter passes. The state checksum remained unchanged.

## Interpretation and next boundary

G4 showed that globally materializing each face once is about twice as fast as
recomputing both adjacent faces, at about twice the workspace. G5 shows that
the most literal one-block-per-line attempt to recover that workspace is not
competitive on this layout and GPU. Saving global face traffic is insufficient
when the replacement destroys access coalescing and pays nine expensive
directional pencil launches.

The scientific conclusion is narrow but actionable:

- retain global face-once R6Q as the current throughput schedule control;
- reject P1 as an implementation candidate without post-result tuning;
- do not infer that shared memory or fused updates are generally bad; and
- require any later memory-recovery hypothesis to solve direction-appropriate
  coalescing explicitly and to be frozen as a new experiment.

No arbitrary-order work, backend admission, production optimization, or
publication claim begins in G5.

## Verification

- Ruff passed for the G5 gate, campaign, verifier, and regression test.
- The G3, G4, and G5 evidence regression tests passed together.
- The full source-path suite completed with 278 passed and 72 skipped. The same
  eight historical environment-sensitive frozen-record tests that preceded G5
  failed (Phase 5A/6A byte regeneration and their dependent Phase 6 verifier
  chain). No failure exercises G5 source or evidence.
- The G5 checksum and semantic verifier passed.
