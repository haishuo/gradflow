# G4 face-once schedule performance results

Date: 2026-08-29 (UTC)

Hardware: NVIDIA GeForce RTX 5070 Ti, driver 580.173.02, CUDA 13, `sm_120`

## Decision

The preregistered face-once scheduling hypothesis is supported at both primary
points. R6Q was approximately 1.91 times faster for one `128^3` SSP-RK3 step
and 1.99 times faster for ten steps than the exact-math cell-recompute control.

R6Q remains a non-admitted research candidate. G4 establishes a scheduling
effect; it does not reverse the G3 gate failures, add differentiation, or
qualify a GradFlow backend.

## Primary result

All values are medians of 30 randomized fresh-process pairs. Resident time is
the synchronized CUDA-event numerical-loop endpoint.

| Grid and workload | Face-once | Cell-recompute | Paired control / face median | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| `128^3`, 1 step | 5.021 ms | 9.608 ms | 1.913x | [1.911, 1.915] |
| `128^3`, 10 steps | 47.363 ms | 94.376 ms | 1.993x | [1.992, 1.993] |

Both paired medians exceeded the frozen 1.10 requirement and both lower
confidence bounds exceeded 1.0. The earlier provisional one-step ratio of
2.047x reduced to 1.914x under randomized fresh-process pairing; the sustained
approximately 2x effect replicated.

## Scaling and crossover

| N | 1-step face / control (ms) | Control / face | 10-step face / control (ms) | Control / face |
|---:|---:|---:|---:|---:|
| 8 | 0.438 / 0.389 | 0.889x | 1.098 / 1.403 | 1.282x |
| 16 | 0.432 / 0.388 | 0.901x | 1.264 / 1.410 | 1.115x |
| 32 | 0.466 / 0.528 | 1.137x | 1.571 / 2.914 | 1.854x |
| 64 | 0.892 / 1.336 | 1.502x | 6.051 / 12.377 | 2.045x |
| 128 | 5.021 / 9.608 | 1.913x | 47.363 / 94.376 | 1.993x |
| 192 | 16.554 / 31.487 | 1.902x | 162.414 / 313.651 | 1.931x |
| 256 | 38.119 / 73.543 | 1.929x | 379.312 / 736.375 | 1.941x |

For a single step, the cell-recompute schedule wins at `N=8` and `N=16`; the
observed crossover occurs between `16^3` and `32^3`. Over ten steps,
face-once wins at every tested size because fixed launch/timing costs are
amortized and the repeated face algebra dominates. Above `64^3`, the resident
advantage settles near 1.9--2.0x rather than growing without bound.

## Fresh-process endpoint

CUDA context creation and process startup dominate the small cases: both lanes
take roughly 167 ms at `N=8` and `N=16`, so sub-millisecond schedule gains are
commercially invisible through this CLI endpoint.

At the primary and largest points, face-once also reduces full process time:

| Workload | Face-once | Cell-recompute | Paired control / face median |
|---|---:|---:|---:|
| `128^3`, 1 step | 237.9 ms | 269.2 ms | 1.132x |
| `128^3`, 10 steps | 279.4 ms | 349.3 ms | 1.257x |
| `256^3`, 1 step | 682.9 ms | 926.4 ms | 1.354x |
| `256^3`, 10 steps | 1027.0 ms | 1591.8 ms | 1.551x |

This does not make a standalone executable the intended commercial ABI. It
shows why persistent/device-resident integration is required to expose the
full schedule benefit at modest problem sizes.

## Memory tradeoff

Face-once buys arithmetic reuse by materializing three five-component face
arrays. Its declared peak workspace is therefore nearly twice the control's
at large grids:

| Grid | Face-once | Cell-recompute | Face / control |
|---:|---:|---:|---:|
| `128^3` | 336.1 MB | 172.3 MB | 1.95x |
| `192^3` | 1.134 GB | 576.5 MB | 1.97x |
| `256^3` | 2.687 GB | 1.360 GB | 1.97x |

This is not a universal replacement result. Memory-constrained workloads may
prefer recomputation or motivate a future tiled/shared-memory face-reuse
schedule. G4 does not implement that optimization.

## Causal profiler evidence

Nsight Systems recorded the same 17-kernel launch structure in both lanes. At
`N=128`, one step:

- face construction totaled 2.512 ms;
- cell-recompute RHS totaled 7.349 ms;
- face-once updates cost 0.847 ms versus 0.547 ms for the control; and
- total numerical kernel time was 4.680 ms versus 9.338 ms.

Thus the gain is localized to calculating the characteristic numerical faces
once; it is not caused by removing RK stages, alpha reductions, or launches.
Nsight Compute hardware counters were unavailable because Forge does not grant
unprivileged performance-counter access, so no occupancy or bandwidth claim is
made.

## Validity and run conditions

- The pre-timing common-input gate passed with maximum state difference
  `4.768e-7` against the `2e-5` bound.
- Every counted process reported finite output.
- All 420 randomized pairs were retained; no outlier was removed or rerun.
- Pair telemetry stayed in P0, with recorded temperatures from 46 C to 58 C
  and SM clocks from 2475 to 2902 MHz.
- First/second order-stratified medians were closely aligned; no material order
  effect explains the result.

## What G4 establishes

For this fixed FP32, periodic, three-dimensional Shu characteristic WENO-5
calculation on the RTX 5070 Ti:

1. duplicated cell-owned face reconstruction leaves about a factor of two of
   sustained numerical-loop performance unused at moderate and large grids;
2. unique-cell face ownership is a real GPU execution improvement, not an
   artifact of deleting mathematical work;
3. that speed is purchased with approximately twice the workspace;
4. small one-step jobs can still favor recomputation; and
5. process startup can hide most of the kernel advantage unless execution is
   persistent or already device-resident.

The result is specific to the tested formulation, precision, boundary type,
GPU, and implementations. It is not yet an arbitrary-order, cross-GPU, or
production-backend result.

## Repository verification

- Ruff passed for the G4 campaign, evidence verifier, and regression test.
- The G3 and G4 evidence regression tests both passed.
- The full repository suite reported 277 passed and 72 skipped. The same eight
  historical environment-sensitive frozen-record tests failed as before G4:
  two byte-exact NumPy regenerations and six dependent Phase 6 verifiers.
- No failing test exercises G4 code or evidence, and no historical record was
  rewritten to hide the environment difference.
