# G6 exact-math occupancy-ablation results

## Outcome

G6 found **no meaningful exact-math occupancy improvement**. All nine
factorial candidates were bitwise identical to the frozen R6Q control on all
five forward specimens, but none passed the preregistered performance rule at
both primary points. R6Q remains a non-admitted schedule control.

The experiment also answers a narrower engineering question: neither a lower
thread count nor a compiler register ceiling is a useful standalone remedy
for this kernel. The only variants that raised theoretical occupancy from
33.33% to 41.67% forced register spills and did not improve performance at
the primary or large-grid points.

## Frozen experiment

The exact R6Q mathematics, state layout, global face arrays, update schedule,
strict-FP32 build policy, and 256-thread non-face kernels were fixed. G6
crossed face blocks `{64, 128, 256}` with register policies `{uncapped, 112,
96}`. The campaign used three fresh-process warmups and 30 randomized complete
ten-lane blocks at `N={64,128,256}`, one and ten SSP-RK3 steps.

## Correctness

All 45 candidate/control comparisons were bitwise identical:

- periodic vortex, `N=32`, one and ten steps;
- perturbed vortex, `N=32`, one step;
- dual-interface Shu--Osher-type state, `N=32`, ten steps; and
- smooth entropy-wave RHS, `N=40`.

Every stepped state remained finite with positive density and pressure. No
candidate changes WENO algebra, and no candidate is admitted as a backend.

## Compiler and declared occupancy

| Register policy | Face registers/thread | Local bytes/thread | Spill stores / loads | Occupancy at 64 threads | Occupancy at 128 threads | Occupancy at 256 threads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| uncapped | 128 | 0 | 0 / 0 | 33.33% | 33.33% | 33.33% |
| cap 112 | 112 | 0 | 0 / 0 | 33.33% | 33.33% | 33.33% |
| cap 96 | 96 | 40 | 80 B / 88 B | 41.67% | 41.67% | 33.33% |

The 112-register ceiling changes the register count but not the residency
step: the 256-thread kernel still fits two blocks per SM. The 96-register
ceiling permits five 128-thread or ten 64-thread blocks, but does so by
spilling live values to local memory. At 256 threads it still fits only two
blocks, so even its theoretical occupancy remains unchanged.

## Preregistered performance decision

The protocol compares each candidate with the older frozen R6Q executable.
At the two primary points:

| Candidate | `N=128`, 1-step paired median ratio | `N=128`, 10-step paired median ratio | Both-point rule |
| --- | ---: | ---: | --- |
| `b256_r112` | 0.94360 | 0.99365 | Fail |
| `b256_u` | 0.94416 | 0.99407 | Fail |
| `b128_r96` | 0.95172 | 1.00058 | Fail |
| `b128_u` | 0.95411 | 1.00389 | Fail |
| `b128_r112` | 0.95449 | 1.00369 | Fail |
| `b64_r96` | 0.95914 | 1.00844 | Fail |
| `b64_u` | 0.96456 | 1.01395 | Fail |
| `b256_r96` | 0.96498 | 1.01644 | Fail |
| `b64_r112` | 0.96512 | 1.01403 | Fail |

The frozen ranking rule therefore selected `b256_r112` for bounded profiling,
but did not select it for use.

## Causal interpretation

`b256_u` is an essential negative control: it retains the frozen 256-thread,
uncapped compilation but includes G6's resource queries before the measured
event. Every rebuilt G6 lane appears faster than the older frozen executable
in short runs. That shared effect is not an occupancy result; the metadata
queries move lazy CUDA function/module setup out of the event interval. The
effect falls from about 5.6% at `N=128`, one step, to about 0.6% over ten
steps.

The causal comparisons are therefore the tuned lanes against rebuilt
`b256_u`:

| Candidate / `b256_u` paired median | `64^3`, 1 / 10 steps | `128^3`, 1 / 10 steps | `256^3`, 1 / 10 steps |
| --- | ---: | ---: | ---: |
| `b256_r112` | 0.99763 / 0.99865 | 0.99989 / 0.99966 | 0.99947 / 0.99935 |
| `b256_r96` | 1.02532 / 1.02192 | 1.02254 / 1.02253 | 1.02102 / 1.02149 |
| `b128_u` | 1.00097 / 1.00050 | 1.01126 / 1.00988 | 1.01008 / 1.01042 |
| `b128_r96` | 0.99833 / 0.99494 | 1.00870 / 1.00647 | 1.01858 / 1.02107 |
| `b64_u` | 1.00287 / 1.00098 | 1.02188 / 1.01989 | 1.05187 / 1.05171 |
| `b64_r96` | 1.00819 / 1.00211 | 1.01629 / 1.01442 | 1.04278 / 1.04618 |

The 112-register cap at 256 threads is reproducible parity, not a useful
speedup. The 96-register variants pay about 2% at moderate/large scale despite
their nominal occupancy gain. Smaller blocks increasingly lose as the grid
grows.

## Profiler corroboration

Nsight Systems gives essentially identical numerical kernels for frozen R6Q
and `b256_r112`: face totals are 2.526008 and 2.527223 ms respectively.
Nsight Compute reports:

| Face-kernel metric (three-stage median unless total) | Frozen R6Q | `b256_r112` |
| --- | ---: | ---: |
| Registers/thread | 128 | 112 |
| Theoretical occupancy | 33.33% | 33.33% |
| Achieved occupancy | 32.39% | 32.44% |
| SM compute throughput | 72.99% | 73.18% |
| DRAM throughput | 16.44% | 16.42% |
| Three-launch replay duration | 3.17 ms | 3.16 ms |

The register count falls, but active blocks, theoretical occupancy, achieved
occupancy, compute use, memory use, and duration do not materially change.
The privileged application times (about 2.951 seconds per lane) are profiler
replay overhead and are not benchmark observations.

## Conclusion and next boundary

Occupancy is a constraint, not an objective function. This exact face kernel
already runs near its 33.33% theoretical occupancy and reaches roughly 73% SM
throughput. Lowering registers without crossing a residency threshold does
nothing; crossing one by spilling loses the benefit. U0's lower register use
was incidental to its unsafe, much simpler numerical work, not evidence of a
successful occupancy design.

Further work, if preregistered separately, must change live-value structure
rather than apply a blunt compiler cap. Plausible experiments include
warp-distributed characteristic work, shorter variable lifetimes, or a
coalesced multidirectional tiling scheme. They are not part of G6, and no such
candidate has begun here.

## Verification

- The G6 checksum and semantic verifier passed.
- Ruff passed for the gate, campaign, counter reducer, verifier, and test.
- The G3, G4, G5, and G6 evidence regression tests passed together (4 tests).
- The full suite reported 279 passed, 72 skipped, and eight failures. The same
  eight historical environment-sensitive frozen-record failures precede G6:
  Phase 5A/6A exact regeneration differs in this environment, and downstream
  Phase 6B--6E verifiers inherit those failures or a subprocess import-path
  limitation. No failure exercises G6 source or evidence.
