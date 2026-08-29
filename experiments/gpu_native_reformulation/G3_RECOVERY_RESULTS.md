# G3 GPU-native correctness-recovery results

Date: 2026-08-29 (UTC)

Hardware: NVIDIA GeForce RTX 5070 Ti, CUDA 13, `sm_120`

## Result in one sentence

The deliberately inaccurate U0 frontier could be brought back to the Shu
characteristic FD-WENO-5 contract without giving up its face-once GPU schedule:
R6 agrees with the float64 oracle to `3.51e-7` after one step and `1.67e-6`
after ten steps at `N=32`, while taking `4.709 ms` for one `128^3` SSP-RK3 step
and `47.227 ms` for ten.

R6 is an experimental native-CUDA result, not yet a qualified GradFlow backend.

## Causal ladder

Error columns compare the frozen `N=32` output with the appropriate float64
qualified characteristic WENO-5 oracle. FE candidates use the same first
timestep; SSP-RK3 candidates use the qualified temporal oracle.

| Candidate | Restored property | `N=128` median | Max error | RMS error / oracle update | Update cosine | Face registers |
|---|---|---:|---:|---:|---:|---:|
| U0 | unsafe frontier | 0.515 ms | `3.27e-4` | 0.839138 | 0.591607 | 72 |
| R1 | strict FP32 arithmetic | 0.854 ms | `3.27e-4` | 0.839132 | 0.591611 | 94 |
| R2 | separate component weights | 1.076 ms | `6.31e-4` | 1.198478 | 0.542289 | 96 |
| R3 | Roe characteristic projection | 1.992 ms | `1.68e-4` | 0.384036 | 0.926769 | 168 |
| R4 | line-family LF speeds, 1.1 enlargement | 1.874 ms | `1.79e-5` | 0.026935 | 0.999648 | 128 |
| R5 | SSP-RK3 | 5.695 ms | `1.77e-5` | 0.026913 | 0.999648 | 128 |
| R6 | Shu difference form and epsilon scaling | 4.709 ms | `3.51e-7` | 0.001449 | 0.999998954 | 128 |

All candidates remained finite and positive on the periodic-vortex specimen.
All compiled kernels reported zero register spilling.

### What caused the damage

- Fast-math semantics were not the cause. R1 cost 66% more than U0 at
  `N=128` and recovered no meaningful accuracy.
- Separate componentwise weights were not a bridge to the characteristic
  oracle on this problem. R2 was both slower and farther away.
- Characteristic projection was important: R3 recovered more than half of the
  missing update and corrected its direction substantially.
- The ancestral line-family LF policy was decisive: R4 reduced the remaining
  update error by about 14x and nearly aligned the update direction.
- Shu's difference-form correction was both more faithful and faster than the
  algebraically related conventional reconstruction in this compiled kernel.
  It restored the exact indicator scaling relative to epsilon and reduced the
  one-step maximum error by about 51x from R5.

The main conclusion is therefore not that the mathematics should be made
reckless. The reusable win came from changing the **execution schedule**:
unique periodic storage and one owner per directional face, while restoring
the qualified characteristic spatial and temporal mathematics.

## R6 accumulated result

At `N=32`, ten R6 steps had:

- maximum absolute error: `1.6694e-6`;
- RMS error / oracle update RMS: `7.1274e-4`;
- update-magnitude ratio: `1.0000138`;
- update cosine: `0.999999746`;
- finite output, minimum density `0.4932502`, minimum pressure `0.3717719`.

The ten-step `N=128` median was `47.2268 ms`, or `4.7227 ms/step`, showing no
meaningful one-step launch-amortization illusion.

## Provisional native-control comparison

The frozen qualified cell-recompute CUDA control recorded:

| `N=128` workload | Cell-recompute control | R6 face-once | Ratio |
|---|---:|---:|---:|
| 1 SSP-RK3 step | 9.6385 ms | 4.7088 ms | 2.047x |
| 10 SSP-RK3 steps | 94.9104 ms | 47.2268 ms | 2.010x |

This is strong evidence that shared-face recomputation left substantial
performance unused. It is still provisional because the control was measured
in the earlier E4 campaign rather than randomized in the same measurement
blocks. Both used the same physical GPU and resident numerical-loop endpoint,
but R6 uses `N^3` unique cells while the control stores `(N+1)^3` duplicated
endpoints. That storage difference is an intended execution optimization and
accounts for only part of the approximately 2x result.

R6 uses 17 launches per step, the same launch count class as the qualified
control: CFL reduction and finish, then three stages of three line-alpha
reductions, face construction, and update. Its observed gain is therefore not
from silently deleting RK stages or LF reductions.

## Thermal interruption record

Forge was restarted after G2 for an unknown loss of responsiveness. No
experiment process was active at the recorded interruption boundary. All G1
checksums verified after restart. The GPU was 40 C and idle before recovery,
41 C after R1/R2, and 42 C and idle after the ten-step R6 runs. These readings
do not establish the cause of the host interruption, but the short recovery
runs did not reproduce overheating.

## Claim boundary and required qualification

R6 has passed this periodic-vortex diagnostic and the existing `2e-5` FP32
numerical bound at the measured points. It has **not** yet passed the complete
GradFlow admission gate. Before any backend or publication claim, freeze and
run:

- exact matched-input comparison against the preserved Shu/qualified FP32
  path at the existing `N=6` one- and ten-step points and `N=32` point;
- non-vortex admissible perturbations and conservation checks;
- smooth convergence and critical-point tests;
- Sod and Shu--Osher shock tests;
- randomized same-session R6 versus cell-recompute timing blocks;
- start-to-finish latency and peak-workspace measurements;
- profiler evidence for memory traffic, occupancy, and achieved issue rate;
- independent replication and, later, arbitrary-order behavior.

No PyTorch, DVEB, public GradFlow, or arbitrary-order implementation was
changed by this experiment.

## Repository verification

- Ruff passed for both new Python comparison tools.
- Every G1, G2, and G3 evidence checksum passed.
- Native U0 and R1--R6 compiled for `sm_120`; every counted executable ran and
  returned finite output.
- The repository suite reported 275 passed and 72 skipped. Eight historical
  frozen-record verifier tests failed in the restarted environment: two
  byte-exact NumPy regenerations differed, and six downstream verifiers lost
  the repository root in spawned subprocess imports under the uninstalled
  `src/` environment. No failed test exercises files changed by this study.

The historical records were not rewritten to hide those environment-sensitive
failures.
