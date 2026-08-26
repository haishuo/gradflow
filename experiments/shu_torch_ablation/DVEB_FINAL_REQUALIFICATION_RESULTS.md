# Final DVEB WENO requalification results

## Held-out selector gate

Fresh-process medians from 30 randomized blocks; calibration sizes are disjoint.

| N | Steps | Selected | Best forced | Auto ms | Best ms | Regret | Loss ms |
|---:|---:|:---|:---|---:|---:|---:|---:|
| 8 | 1 | cpu_simd[6] | cpu_simd[12] | 2.437 | 2.230 | 1.0929 | 0.207 |
| 16 | 1 | cpu_simd[12] | cpu_simd[12] | 5.035 | 5.182 | 0.9716 | -0.147 |
| 32 | 1 | cpu_simd[12] | cpu_simd[12] | 24.898 | 24.885 | 1.0005 | 0.014 |
| 48 | 1 | cpu_simd[12] | cpu_simd[12] | 76.744 | 76.727 | 1.0002 | 0.016 |
| 64 | 1 | cuda | cpu_simd[12] | 180.694 | 177.046 | 1.0206 | 3.648 |
| 8 | 10 | cpu_simd[12] | cpu_simd[12] | 6.499 | 6.385 | 1.0179 | 0.114 |
| 16 | 10 | cpu_simd[12] | cpu_simd[12] | 31.472 | 31.438 | 1.0011 | 0.033 |
| 32 | 10 | cuda | cuda | 173.994 | 173.689 | 1.0018 | 0.305 |
| 48 | 10 | cuda | cuda | 182.009 | 182.643 | 0.9965 | -0.634 |
| 64 | 10 | cuda | cuda | 255.162 | 208.209 | 1.2255 | 46.953 |

Decision: **PASS** within the
declared WENO-specific N=8..64 envelope.
Median regret: `1.0014`; maximum: `1.2255`. All 300 automatic runs made a
stable decision and all 1,200 held-out runs completed successfully.
Nine of ten points were within 15% of the best forced target. At N=64 /
ten steps the selector chose the correct CUDA family, but automatic and
forced-CUDA fresh-process medians differed by 22.6%, exposing startup
variability rather than a target-choice miss.

## Large-grid confirmation

| N | Steps | Selected | Auto ms | Forced CUDA ms | Ceiling ms | Generated/ceiling |
|---:|---:|:---|---:|---:|---:|---:|
| 96 | 1 |  | refused (outside model range) | 203.442 | 203.695 | 0.9988 |
| 128 | 1 |  | refused (outside model range) | 245.577 | 245.414 | 1.0007 |
| 96 | 10 |  | refused (outside model range) | 246.094 | 243.351 | 1.0113 |
| 128 | 10 |  | refused (outside model range) | 330.571 | 325.204 | 1.0165 |

Automatic placement safely refused every large-grid point because N=96 and
N=128 lie outside the calibration model's bounded N=7..72 range. Forced CUDA
therefore confirms generated-backend performance but does not qualify automatic
dispatch outside the held-out N=8..64 envelope. Across these four points,
generated CUDA was within 1.65% of the independent ceiling at the complete
fresh-process endpoint.

## Decision boundary

The final committed DVEB artifact preserves the prior correctness and
ceiling-class CUDA result. DVEB therefore has a validated role as an optional
native WENO backend. WENO-specific automatic placement is qualified only inside
the declared machine-specific envelope; outside it GradFlow must fall back or
require an explicit target until a separately frozen calibration is qualified.

This is a WENO-specific, machine-specific qualification. DVEB's generic
automatic selector remains NO-GO at commit `2f1f3ab`.
