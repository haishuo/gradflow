# FD/FV Phase-4 scalar CPU bakeoff result

Status: **CPU matrix complete; CUDA unavailable to this recorded process**.

Post-study note: the preregistered Phase-4R investigation is complete. It did
not strongly replicate the isolated `N=27^3` result and associated the fast FV
mode with unstable CPU multithread execution rather than a unique generated-
kernel transition. This document preserves the original Phase-4B observation;
its updated interpretation is in `FD_FV_PHASE_4_REPLICATION_RESULTS.md`.

Measurement date: 2026-08-27 UTC.

Measurement source commit:
`5736a8d4f1673a5cb7a42914d0942e822c90ec4b`.

The aggregate record is
`experiments/fd_fv_bakeoff/results/phase_4b_20260827/benchmark.json`,
SHA-256
`056f61997b13faddd36f0d80dd541da8c0cb5da6ffda8a0951b48b38af25737a`.
All 24 warm cells and all six cold pilots passed their numerical eligibility
checks. The manifest preserves 1,680 raw timing samples.

## Bottom line

For this matched, smooth, periodic, float64 linear-advection problem on the
six-core Ryzen 5 7600X:

- 1-D FD and FV warm complete solves were unresolved within 5% at every size;
- 2-D was also close: FD exceeded the 5% decision boundary at only two of four
  sizes, by `1.051x` and `1.054x`;
- 3-D remained unresolved through `N=18`; at `N=27` the compiled FV solve was
  `2.69x` faster and its compiled step was `5.02x` faster than compiled FD; and
- compilation dominated one-shot latency for both methods. Warm compilation
  improved complete solves by roughly `6.4x--16.5x`, but first compiled solves
  cost approximately 8--20 seconds.

There is no general FD or FV winner in this matrix. The isolated large-3-D
result is a strong compiler-behavior observation, not yet a formulation-wide
claim. It needs same-machine replication, larger grids, and compiler-level
causal inspection before being elevated.

## Warm complete-solve results

Compiled was the fastest measured mode for both formulations in every cell.
Times below are medians in milliseconds; `FV/FD > 1` favors FD.

| Dim. | N | FD L2 error | FV L2 error | FD ms | FV ms | FV/FD | Decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 24 | 1.621e-6 | 1.634e-6 | 0.460 | 0.468 | 1.017 | unresolved |
| 1 | 36 | 2.026e-7 | 2.000e-7 | 0.920 | 0.964 | 1.048 | unresolved |
| 1 | 54 | 2.537e-8 | 2.500e-8 | 1.843 | 1.879 | 1.020 | unresolved |
| 1 | 81 | 3.189e-9 | 3.188e-9 | 3.696 | 3.754 | 1.016 | unresolved |
| 2 | 12 | 1.956e-5 | 2.003e-5 | 0.361 | 0.380 | 1.051 | FD faster |
| 2 | 18 | 2.495e-6 | 2.606e-6 | 0.705 | 0.733 | 1.039 | unresolved |
| 2 | 27 | 3.153e-7 | 3.146e-7 | 1.554 | 1.638 | 1.054 | FD faster |
| 2 | 40 | 4.155e-8 | 4.173e-8 | 3.891 | 3.979 | 1.023 | unresolved |
| 3 | 8 | 8.088e-5 | 4.402e-5 | 0.314 | 0.321 | 1.023 | unresolved |
| 3 | 12 | 1.082e-5 | 1.078e-5 | 0.945 | 0.940 | 0.994 | unresolved |
| 3 | 18 | 1.356e-6 | 1.425e-6 | 7.536 | 7.322 | 0.972 | unresolved |
| 3 | 27 | 1.719e-7 | 1.715e-7 | 45.713 | 16.970 | 0.371 | FV faster |

Equal `N` is a diagnostic, not the primary fairness claim. Here the errors at
all but the coarsest 3-D cell are close enough to make the warm comparisons
informative. At `N=8` in 3-D, FV was materially more accurate, so equal-grid
timing alone would not be an equal-accuracy comparison.

The 3-D `N=27` compiled-step distributions support the recorded difference:
FD's median was `2.887 ms`; FV's median was `0.575 ms`. FV's samples were
multimodal (`0.453--2.723 ms`), while FD's were narrower
(`2.676--3.204 ms`). This makes the median difference observable but also
signals scheduler/kernel behavior requiring replication. No causal conclusion
about the WENO mathematics is made from it.

## Cold and compilation boundaries

The single frozen process-launch-to-answer pilots were:

| Dim. | Largest N | FD seconds | FV seconds | Faster pilot |
|---:|---:|---:|---:|---|
| 1 | 81 | 11.398 | 9.989 | FV |
| 2 | 40 | 16.023 | 18.423 | FD |
| 3 | 27 | 22.381 | 20.367 | FV |

Each is one pilot, not a stable latency estimate. Python/PyTorch import,
initialization, Inductor compilation, the complete solve, host-visible error
calculation, serialization, and process exit are included. The result directly
shows why warm compiled throughput must not be substituted for “press run and
get an answer” latency.

No packaged AOT endpoint existed and none was simulated. The record marks AOT
`not_implemented`.

## Memory

Persistent state is identical at equal grid size: one float64 scalar per node
for FD or per cell for FV. Isolated-worker peak process RSS was approximately
`789--841 MiB`; paired FD/FV differences were generally below 1%. These RSS
values include Python, PyTorch, compiler state, and temporary tensors, so they
do not isolate the numerical live set. Compiler caches were also preserved in
the raw records. This matrix provides no evidence of a material CPU memory
winner.

## Interpretation and limitations

The near-tie in 1-D is consistent with the shared reconstruction algebra for
linear advection. The mostly small 2-D differences likewise do not support a
broad structural advantage. The large compiled 3-D separation appears only at
the largest admitted grid and may reflect different Inductor simplification,
fusion, scheduling, or threading behavior in the split-flux FD and Rusanov FV
expressions. Those are hypotheses, not measured causes.

The bounded best-practical lane only selected eager or compiled ordinary
PyTorch. The FV wrapper is newer; both formulations share the generated
reconstruction core; neither received formulation-specific low-level tuning;
and no matched external native ceiling participated. This is not “best
possible FD versus best possible FV.”

CUDA and MPS were unavailable. Nothing in this record establishes GPU
performance, the RTX crossover, or the effect of consumer-GPU FP64 limits.

## Superseded next boundary

Before extending to nonlinear scalar flow or Euler:

1. replicate the CPU matrix, especially 3-D `N=27`, in fresh runs;
2. add preregistered larger 3-D operator points to locate the compiler cliff;
3. record generated-kernel counts, fusion, vectorization, and threading
   evidence without optimizing either formulation mid-study;
4. run the unchanged device-admission and timing protocol when CUDA is visible;
5. make an explicit value-of-information decision before renting FP64 data-
   center hardware.

Items 1--3 were completed by Phase 4R. CUDA was later identified as hidden by
the default sandbox and item 4 was completed on Forge's RTX 5070 Ti as a linked
supplement. The Phase-4R result does not support a persistent FV crossover;
see `FD_FV_PHASE_4_REPLICATION_RESULTS.md` and
`FD_FV_PHASE_4_CUDA_RESULTS.md`. No universal FD/FV or publication claim is
supported yet.
