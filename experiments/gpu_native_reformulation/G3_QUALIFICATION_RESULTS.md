# G3 R6Q comprehensive qualification results

Date: 2026-08-29 (UTC)

Hardware: NVIDIA GeForce RTX 5070 Ti, CUDA 13, `sm_120`

## Decision

G3 is complete. R6Q provides strong corroborating evidence that the face-once
CUDA schedule preserves the Shu characteristic FD-JS-WENO-5 forward
calculation, but it **does not pass the frozen qualification protocol as
written and is not admitted as a GradFlow backend**.

That distinction is intentional. The candidate passed the most consequential
state-level numerical checks by wide margins, but a frozen criterion is not
rewritten after results exist. R6Q also has no reverse-mode/autograd ABI, which
remains an independent backend-admission blocker.

## What passed

- The R6Q interface extension reproduced the frozen R6 `N=32`, one-step output
  byte-for-byte. Both files have SHA-256
  `f729952e4195c6c62c6eef5109e7c82ec25b29dc8cd7901af69758f6759e132a`.
- Across both vortex families and `(N, steps)={(6,1),(6,10),(32,1)}`, the worst
  FP32 full-state error was `7.153e-7`, versus the unchanged `2e-5` bound.
- All periodic discontinuity states remained finite with positive density and
  pressure. Their worst FP32 full-state error was `5.722e-6`, also below
  `2e-5`, after ten steps of the dual-interface Shu--Osher-type specimen.
- Smooth entropy-wave analytic RMS error decreased at every grid size:
  `8.187e-3`, `1.627e-3`, `9.527e-4`, and `4.037e-4` on
  `N={12,18,27,40}`. Observed rates were `3.985`, `1.321`, and `2.185`; the
  preregistered requirement of at least one rate at or above 3 passed.
- The critical-point errors also decreased at every size. The observed rates
  `1.947`, `3.000`, and `1.885` record the expected nonuniform Jiang--Shu
  critical-point behavior; no fifth-order critical-point claim is made.
- The fixed `N=6` finite-difference directional response differed from the
  FP32 oracle by `2.358e-5` relative RMS, far below the `2e-2` bound.

## Why the frozen overall gate failed

### RHS relative-RMS criterion

Every RHS maximum error passed its absolute `5e-5` bound; the worst was
`5.722e-6`. The independently ordered FP32 expressions nevertheless exceeded
the additional `2e-5` *relative RMS* bound at six of eight sizes. The worst
relative RMS discrepancies were `5.212e-5` for the smooth family and
`5.586e-5` for the critical family.

This is a small floating-point expression-order discrepancy, not evidence of
a different resolved evolution: all ten full-step comparisons passed, and
the native RHS was generally closer to the promoted-FP64 oracle than to the
ordinary-PyTorch FP32 oracle. It still counts as a protocol failure.

### Zero conservation budget

The frozen conservation expression was

```text
128 * steps * eps_float32 * sum(abs(initial_component)).
```

It therefore assigns exactly zero tolerance to an initially zero conserved
component. The perturbed vortex has zero z-momentum and accumulated z-momentum
drifts between `3.13e-15` and `3.67e-12`. The Sod specimen has zero initial
x-momentum and accumulated `2.98e-8` after one step and `-3.69e-6` after ten.
These roundoff-scale values fail a literal zero bound. All components with a
nonzero budget passed. The protocol was intentionally left unchanged; a later
constitution may preregister a nonzero absolute roundoff floor.

## Harness correction record

The first qualification execution accidentally used CFL 0.6 in the Python
oracle instead of the frozen Shu/R6 CFL 0.1. Its JSON and arrays are retained
under names containing `invalid_harness_cfl_0p6`; they are not candidate
evidence. The harness was corrected to the pre-existing policy without
changing the candidate, test matrix, or any tolerance, and the complete matrix
was rerun.

## Scientific interpretation

The face-once execution schedule survives a much broader forward test than the
vortex specimen used during recovery. Arbitrary admissible smooth states,
periodic shocks, analytic spatial tests, and a forward sensitivity probe did
not reveal a material mathematical defect. G3 therefore supports advancing
the *research hypothesis* that shared-face ownership is a valid GPU-native
reformulation of this fixed WENO-5 calculation.

It does not support shipping R6Q. The strict G3 protocol did not pass,
differentiation is absent, boundary support remains periodic-only, and the
randomized same-session performance comparison belongs to G4. No G4 timing or
arbitrary-order claim was begun here.

## Reproduction

The evidence directory contains the frozen protocol, exact source and build
recipe, compiler log, executable, full JSON, all comparison arrays,
environment metadata, invalid-harness record, and SHA-256 manifest. Verify it
with:

```bash
python experiments/gpu_native_reformulation/verify_g3_qualification.py \
  experiments/gpu_native_reformulation/evidence/g3_qualification_20260829
```
