# FD/FV Phase-3 scalar qualification result

Post-study note: the formerly missing CUDA agreement and compiler gates passed
prospectively on Forge's RTX 5070 Ti; see
`DEFERRED_CUDA_GATES_RESULTS.md`. This document and its failed frozen decision
remain unchanged in evidentiary meaning.

Status: **completed and did not pass the frozen qualification**.

Run date: 2026-08-27 UTC.

Candidate source commit:
`1d920ea97ed7abec9e4e451b377343cf72316f4c`.

The immutable record is
`experiments/fd_fv_qualification/results/phase_3_20260827/qualification.json`.
Its SHA-256 is
`6d0c9585f31a32a2faa24ecf33c9562f656a7fb4873fcc597c07c00ae3fcd4df`.
No performance measurements were collected.

## What was implemented

`src/gradflow/fv_weno5.py` implements the frozen
`fv_dimensional_js5_global_lf_periodic_v1` formulation in ordinary PyTorch.
The persistent values are physical cell averages. The implementation
reconstructs left and right point states at faces, evaluates the explicit
global Lax--Friedrichs/Rusanov numerical flux, and differences those face
fluxes conservatively. It does not call the finite-difference RHS or split
physical flux samples as though cell averages were nodal values.

The implementation deliberately remains scalar, periodic,
dimension-by-dimension WENO-JS5. It is not an Euler solver, a genuinely
multidimensional FV method, a performance result, or a general FV framework.

Before implementation, the protocol received one named amendment. Python
`dx` and `alpha` values are checked for finite positivity. Tensor scalars are
checked for scalar shape, dtype, and device, while value positivity is a
caller precondition. Inspecting a CUDA scalar inside every RHS would force a
synchronization; the future prepared-problem layer is the appropriate place
for a one-time value check. The amendment changed no mathematical formula,
oracle, or numerical tolerance.

## Gate result

Nine of eleven recorded gate areas passed. Exactly two failed:
`smooth_spatial` and `transfer_evidence`. Under the preregistered all-gates
rule, the Phase-3 result is therefore failed.

### Evidence that passed

- The independent exact-Fraction face-state, numerical-flux, and RHS oracles
  passed in both advection directions. Maximum float64 RHS differences were
  `1.42e-14` and `2.84e-14`.
- Constant preservation and the refusal contract passed.
- The smooth complete solve converged monotonically. Consecutive L2 rates were
  `4.0625`, `4.5863`, and `4.5940`; all conservation checks passed.
- The translated discontinuity L1 errors decreased from `0.0239080` through
  `0.0145629` to `0.00866280`. Solutions remained finite, extrema stayed
  within the frozen bounds, and mass checks passed.
- Float64 RHS gradcheck passed. The three-step SSP-RK3 directional derivative
  differed from the centered check by `1.84e-11` absolute and `1.17e-9`
  relative.
- Eager CPU float32 and float64 execution passed.
- Both the fixed CPU RHS and SSP-RK3 step compiled as one graph with zero graph
  breaks and exact eager agreement. Compilation latency was not timed.
- CUDA and MPS were unavailable and are recorded as untested, not simulated.

### Frozen smooth-spatial failure

For positive advection, errors decreased monotonically and the maximum
consecutive rate was `5.14029`, passing the required `>=4.7` threshold. For
negative advection, errors also decreased monotonically, but the maximum rate
was `4.692164`, missing the frozen threshold by `0.007836` (about 0.17% of the
threshold).

The exact oracle parity and the other convergence results do not permit this
miss to be described as a wrong reconstruction. Conversely, their success
does not permit the preregistered threshold to be lowered. The mixed Fourier
field contains critical-point behavior relevant to the known asymptotic
limitations of classical WENO-JS. A fresh protocol must separate a
noncritical design-order experiment from an explicit critical-point
characterization before this issue is resolved.

### Frozen transfer-evidence failure

Static inspection found none of `.cpu()`, `.cuda()`, `.to()`, `.item()`, or
`.numpy()` in the canonical FV source. The CPU profiler nevertheless reported
`aten::to`, which the frozen protocol named as forbidden, so this gate failed.
The same probe found no `aten::_to_copy` or `aten::copy_` event. Thus this
record does not show an actual host/device transfer, but the original
event-name-only rule cannot distinguish a no-copy dtype/device-preserving
dispatch from real data movement.

This is a measurement-contract limitation, not permission to retroactively
change the result. A new protocol should require direct copy/data-movement
evidence, including CUDA evidence when a device is available, while retaining
the static source prohibition and preserving this failed first run.

## Decision and next boundary

The candidate is mathematically credible and useful research infrastructure,
but it is not yet the qualified FV comparator required by the Phase-1
constitution. No matched-component or best-practical FD/FV performance timing
may begin from this result.

The next justified boundary is a separately frozen Phase-3 resolution study:

1. preserve this result and its thresholds unchanged;
2. freeze distinct noncritical and critical-point spatial tests;
3. replace profiler-name inference with an auditable data-movement criterion;
4. rerun the bounded qualification on available CPU/CUDA environments; and
5. either qualify this seed under the new prospective contract or record the
   unresolved numerical/device limitation.

Only a qualified scalar seed permits the Phase-4 FD/FV timing matrix.
