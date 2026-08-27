# FD/FV Phase-3R resolution result

Status: **passed on the available CPU environment**.

Run date: 2026-08-27 UTC.

Diagnostic source commit:
`bd07370b9f36954567687e8c0cbc5f1a27ae24d3`.

The canonical FV numerical source remained byte-identical to commit
`1d920ea97ed7abec9e4e451b377343cf72316f4c`, with SHA-256
`58b6c55b1fe1e84a5f0eaeb30f31acabf25d0cda713b02c9090085c04c3dbed0`.

The immutable record is
`experiments/fd_fv_qualification/results/phase_3r_20260827/resolution.json`.
Its SHA-256 is
`b0d42fb34f2999d43a4b9591e65ca7edab59ea15b51fa05deb083e5781d0b01f`.
No performance measurements were collected.

## Decision

All five prospectively frozen Phase-3R gates passed:

- original Phase-2 and failed Phase-3 records verified unchanged;
- the canonical numerical-source identity matched;
- the noncritical fifth-order gate passed;
- the critical-point characterization gate passed; and
- CPU data-movement evidence passed.

Together, Phase 2, the original Phase 3, and Phase 3R qualify this exact scalar
periodic dimension-by-dimension FV-WENO-JS5 seed on CPU. The original Phase-3
result remains failed under its original rules; Phase 3R neither overwrites it
nor relabels either old gate.

CUDA and MPS were unavailable. This result does not qualify GPU agreement or
GPU residency. Any future GPU timing point must first pass the frozen
per-device correctness and movement checks on the machine used.

## Noncritical design order

The new local test used exact physical cell averages of `exp(x)`, whose
derivative never vanishes, and excluded every face/cell whose WENO stencil
crossed the artificial periodic wrap. Both face biases and both advection
directions decreased monotonically.

The final two observed rates were:

| Sequence | Penultimate rate | Final rate |
|---|---:|---:|
| Left face | 4.98964 | 4.99341 |
| Right face | 4.98633 | 4.99116 |
| Positive-speed RHS | 4.98734 | 4.99182 |
| Negative-speed RHS | 4.98407 | 4.98952 |

Every value exceeded the prospectively frozen `4.7` threshold. This is direct
evidence that the shared exact-generated reconstruction and FV divergence
retain fifth-order behavior away from critical points in both orientations.

## Critical-point evidence

The original mixed-Fourier errors and rates reproduced bit-for-bit, including
the negative-direction maximum rate of `4.692164`. That historical miss is
therefore preserved, not dismissed as noise.

At the deliberately aligned symmetric maximum of `sin(2*pi*x)`, both left and
right reconstructions produced identical monotonically decreasing errors and
rates `6.01785`, `6.00481`, and `6.00178`. This is an observed symmetry-driven
superconvergence result for that particular face—not a claim that WENO-JS is
sixth order or immune to critical-point degradation.

Consequently, Phase 3R rules out a general fifth-order implementation defect,
but it does not establish that a simple critical point alone caused the mixed-
Fourier `4.692164` rate. Grid alignment, asymmetric/higher-order critical
structure, epsilon scaling, and pre-asymptotic behavior remain candidates for
later numerical-limit characterization. They are not blockers for the bounded
noncritical design-order qualification, provided the original behavior remains
disclosed.

## Data-movement evidence

Static analysis of both the FV module and its shared WENO-JS dependency found:

- no `.cpu()`, `.cuda()`, `.item()`, or `.numpy()` call;
- no device-selecting `.to()` call; and
- one declared dtype-only `.to(dtype=...)` site in the generic precision
  helper.

On the native float64 CPU RHS, the profiler recorded 18 `aten::to` dispatches.
They reported zero CPU and device memory allocation. There was no
`aten::_to_copy`, `aten::copy_`, memcpy, H2D, or D2H event, and the output
preserved the input's CPU device and float64 dtype.

This supports the narrow conclusion that the original `aten::to` labels were
not data movement in the tested native CPU path. It does not infer the same
result for CUDA; that experiment remains untested until CUDA is visible.

## Research boundary

Phase 3R permits preparation of the Phase-4 scalar FD/FV protocol. It does not
itself authorize or contain timing. Phase 4 must preregister:

- matched-component and best-practical lanes separately;
- accuracy-to-time and accuracy-to-memory as primary outcomes;
- cold, prepared, warm, resident, and operator endpoints;
- per-device correctness admission before timing;
- CPU threading and GPU synchronization/residency rules; and
- failure and unavailable-hardware reporting.

No Euler, arbitrary-order FV, genuinely multidimensional FV, or general
FD-versus-FV superiority claim follows from Phase 3R.
