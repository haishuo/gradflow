# FD/FV Phase-4A multidimensional admission result

Status: **passed on CPU; CUDA unavailable to this recorded process**.

Post-study correction: Forge's RTX 5070 Ti was hidden by the default Codex
device sandbox, not absent from the host. A later fresh CUDA admission passed;
see `FD_FV_PHASE_4_CUDA_RESULTS.md`. This document and its immutable record
continue to describe the original process observation.

Run date: 2026-08-27 UTC.

Source commit: `7ff5708449d2b5e833a33cbf017a7ce98f5e272d`.

The immutable record is
`experiments/fd_fv_bakeoff/results/phase_4a_20260827/qualification.json`,
SHA-256
`6fd933cb44b1aa9350dd3c52cd7d446e182dce7411f93ba4cd8e6b3a0abe5362`.
No timing or performance measurement was collected.

Both registered formulations passed the frozen 1-D, 2-D, and 3-D
constant-coefficient linear-advection gate. L1/L2 errors decreased on every
refinement, every mass check passed, and every representative compiled
SSP-RK3 step produced one graph, zero graph breaks, and exact eager agreement.

Observed L2-rate ranges were:

| Formulation | 1-D | 2-D | 3-D |
|---|---:|---:|---:|
| Classical FD | 5.115–5.129 | 5.079–5.156 | 4.961–5.122 |
| Dimension-by-dimension FV | 5.079–5.180 | 5.030–5.215 | 3.470–5.222 |

The coarse `N=8` to `N=12` FV 3-D transition was pre-asymptotic at `3.470`;
the following rates were `4.991` and `5.222`. This satisfies the frozen rule
of monotone error plus at least one rate `>=4.0`, while preserving the coarse-
grid behavior rather than hiding it.

Phase 4A admits the exact CPU cells to Phase 4B measurement. It does not imply
equal accuracy at equal `N`, and it does not qualify CUDA: CUDA and MPS were
unavailable in this environment. Every later GPU timing point still requires
device admission before measurement.
