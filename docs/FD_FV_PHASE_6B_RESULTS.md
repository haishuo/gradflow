# FD/FV Euler Phase-6B matched correctness result

Status: **PASS**.

Record date: 2026-08-28 UTC.

Qualified source commit: `c23771658c7d4ad7d2d59bb27194ff9ffeea3e4a`.

## Decision

The prospectively registered
`fv_dimensional_characteristic_js5_global_matrix_lf_euler1d_v1`
implementation satisfies the complete frozen Phase-6B correctness boundary.
The existing
`fd_classical_characteristic_js5_global_lf_euler1d_v1` implementation also
passes every common current-environment gate. Phase 6C may now freeze a
performance protocol; no timing was collected in Phase 6B.

This is a matched one-dimensional ideal-gas Euler WENO-JS5 result. It is not a
claim that FD and FV are mathematically identical, that either method is
universally preferable, or that the result extends to multiple dimensions,
Navier--Stokes, other fluxes, other WENO families, or other orders.

## Smooth and invariant gates

Both methods reproduced their independently frozen point-value or physical
cell-average projections exactly. Constant-state RHS values were below
`2e-12`, and all four periodic/transmissive conservation residual ratios were
exactly zero in this run.

Spatial entropy-wave L2 errors and observed rates were:

| Method | N=24 | N=36 | N=54 | N=81 | Consecutive rates |
| --- | ---: | ---: | ---: | ---: | --- |
| FD | 4.59245e-5 | 5.62335e-6 | 6.76397e-7 | 8.48810e-8 | 5.179, 5.223, 5.119 |
| FV | 4.58361e-5 | 5.65133e-6 | 6.87539e-7 | 8.75235e-8 | 5.162, 5.195, 5.084 |

Complete entropy-wave solves to `T=0.1` used SSP-RK3, CFL `0.1`, and exact
final-step shortening. Their L2 errors were:

| Method | N=24 | N=36 | N=54 | N=81 | Consecutive rates |
| --- | ---: | ---: | ---: | ---: | --- |
| FD | 4.20800e-6 | 5.26686e-7 | 6.60053e-8 | 8.32294e-9 | 5.125, 5.122, 5.107 |
| FV | 4.21100e-6 | 5.30551e-7 | 6.71091e-8 | 8.55454e-9 | 5.109, 5.099, 5.080 |

These rates show that both registered spatial formulations recover the
expected fifth-order smooth behavior on this problem. They do not turn the
third-order time integrator into a fifth-order method generally; at the
frozen small CFL and resolutions, spatial error dominates this particular
study.

All four fixed-step directional-derivative probes were finite and satisfied
the declared combined tolerance. Relative discrepancies ranged from
`7.60e-9` to `4.18e-7`.

## Shock gates

The new FV formulation completed Sod and Shu--Osher at `N=200,400,800`
without clipping, retry, fallback, or nonphysical SSP-RK stage.

Sod primitive L1 errors decreased monotonically:

| N | Density | Velocity | Pressure |
| ---: | ---: | ---: | ---: |
| 200 | 2.84367e-3 | 5.08300e-3 | 2.20183e-3 |
| 400 | 1.41888e-3 | 2.31148e-3 | 1.06270e-3 |
| 800 | 6.91164e-4 | 1.01850e-3 | 5.03378e-4 |

The finest/coarsest ratios were `0.2431`, `0.2004`, and `0.2286`; the contact
and right-shock errors were `0.3924` and `0.6551` cells. All inherited
thresholds passed.

Shu--Osher density L1 error fell from `7.13731e-2` at `N=200` to
`1.13323e-2` at `N=800`, a finest/coarsest ratio of `0.1588`. At `N=800`,
density correlation was `0.998619` and the density total-variation ratio was
`0.946369`. The inherited primitive-error and structure thresholds passed.

The earlier FD shock PASS remains inherited under its point-value oracle. Its
errors are not directly compared with the FV average errors here because the
two methods have different persistent-state semantics.

## Compiler, device, and movement gates

All eight fixed-shape CPU/CUDA and periodic/transmissive full-graph cases
produced one graph and zero graph breaks. CPU compiled/eager differences were
zero. CUDA compiled/eager differences were at most `1.151e-13`, and float64
CPU/CUDA differences were at most `8.216e-14`.

Profiler probes for both methods on CPU and CUDA found no host/device movement
event inside the numerical path. Outputs retained input shape, dtype, and
device. Static inspection found no scalar extraction, NumPy conversion,
custom operator, handwritten CUDA, or handwritten Triton in the called FV
reconstruction/RHS path.

The admitted accelerator was Forge's NVIDIA GeForce RTX 5070 Ti, compute
capability 12.0, under PyTorch `2.13.0+cu130` and CUDA runtime 13.0. MPS was not
tested.

## Reproducibility

The independent verifier recomputes projection identities, numerical errors,
rates, conservation bounds, shock metrics, threshold decisions, hashes, and
the top-level decision from preserved raw arrays without rerunning production
work.

Artifact hashes:

- `qualification.json`:
  `202269228ba7c2281b3b9edd236a1d7ae5123e50e5e69359398ca876977fafc7`
- `raw_arrays.npz`:
  `b6039acadfaba47ce2e0829ba6a229bcb2aaec8eddd1a93bc2d8f95a0ef7f8ea`
- `SHA256SUMS`:
  `f8bdea3459967c126137b4e5cbabd76bb2a2165073dc165721b052f7d9944ac2`

Run:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6b.py
```

The final configured repository suite passed on Forge with CUDA and the
declared read-only DVEB fixtures: `340 passed`, with 14 upstream PyTorch
deprecation warnings.

## Scientific interpretation

Phase 6B removes correctness, differentiability, compilation, device
agreement, and shock admissibility as immediate blockers to a matched Euler
FD/FV timing study. It does not say which method is faster or more accurate
per unit cost. The next scientifically valid boundary is a separately frozen
Phase-6C protocol comparing achieved error, complete-solve latency, memory,
and failure behavior while preserving distinct FD point and FV cell-average
semantics.

No Phase 6C experiment, multidimensional extension, DVEB development, or
performance campaign began in this phase.
