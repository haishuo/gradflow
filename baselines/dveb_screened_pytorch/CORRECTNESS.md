# Comparator and Ceiling Correctness Record (2026-08-25)

Correctness state of every screened implementation before comparative
timing. Machine-readable data: the check-runner `report.json` produced by
`comparator/run_checks.py`; committed calibration data under
`tools/ceiling/calibration/`.

## Result under the committed tolerances

All five ordinary-PyTorch variants (eager, compile, compile-ro, compile-ta,
conv) and all 11 ceiling variants pass **every** predeclared check at
**every** declared grid point, under the committed scaled RHS tolerance
`T_rhs(n) = max(1e-11, 8·ε_machine·n)`:

- committed n = 400 oracle fixture: PASS (all implementations, ≤ 1e-12);
- analytic roundoff-aware bounds: PASS at all 12 points, every variant;
- single-RHS cross-implementation agreement vs DVEB: PASS at all 12
  points, every variant (see the table below for actual values);
- 50-step final state ≤ 1e-10: PASS at all 12 points, every torch variant
  (worst 4.1e-15) and at the shared subset for all 11 ceiling variants;
- full-period subset {P1, D1, D2, D5}: agreement ≤ 1e-10 and Ladder-A L2
  bounds: PASS, every torch variant.

## The original flat-tolerance discrepancies (preserved, not erased)

Under the original flat `≤ 1e-11`, the single-RHS check failed **only** at
P5 (n = 25600) and P6 (n = 102400), for every implementation pair:

| Pair | P5 err | P6 err |
|---|---|---|
| eager vs DVEB | 8.9e-12 (passed) | 1.137e-11 |
| compile / compile-ro / compile-ta vs DVEB | 1.421e-11 | 5.684e-11 |
| conv vs DVEB | 1.421e-11 | 6.821e-11 |
| every ceiling variant vs DVEB | 1.421e-11 | 5.684e-11 |
| ceiling vs compiled torch | 5.684e-12 | 2.274e-11 |

Diagnosis: the contraction-off DVEB gate build is the outlier in every
pair, and even the two contraction-matched implementations differ by
2.27e-11 at P6 — cross-implementation f64 roundoff scales ≈ ε_machine·n
through the 1/dx division, making a flat 1e-11 unattainable at n = 102400
between *any* two distinct implementations. This motivated the committed
tolerance correction (see the trunk document's correction record); the
correction was made after correctness inspection and **before any
comparative performance timing existed**.

Scaled tolerances at the affected points: T_rhs(25600) ≈ 4.55e-11,
T_rhs(102400) ≈ 1.82e-10. Every value above falls within them.

## Ergonomic incidents recorded during comparator bring-up

1. CUDA-Graphs modes (reduce-overhead, max-autotune) crash on the naive
   output→input step loop; the documented per-iteration protocol
   (`cudagraph_mark_step_begin()` + clone-per-step, one extra full-field
   copy per step) is required and is part of their ordinary execution cost.
2. The conv variant's first revision constructed its constant stencil
   kernels inside the RHS, silently performing two HtoD PCIe copies per
   evaluation; fixed by hoisting (zero transfers verified, mathematics
   unchanged).

## Structural inspection summary (not timing)

At D2: eager = 128 kernel launches per RHS (396 per RK step);
TorchInductor (fullgraph, dynamic) = **2 kernels per RHS, 6 per step, one
graph, zero breaks, no recompiles across the 12 shapes**; DVEB generated
path = 2 launches per RHS (halo + fused RHS), 9 per step. All three
implementation families **execute** the mathematically-zero negative LF
split; none eliminates it (two WENO-correction normalization chains present
in the generated Triton; `em` is a runtime argument in DVEB and the
ceiling). Generated-code artifacts preserved under `comparator/artifacts/`.
