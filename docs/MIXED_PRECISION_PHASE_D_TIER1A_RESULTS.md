# Phase-D Tier-1a scalar mixed-precision results

Execution date: 2026-08-27 UTC.

Status: **the complete frozen 384-point matrix executed and verified**.

## Result

The first exhaustive search found one candidate internal demotion under the
full frozen contract: compute Jiang--Shu smoothness indicators in binary32 and
retain the binary64 state, flux split, candidate polynomials, nonlinear-weight
formation and normalization, face combination, and divergence.

That assignment was classified:

- `engineering` at WENO-5, with worst normalized maximum error
  `1.9876063673387654e-4`; and
- `tight` at orders 7, 9, 11, 13, and 15, with worst normalized maximum error
  between `2.714073328182201e-7` and `3.211653438348938e-6`.

No assignment demoting two or more of the six Tier-1a blocks passed even the
`coarse` full-case contract. The all-binary64 assignment was the only `tight`
result at WENO-5. Across all orders, the classification count was:

| Class | Count |
|---|---:|
| `tight` | 11 |
| `engineering` | 1 |
| `coarse` | 0 |
| `failed` | 372 |

The 11 tight records include the six all-binary64 controls.

## Why most assignments failed

The near-constant family was decisive. For a `1e-7` sinusoidal perturbation
over a unit background, independently demoting flux splitting, candidate
polynomials, nonlinear weights, face combination, or divergence produced a
worst normalized maximum discrepancy of roughly 43--97 times the derivative
signal, depending on block and order. The all-binary32-internal assignment was
roughly 96--145 times the signal.

This is expected binary32 spacing, not a conservation or NaN failure. Every
reported assignment remained finite, preserved the exactly constant state,
and passed the deliberately loose conservative-sum safety bound. Without the
three near-constant cases, all 384 assignments would have met the
`engineering` parity thresholds and 88 would have met `tight`. Ordinary-scale
tests alone would therefore have produced a dangerously permissive answer.

The result also confirms that high-order convergence and mixed precision
cannot be assessed independently. Whole-weight binary32 calculation reduced
the last smooth refinement rate from approximately 8.59 to -1.01 at order 9,
from 9.84 to -1.03 at order 11, and led to roundoff-dominated behavior at
orders 13 and 15. Demoting only the indicators retained much better behavior,
although the highest-order finest grids were already approaching numerical
floors and are not evidence of asymptotic rates.

## Interpretation boundary

This result does **not** establish that binary32 indicators are safe for Euler
or shocks. Tier 1a used scalar periodic global-LF problems with a binary64
persistent state. It did not test characteristic matrices, Euler fluxes,
positivity, general boundaries, low-precision RK accumulation, or gradients.

It also does not settle the nonlinear-weight question. Tier 1a grouped weight
formation and weight normalization together. The direct Field--Gottlieb prior
work identifies high-precision normalization as important. The strong failure
of the grouped block therefore motivates, but cannot answer, a refined search
that separates numerator formation from normalization. That refinement must
be frozen before it is run.

No performance conclusion is drawn from Tier 1a. The only non-control split
that passed demotes smoothness indicators, which are a substantial but not
exclusive part of WENO cost. Its actual benefit depends on compiler fusion,
cast traffic, device, order, and problem size.

## Reproducibility

The machine-readable record is
`experiments/mixed_precision/results/phase_d_tier1_20260827/search.json`.

- Protocol commit: `c840c61`
- Implementation/source commit: `a1cc160fd71cac31ad3507bdc9208db562bfa929`
- Records: 384 = 6 orders times 64 assignments
- Result SHA-256:
  `d8415192ee96d450718c41a3167c91a0522f1d124ca1d8e9c7a9cc53c74d1e6d`
- Execution environment: Python 3.11.13, PyTorch
  `2.9.0.dev20250705+cu128`, one CPU thread; CUDA unavailable to this process

Verification command:

```bash
python experiments/mixed_precision/verify.py \
  experiments/mixed_precision/results/phase_d_tier1_20260827
```

The record contains every per-case normalized and absolute error, analytic
smooth/critical error, observed rate, safety metric, policy assignment,
environment field, and threshold needed to reproduce these classifications.
