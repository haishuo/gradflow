# Phase-D Tier-1b weight-normalization refinement results

Execution date: 2026-08-27 UTC.

Status: **the complete frozen 768-point refinement executed and verified**.

## Result

Separating nonlinear-weight formation from normalization changed the answer.
Computing the unnormalized nonlinear weights in binary32 and then converting
them to binary64 for their sum and normalization passed the `tight` contract
at every tested WENO-JS order from 5 through 15.

Computing both smoothness indicators and unnormalized weights in binary32,
while retaining binary64 normalization and all other blocks, was:

- `engineering` at WENO-5, limited by the `1e-6`-amplitude near-constant
  case; and
- `tight` at WENO-7 through WENO-15.

The complete counts were:

| Class | Count |
|---|---:|
| `tight` | 22 |
| `engineering` | 2 |
| `coarse` | 0 |
| `failed` | 744 |

The 24 passing records comprise four assignments per order: all binary64,
indicator-only binary32, weight-formation-only binary32, and both indicator
and weight-formation binary32. At WENO-5 the indicator-containing assignments
are `engineering`; all other members are `tight`.

## Error boundary

Weight-formation-only binary32 produced worst normalized maximum discrepancies
of:

| Order | Worst normalized maximum discrepancy |
|---:|---:|
| 5 | `1.809382219e-7` |
| 7 | `1.809382219e-7` |
| 9 | `2.714073328e-7` |
| 11 | `2.714073328e-7` |
| 13 | `2.714073328e-7` |
| 15 | `2.714073328e-7` |

The combined indicator/formation split remained below
`3.216176894e-6` at orders 7--15. At WENO-5 its maximum was
`1.987606367e-4`, inherited from indicator demotion.

No assignment demoting weight normalization passed. Nor did any assignment
demoting flux splitting, candidates, face combination, or divergence. Under
this contract, the observed seam is therefore not "the weights can be FP32."
It is more precise:

> Smoothness indicators and unnormalized nonlinear-weight formation may be
> candidates for binary32, but normalization and the reconstructed face-flux
> path remain binary64.

This result is consistent in shape with the prior high-precision-normalization
strategy, while testing a different precision pair and a wider generated-order
family. It is not a replication of the prior application result.

## Claim boundary

The conclusion is empirical and conditional on the frozen scalar cases and
three declared tolerance classes. It is not a proof, a universal WENO rule, or
an Euler/full-CFD qualification. In particular, a characteristic projection
can introduce a different signal scale and conditioning, shocks can change
stencil selection, and low-precision persistent RK state can accumulate errors
not exercised here.

No speedup is claimed yet. Binary32 subexpressions can introduce cast traffic,
and a compiler may already make the binary64 expression bandwidth- or
instruction-efficient. The eligible assignments must be timed after this
accuracy result is frozen.

## Reproducibility

Machine-readable record:
`experiments/mixed_precision/results/phase_d_tier1b_20260827/search.json`.

- Tier-1b protocol commit: `dc12435`
- Implementation/source commit: `3966aad2c252e27e90924cf7efa8d4d82a5e95fa`
- Records: 768 = 6 orders times 128 assignments
- Result SHA-256:
  `f8c3cb7510a1df68c31d556a74b869c857f54816a6aee176d2e55e49da7baa8e`
- Environment: Python 3.11.13, PyTorch `2.9.0.dev20250705+cu128`, one CPU
  thread; CUDA unavailable to the numerical-search process

The updated verifier continues to validate both the preserved Tier-1a schema
and this Tier-1b schema.
