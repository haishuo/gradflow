# Euler boundary/shock Phase-A results

## Decision

Phase A passed. GradFlow now has a hash-frozen exact Sod oracle, a separately
implemented finite-volume WENO-Z/HLLC reference procedure validated against
that exact solution, a resolved 12,800-cell Shu--Osher reference, and
acceptance thresholds chosen before the GradFlow nonperiodic implementation
exists.

The reference source is clean commit
`6ba01248d8c55303c87771931263f35319265451`. The immutable record is:

```text
experiments/euler_boundary_shock/results/phase_a_20260827/
```

Phase B may now implement the canonical one-dimensional finite-difference
boundary/shock path. This result does not qualify that implementation in
advance.

## Independence

The exact Sod solver derives the ideal-gas Euler pressure function, star state,
shock relation, and rarefaction fan directly. It imports neither GradFlow nor
PyTorch.

The high-resolution numerical procedure is intentionally different from the
future GradFlow path:

| Property | Phase-A numerical reference | Future GradFlow implementation |
|---|---|---|
| Discretization | cell-centered finite volume | pointwise finite difference |
| Reconstruction | componentwise primitive WENO-Z | Roe-characteristic WENO-JS |
| Interface flux/split | HLLC | global Lax--Friedrichs |
| Epsilon policy | WENO-Z reference policy | fixed `1e-6`, 12-scaled JS indicators |
| Time integration | SSP-RK3, CFL 0.4 | SSP-RK3, CFL 0.1 |
| Boundary | transmissive constant extrapolation | transmissive/periodic under test |

Agreement in Phase B will therefore not be agreement between two copies of
the same reconstruction or flux policy.

## Exact Sod validation

For gamma 1.4 and primitive states `(1, 0, 1)` and `(0.125, 0, 0.1)`, the
oracle recorded:

| Quantity | Value/error |
|---|---:|
| Star pressure | `0.3031301780506469` |
| Star velocity | `0.9274526200489501` |
| Left/right star density | `0.4263194281784953` / `0.2655737117053071` |
| Pressure-function residual | `3.331e-16` |
| Right-shock Rankine--Hugoniot residual | `2.776e-17` |
| Rarefaction isentropy error | `0` |
| Rarefaction characteristic error | `0` |
| Rarefaction Riemann-invariant error | `0` |
| Far-field error | `0` |

The independently implemented finite-volume procedure then refined
monotonically against the exact solution:

| Cells | Density L1 | Velocity L1 | Pressure L1 | Fallbacks |
|---:|---:|---:|---:|---:|
| 100 | `4.382e-3` | `8.987e-3` | `3.384e-3` | 0 |
| 200 | `2.150e-3` | `4.588e-3` | `1.690e-3` | 0 |
| 400 | `1.148e-3` | `2.377e-3` | `8.423e-4` | 0 |
| 800 | `6.364e-4` | `1.323e-3` | `4.442e-4` | 0 |
| 1600 | `3.331e-4` | `5.102e-4` | `2.084e-4` | 0 |

## Shu--Osher resolution

The reference used the standard domain `[-5,5]`, interface `-4`, and final
time `1.8`. Each lower grid was sampled against the 12,800-cell array:

| Cells | Density L1 | Velocity L1 | Pressure L1 | Density correlation | TV ratio | Fallbacks |
|---:|---:|---:|---:|---:|---:|---:|
| 800 | `1.316e-2` | `4.116e-3` | `2.237e-2` | `0.998276` | `0.995400` | 0 |
| 1600 | `6.000e-3` | `2.268e-3` | `1.120e-2` | `0.999098` | `1.026361` | 0 |
| 3200 | `2.548e-3` | `1.015e-3` | `5.182e-3` | `0.999752` | `1.016854` | 0 |
| 6400 | `1.022e-3` | `4.009e-4` | `2.266e-3` | `0.999961` | `1.004430` | 0 |

The frozen resolution gate required the 6,400/12,800 density L1 difference to
be at most `2.5e-3`; the observed value was `1.0215728511964153e-3`.
All five Shu--Osher runs remained positive and required zero reconstruction
fallbacks. The 12,800-cell run used 27,709 SSP-RK3 steps.

The roundoff-normalized boundary-flux conservation residual was at most
`1.130` across the reference runs, below the frozen allowance of 64.

## Frozen Phase-B thresholds

Thresholds were derived from the independent 800-cell results using a factor
of three, with no GradFlow boundary implementation present.

For Sod at 800 points, Phase B must meet L1 ceilings:

```text
density  0.001909058930464399
velocity 0.003967892198278969
pressure 0.0013325232068413769
```

Each variable must decrease over 200, 400, and 800 points; the finest/coarsest
error ratio must be at most 0.75; wave locations must be within three cells;
and density and pressure must remain strictly positive.

For Shu--Osher at 800 points against the 12,800-cell reference, Phase B must
meet L1 ceilings:

```text
density  0.039490369039826
velocity 0.012346812213100595
pressure 0.06710303053685818
```

The 800/200 density-error ratio must be at most 0.8. In `[-3,3]`, density
correlation must be at least `0.9482758575795494`, and the total-variation
ratio must lie in `[0.7953996862683976, 1.1953996862683978]`.

These are acceptance ceilings, not expected GradFlow errors and not a claim
that the independent reference method is optimal.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| `manifest.json` | `c99f4b9687f818af486f8fb5905e5363ca503cb17747626332fe4952e2e056fe` |
| `thresholds.json` | `7c3d3c057d9b291a197a8d0c14b1cdeee79b272a39522ed351ff84909513486e` |
| `sod_exact_t0p2_n8192.npz` | `d7aa679fb05021edad4b494ac1ff3f33bfda07a9fd09b9c71c44f554d16b6858` |
| `sod_fv_wenoz_hllc_t0p2_n1600.npz` | `285694d08b55196505551e75eba8908a92703ead96b0287b10b0794612bac04f` |
| `shu_osher_fv_wenoz_hllc_t1p8_n12800.npz` | `67d551dd2560c7ddead29b9c805082ef896bc2ff8bcd5a5e2cfc9856f8b02f65` |
| `SHA256SUMS` | `b96706606e3fb33bb3abca42e35aaf6be99e6645dbab3a4e2e5bb76ed8626aa7` |

`sha256sum -c SHA256SUMS` passed. Every compressed array was reopened and
checked for expected shapes, finite values, and positive density/pressure.
The reusable `experiments/euler_boundary_shock/verify_phase_a.py` performs
both checks.

## Test and environment record

The source qualification passed 117 tests with 46 declared skips in the
available CPU environment. The seven Phase-A-specific tests passed. CUDA was
unavailable to this execution environment, so existing CUDA tests skipped;
Phase A itself is a float64 CPU reference preparation and makes no device
performance claim.

The frozen manifest records Python 3.11.13, NumPy 2.2.6, and Linux
6.8.0-134-generic x86-64. The numerical recorder measured no performance.

## Claim boundary

Phase A establishes oracles and preimplementation thresholds only. It does not
establish GradFlow nonperiodic boundary correctness, shock robustness,
arbitrary-order discontinuity accuracy, CUDA agreement, compiler behavior,
performance, DVEB capability, Navier--Stokes, or publication novelty. Those
claims remain gated on later trunks.
