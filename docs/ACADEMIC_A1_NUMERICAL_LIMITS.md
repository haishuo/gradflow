# Academic A1 order-dependent numerical limits

Status: **complete under the frozen A1 scalar contract**.

Date: 2026-08-30 (UTC)

## Coefficient and construction diagnostics

These are binary64, coordinate-basis diagnostics. They quantify growing
numerical difficulty; they are not intrinsic condition numbers for the WENO
method and do not replace exact polynomial-reproduction proofs.

| Order | Minimum optimal weight | Weight range | Max candidate coefficient L1 | Full moment condition | Max restricted smoothness condition | Max exact numerator / denominator bits |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | `1.000e-1` | 6 | 3.333 | `5.317e1` | `4.998e1` | 6 / 6 |
| 7 | `2.857e-2` | 18 | 5.333 | `2.761e3` | `2.344e3` | 16 / 18 |
| 9 | `7.937e-3` | 60 | 8.533 | `2.717e5` | `1.215e5` | 35 / 37 |
| 11 | `2.165e-3` | 200 | 13.867 | `4.313e7` | `6.732e6` | 60 / 62 |
| 13 | `5.828e-4` | 700 | 23.010 | `1.007e10` | `3.900e8` | 100 / 104 |
| 15 | `1.554e-4` | 2450 | 39.010 | `3.250e12` | `2.333e10` | 145 / 148 |

The exact generator remains valid through order 15, but floating execution is
not numerically order-neutral. From order 5 to 15, the raw-monomial full
moment condition grows from about 53 to `3.25e12`, the smallest optimal weight
shrinks by roughly 644, and the maximum candidate L1 norm grows by about 12.

## Sampled roundoff floors

The smooth linear-advection RHS was evaluated at powers of two from `N=32` to
`8192`. “Onset” is the first sampled post-minimum point whose L2 error rises by
more than 5% from the preceding point.

| Order | Float32 minimum L2 at N | Float32 onset | Float64 minimum L2 at N | Float64 onset |
| ---: | ---: | ---: | ---: | ---: |
| 5 | `1.493e-5` at 256 | 512 | `8.769e-12` at 8192 | not reached |
| 7 | `8.872e-6` at 128 | 256 | `4.746e-13` at 2048 | 4096 |
| 9 | `6.252e-6` at 64 | 128 | `1.041e-13` at 512 | 1024 |
| 11 | `4.769e-6` at 64 | 128 | `5.532e-14` at 256 | 512 |
| 13 | `2.841e-6` at 32 | 64 | `5.059e-14` at 128 | 256 |
| 15 | `2.475e-6` at 32 | 64 | `3.299e-14` at 128 | 256 |

Higher order reaches a lower sampled error with fewer points, but exhausts
the useful refinement range earlier. This is the important boundary: WENO-15
is not simply “more accurate at every resolution.” All 108 roundoff samples
were finite and passed the frozen conservation bound.

## Critical-point behavior retained from qualification

For `u(x)=sin(2*pi*x)^3` at the higher-order critical point, the recorded
successive pointwise rates were:

| Order | Successive rates |
| ---: | --- |
| 5 | 1.979, 1.995, 1.999 |
| 7 | 6.166, 6.241, 6.070 |
| 9 | 6.927, 7.280, 6.935 |
| 11 | 9.396, 9.786, 9.959 |
| 13 | 11.611, 11.328, 10.854 |
| 15 | 13.579, 13.778, 12.350 |

The WENO-5 loss to approximately second order is especially strong. These
results describe classical WENO-JS; they do not evaluate WENO-Z or another
critical-point correction.

## Epsilon sensitivity

The canonical scalar epsilon `1e-29` produced no material change relative to
the `1e-40` comparison lane for any order, family, or amplitude in the frozen
sweep. The largest differences were at rounding scale.

The number of material-change records across all 30 epsilon/family/amplitude
lanes per order was:

| Order | Material changes |
| ---: | ---: |
| 5 | 11 |
| 7 | 8 |
| 9 | 6 |
| 11 | 6 |
| 13 | 3 |
| 15 | 0 |

Material changes came from larger epsilons, especially when amplitude was
small. Many changes reduced the smooth-error norm by driving weights toward
their optimal linear values; that is not evidence for a better shock-capturing
default. The sweep does not exercise discontinuity selection, and the
characteristic Euler implementation has a distinct ancestral epsilon policy.
No epsilon changed as a result of A1.

## Declared failure and qualification boundaries

| Boundary | Current status |
| --- | --- |
| Scalar periodic orders 5--15 | Qualified under the frozen smooth, critical, conservation, representative differentiation/device/compiler gates |
| Characteristic periodic Euler orders 5--15 | Qualified under its distinct Shu contract |
| Sod and Shu--Osher boundaries | Qualified at representative orders 5, 11, and 15 |
| Scalar mixed precision | Bounded passing seam exists |
| Higher-order characteristic mixed precision | Frozen strict gate failed; all-FP64 remains default |
| Extreme-resolution 1-D compilation | Compiled/eager gate failed in the face-ownership study; no timing claim |
| Orders above 15 | Constructible but unqualified |
| MPS | Untested |
| General multidimensional boundaries, Navier--Stokes, geometry | Untested and outside first-paper scope |

Machine-readable values are in
`experiments/academic_a1/evidence/a1_20260830/numerical_limits.json`.
