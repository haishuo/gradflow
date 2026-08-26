# Arbitrary-order finite-difference WENO-JS seed results

Status: **the frozen scalar periodic qualification gate passed**.

## Outcome

GradFlow now constructs finite-difference Jiang--Shu WENO schemes from exact
rational mathematics rather than order-specific coefficient tables. One
constructor generated and qualified orders 5, 7, 9, 11, 13, and 15. The
ordinary-PyTorch numerical path supports an arbitrary tensor axis, leading
batch dimensions, float32 and float64, global Lax--Friedrichs splitting,
autograd, CPU, and CUDA.

This completes only the scalar unique-periodic-node seed. It does not yet make
the public `Solver` arbitrary-order, migrate the characteristic Euler path,
add boundaries, lower a generated scheme through DVEB, or establish any
performance result.

## Exact construction

For order `p=2r-1`, the generator uses `fractions.Fraction` arithmetic to:

1. recover each degree-`r-1` auxiliary flux polynomial from its `r` cell
   averages;
2. evaluate it at the interface;
3. construct the `2r-1` point optimal reconstruction;
4. solve exactly for the positive optimal linear weights;
5. integrate squared derivatives exactly to obtain every Jiang--Shu
   smoothness matrix; and
6. compute an exact LDLT factorization for nonnegative sum-of-squares runtime
   evaluation.

The canonical serialization of all qualified exact coefficients, matrices,
and factors has SHA-256
`bee81f0ba84338fc2136039e4ef4a680c89cab38fb124391a354bdcd58ae553b`.

Generated order five exactly reproduces the independently written standard
candidate coefficients, full-stencil coefficients, smoothness matrices, and
weights `(1/10, 3/5, 3/10)`. Its floating RHS differs from the historical
Gottlieb correction form by at most `1.7141843500e-13` across positive linear,
negative linear, and Burgers probes, within the frozen `5e-13` bound.

## Smooth convergence

The table reports measured L2 rates across `N=(24,36,54,81)` for the frozen
smooth periodic mixed Fourier problem. Exact polynomial reproduction through
degree `p-1` independently proves the optimal stencil's design order.

| Order | Measured successive rates |
|---:|---|
| 5 | 4.228, 5.061, 5.074 |
| 7 | 5.793, 6.306, 6.059 |
| 9 | 8.525, 9.134, 8.944 |
| 11 | 10.671, 10.825, 10.163 |
| 13 | 13.259, 13.896, 12.545 |
| 15 | 16.087, 15.296, 14.241 |

Every order refined monotonically and met the frozen requirement of at least
one rate `p-1` or better before the float64 floor. These measurements should
not be read as a universal asymptotic statement for every smooth function;
WENO-JS nonlinear weights are sensitive to critical points.

## Critical-point result

The frozen family `u(x)=sin(2*pi*x)^3` has a higher-order critical point at
`x=0`. Pointwise RHS errors decreased monotonically for every order, but the
observed rates demonstrate genuine Jiang--Shu order loss:

| Order | Successive critical-point rates |
|---:|---|
| 5 | 1.979, 1.995, 1.999 |
| 7 | 6.166, 6.241, 6.070 |
| 9 | 6.927, 7.280, 6.935 |
| 11 | 9.396, 9.786, 9.959 |
| 13 | 11.611, 11.328, 10.854 |
| 15 | 13.579, 13.778, 12.350 |

This is a mathematical limitation of the qualified WENO-JS policy, not a
compiler defect. GradFlow did not change epsilon, introduce WENO-Z, or retune
the gate after observing it.

## Differentiability, devices, and compilation

- Float64 `gradcheck` passed for generated orders 5, 11, and 15.
- Conservation residuals were below `0.154 * eps * sum(abs(rhs))` for every
  qualified order, compared with the declared factor-eight bound.
- The worst CPU/CUDA difference was `1.3113021851e-05` in float32 and
  `2.0872192863e-14` in float64.
- Orders 5, 11, and 15 each produced one Dynamo graph with zero graph breaks
  on CPU and CUDA under the fixed-shape gate.
- Compiled CPU output matched eager exactly in the probes. The worst compiled
  CUDA/eager difference was `5.7220458984e-06` in float32.
- Static inspection found no hidden device/dtype conversion, scalar
  extraction, NumPy conversion, custom operator, CUDA source, or Triton source
  in the numerical path.

The RTX 5070 Ti float64 observations are correctness checks only. No float64
or other performance inference is made from this consumer GPU.

## Reproducibility and limits

The focused gate passed 61 tests. The final complete GradFlow suite with real
CUDA and DVEB v1/v2 artifacts passed 101 tests with only the optional verified
placement-model check skipped. The machine-readable record is
`experiments/weno_js_arbitrary_order/results/qualification_20260826.json`,
SHA-256
`cbf9d20b7967c3b6832da24764954944e458be24d18c1adcd58bfcf6620ac2d5`.

Orders beyond 15 are constructible but unqualified. MPS is untested. General
boundaries, systems, characteristic projection, alternate nonlinear weights,
adaptive epsilon policies, coefficient-conditioning limits, representation
performance, and native lowering are separate future trunks. Novelty and
publishability remain unclaimed pending systematic literature review.
