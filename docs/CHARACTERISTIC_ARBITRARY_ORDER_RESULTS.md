# Characteristic arbitrary-order WENO-JS results

Status: **the frozen 3-D Euler qualification gate passed**.

## Outcome

GradFlow now applies the exact generated WENO-JS reconstruction to the
existing Roe-characteristic compressible-Euler formulation. The public
`Solver` accepts JS orders 5, 7, 9, 11, 13, and 15 in float32 or float64 for
direct PyTorch execution on CPU or CUDA.

The migration preserves Shu's duplicated periodic endpoints, gamma `1.4`,
per-line characteristic global LF speeds enlarged by `1.1`, epsilon `1e-6`,
12-times-scaled Jiang--Shu indicators, and SSP-RK3 integration. At each face,
the required split-flux samples are projected through that face's frozen left
Roe matrix, reconstructed with the generated scalar algebra, and transformed
back with the matching right Roe matrix.

This is not a general CFD solver. It covers ideal-gas Euler in three
dimensions with duplicated periodic endpoints. It does not add Navier--Stokes
viscosity, general boundaries, alternative flux splitting, arbitrary
equations, curvilinear geometry, or performance qualification.

## WENO-5 preservation

The generated characteristic reconstruction was compared with the preserved
bakeoff implementation on the periodic vortex:

| Dtype | RHS maximum difference | Full SSP-RK3 step difference | Frozen bound |
|---|---:|---:|---:|
| float32 | `2.5332e-7` | `5.9605e-8` | `5e-6` |
| float64 | `1.2212e-15` | `1.1102e-16` | `2e-12` |

The active order-five path therefore retains the established Shu result while
using the same generated reconstruction machinery as the higher orders.

## Smooth characteristic convergence

The frozen entropy wave has constant pressure and velocity with smooth
periodic density variation. It is an exact 3-D Euler solution whose spatial
RHS is known analytically. The table reports L2 rates across
`N=(24,36,54,81)` in the varying direction:

| Order | Measured successive rates |
|---:|---|
| 5 | 4.824, 4.809, 4.659 |
| 7 | 6.260, 6.185, 6.022 |
| 9 | 8.648, 8.596, 8.151 |
| 11 | 11.376, 10.467, 10.087 |
| 13 | 14.510, 12.359, 12.096 |
| 15 | 18.074, 14.643, 10.540 |

Every family refined monotonically and exceeded its frozen `order-2`
threshold before the float64 floor. The separately qualified exact scalar
coefficient payload is unchanged, with SHA-256
`bee81f0ba84338fc2136039e4ef4a680c89cab38fb124391a354bdcd58ae553b`.

## Invariants, differentiation, and devices

- Uniform physical states produced exactly zero RHS in both float32 and
  float64 for all six orders.
- The worst conservation residual was `0.2243 * eps * sum(abs(rhs))`, below
  the declared factor-eight bound.
- Fixed-step `Solver` gradients were finite and nonzero for orders 5, 11, and
  15 in float64.
- The worst CPU/CUDA difference was `2.0266e-5` in float32 and
  `3.7748e-14` in float64, below the frozen `3e-4` and `5e-11` bounds.
- Orders 5, 11, and 15 each captured as one Dynamo graph with zero graph
  breaks on CPU and CUDA.
- Compiled CPU output differed from eager by at most `2.6646e-14`; compiled
  CUDA output differed by at most `2.8611e-5` in float32.
- Static tests found no hidden device conversion, scalar extraction, NumPy
  conversion, custom operator, handwritten CUDA, or handwritten Triton in the
  numerical loop.

Fresh-cache lowering of the larger characteristic graphs was operationally
substantial. This run deliberately did not measure compilation or execution
as a benchmark, so it supports only a compiler-correctness claim. Compile
economics, AOT packaging, and representation selection remain future work.

## Backend boundary

DVEB was not modified. Its hash-qualified native artifact remains fixed to
float32 characteristic WENO-5. Explicit native requests for higher orders or
float64 now refuse with a precise reason. Automatic placement may fall back
to direct PyTorch; it cannot silently substitute order five.

## Reproducibility

The machine-readable record is
`experiments/characteristic_arbitrary_order/results/qualification_20260826.json`.
It identifies clean source commit
`2fe0b47a5c0d173a8a11ab83ed780a74e26628f2` and has SHA-256
`b9d560353fdd255ca489b00e3fb20693afebbf7a9cf85c6a2b7214f3b0d7e98f`.

The final complete GradFlow suite with real CUDA and DVEB v1/v2 artifacts
passed 155 tests. Only the separately supplied, verified DVEB placement-model
test was skipped.

The run used Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0, and the NVIDIA
GeForce RTX 5070 Ti. MPS remains untested. Orders beyond 15 are not qualified
for characteristic Euler or `Solver`. Novelty and publishability remain
unclaimed.
