# Euler boundary/shock Phase-B results

Post-study note: the explicitly deferred CPU/CUDA RHS agreement gate has now
passed across all six generated orders, both dtypes, and both boundary modes.
Representative CUDA compiler and device-resident CFL gates also passed; see
`DEFERRED_CUDA_GATES_RESULTS.md`. This document preserves the original record.

## Decision

Phase B passed its frozen correctness gate. GradFlow now has a bounded,
one-dimensional, nonperiodic characteristic finite-difference WENO-JS Euler
path for physical caller-owned point samples. Orders 5, 11, and 15 completed
the full Sod and Shu--Osher refinement studies and passed every threshold that
Phase A fixed before this implementation was run.

Orders 7, 9, and 13 completed the two 200-point admissibility pilots with
finite, positive states. Those pilots are not refinement qualifications and
do not establish the same nonperiodic shock claim as the representative
orders.

The source qualification ran from clean commit
`3b64b1a75b0e412fee195b12c341901ff0c1136d`. The immutable record is:

```text
experiments/euler_boundary_shock/results/phase_b_20260827/
```

## Implemented scientific surface

The public functions accept a conservative state with shape `(3, points)` and
no caller-visible ghost cells:

```python
euler1d_rhs(state, dx, order=..., boundary=...)
euler1d_rhs_with_boundary_fluxes(state, dx, order=..., boundary=...)
euler1d_cfl_timestep(state, dx, cfl=0.1)
euler1d_ssp_rk3_step(state, dx, dt, order=..., boundary=...)
```

The boundary choices are `periodic` and `transmissive`. Internal ghost
construction preserves tensor dtype, device, and autograd. The new path calls
the same generated WENO-JS coefficient objects and the same Roe
characteristic reconstruction algebra as the previously qualified periodic
Euler implementation; it does not contain a second copy of that mathematics.

The frozen shock formulation is face-frozen Roe characteristic projection,
global Lax--Friedrichs splitting with the preserved 1.1 enlargement,
Jiang--Shu nonlinear weights with power two and epsilon `1e-6`, 12-scaled
smoothness indicators, conservative flux differencing, and SSP-RK3 at CFL
0.1. No positivity limiter, WENO-Z substitution, adaptive epsilon, artificial
viscosity, or output-dependent fallback was added.

## Sod exact-solution results

All values below are at 800 points and final time 0.2. The three primitive L1
ceilings frozen in Phase A were `1.909e-3`, `3.968e-3`, and `1.333e-3`.

| Order | Density L1 | Velocity L1 | Pressure L1 | Energy L1 | Contact error (cells) | Shock error (cells) |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | `8.667e-4` | `1.687e-3` | `6.388e-4` | `1.712e-3` | `0.392` | `0.655` |
| 11 | `5.889e-4` | `1.299e-3` | `4.884e-4` | `1.272e-3` | `0.392` | `0.345` |
| 15 | `5.760e-4` | `1.103e-3` | `4.578e-4` | `1.198e-3` | `0.392` | `0.345` |

Density, velocity, pressure, and energy error decreased at every refinement
from 200 to 400 to 800 points for each representative order. The finest to
coarsest primitive-error ratios were between `0.246` and `0.289`, below the
frozen maximum 0.75. Both detected discontinuous wave locations were within
one cell, below the frozen three-cell maximum.

## Shu--Osher reference results

All values below are at 800 points and final time 1.8, compared by fixed
linear sampling with the independent Phase-A 12,800-cell finite-volume
WENO-Z/HLLC record.

| Order | Density L1 | Velocity L1 | Pressure L1 | Density correlation | Density TV ratio |
|---:|---:|---:|---:|---:|---:|
| 5 | `1.181e-2` | `4.035e-3` | `1.885e-2` | `0.997997` | `0.923830` |
| 11 | `6.714e-3` | `2.622e-3` | `1.232e-2` | `0.998820` | `0.969786` |
| 15 | `5.675e-3` | `2.425e-3` | `1.079e-2` | `0.998952` | `0.992856` |

Every L1 value is below its frozen ceiling. The 800/200 density-error ratios
were `0.161`, `0.207`, and `0.205`, below the maximum 0.8. Correlations exceed
the frozen minimum `0.948276`, and total-variation ratios lie inside the
frozen interval `[0.795400, 1.195400]`.

Higher order improved these observed 800-point errors and structure metrics,
but this three-order, two-problem observation is not a universal superiority
claim and contains no cost comparison.

## Admissibility and conservation

All 24 frozen shock runs completed their exact requested final times. Density
and pressure were checked after every SSP-RK3 stage, not just in saved final
arrays. The lowest recorded stage values remained positive for every case.
The smallest were density `0.122677` and pressure `0.097366` in the order-15
Sod study, and density `0.730016` and pressure `0.848706` in the order-15
Shu--Osher study.

The periodic and transmissive boundary-flux conservation residuals were zero
at the precision recorded by the roundoff-scaled gate for every generated
order 5 through 15. Uniform states and the overlap with the pre-existing
periodic line algebra also passed for float32 and float64.

## Smooth order, differentiation, and compilation

The periodic entropy-wave RHS showed these observable rates before the fixed
float64 roundoff floor `1e-11`:

| Order | Recorded evidence |
|---:|---|
| 5 | rates `5.179`, `5.223`, `5.119` |
| 7 | rates `5.836`, `5.944`, `5.882`; exceeds the frozen order-minus-two gate |
| 9 | first observable rate `9.311`; finer values crossed the floor |
| 11 | first observable rate `9.968`; finer values crossed the floor |
| 13 | coarsest error `5.191e-12`; floor-limited |
| 15 | coarsest error `2.194e-13`; floor-limited |

The floor-limited cases rely on the separately qualified exact coefficient
construction and periodic assembly-overlap evidence; no convergence rate is
inferred from roundoff-level values.

Orders 5, 11, and 15 produced finite boundary-sensitive autograd derivatives.
Their relative differences from centered finite differences were between
`4.60e-9` and `9.03e-9`, within the frozen `2e-5` relative and `2e-7`
absolute limits.

Both boundary policies at those representative orders executed eagerly and
with `torch.compile(fullgraph=True)`. Each compiler observation contained one
graph and zero graph breaks; eager/compiled maximum absolute differences were
at most `1.87e-14`. Static inspection found no host/device transfer, scalar
extraction, custom operator, handwritten CUDA, or handwritten Triton token in
the boundary/RHS numerical path.

## Artifact identities

| Artifact | SHA-256 |
|---|---|
| `qualification.json` | `95d3da968fc063d204e13effc8d6190e027e4550a4fe2063462ccbbf170c6b5d` |
| `sod_order5_n800.npz` | `fcea45bb5bd7731981d311fe8f5ed7282379d5ab621ebfb9d47c3fe87ecca939` |
| `sod_order11_n800.npz` | `e888363b4161bdf53e79de1fbf4c10567532c90a0c0894a6eced56c27b916b02` |
| `sod_order15_n800.npz` | `b2bf75b7f9b0c9029819aacb15ba48d2cb9ae27f66e6b3539fe30df93522edea` |
| `shu_osher_order5_n800.npz` | `c8767f5bca98c6a05f41d3b786ab9a41691d2e165abaabf2b5a594150b54bde6` |
| `shu_osher_order11_n800.npz` | `37c8e12f7520eb0c329cc1f901109daeeaec163b1deaddb59645a0d2a9c2653a` |
| `shu_osher_order15_n800.npz` | `7c44620901be819ebf4a7d262172e7a974c13ec1062cb5c14a8cd7a11ef1421c` |
| `SHA256SUMS` | `0950b576b3af4039a3f2c10fc04da28a5594e605cf0bce37005a817092ad36b3` |

`sha256sum -c SHA256SUMS` passed. The reusable
`experiments/euler_boundary_shock/verify_phase_b.py` additionally validates
schemas, Phase-A dependency hashes, decisions, array shapes, finiteness, and
positive density and pressure.

The coefficient payload identity for all six generated orders is
`ff02863db1c54b7d30e8ca3687e28ea4319953198c44d32e1df3789867780501`.

## Environment and unresolved device evidence

The record used Python 3.11.13, NumPy 2.2.6, PyTorch
2.9.0.dev20250705+cu128, one CPU thread, and Linux 6.8.0-134 x86-64. CUDA was
unavailable inside this execution environment. Therefore the Phase-B
CPU/CUDA agreement gate is explicitly untested in this immutable record. The
prospective Forge supplement now supplies that missing evidence. MPS remains
untested.

## Claim boundary

Phase B establishes a bounded one-dimensional Euler boundary/shock
correctness result for representative WENO-JS orders 5, 11, and 15 under the
exact frozen formulation and problems. It also establishes only admissibility
pilots for orders 7, 9, and 13.

It does not establish performance, real-time capability, production CFD,
multidimensional nonperiodic boundaries, arbitrary equations, stabilization,
Navier--Stokes, CUDA agreement in this environment, or publication novelty.
No DVEB code or result changed, and no performance campaign began in Phase B.
