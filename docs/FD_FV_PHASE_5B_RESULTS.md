# FD/FV nonlinear Phase-5B qualification result

Status: **passed on CPU and Forge CUDA; no performance timing collected**.

Qualification source commit:
`7eb2ba2f8d8a181557bffcda3e49214d3bb6e0b5`.

The protocol was prospectively frozen at commit `0d7b427` before the production
Burgers module existed. The immutable qualification record is
`experiments/fd_fv_nonlinear/results/phase_5b_20260828/qualification.json`,
SHA-256
`1d3d79cb7be53b46be2e0514de3fc03cbcf60099ca8f0883db2e15d10e559799`.

## Implemented mathematical surface

`src/gradflow/burgers.py` now exposes three narrow ordinary-PyTorch operations:

- `burgers_flux(u)` for `u^2/2`;
- `burgers_fd_weno5_rhs`, which applies classical split-physical-flux
  FD-WENO-JS5 to point values; and
- `burgers_fv_weno5_rhs`, which reconstructs physical cell averages to face
  states before applying the Rusanov flux.

Both accept an explicit caller-supplied global LF speed. No point/average
conversion, wave-speed estimation, device selection, general solver framework,
or duplicated WENO algebra was added. The frozen experiment uses `alpha=0.7`.

## Oracle, conservation, and differentiation

All eight frozen Phase-5A initial/terminal point and cell-average projections
were reproduced exactly at their hexadecimal binary64 representation. The
independent oracle retained no PyTorch, NumPy, GradFlow, or WENO dependency.

Both formulations preserved the constant-state RHS and one SSP-RK3 step within
the `5e-13` gate. Every semidiscrete and complete-solve conservation check
passed. Maximum three-step JVP versus centered-difference discrepancies were:

- FD: `1.3321e-10` absolute and `3.8121e-10` relative L2;
- FV: `1.0167e-10` absolute and `3.5296e-10` relative L2.

These are smooth, fixed-alpha discrete-map results. They do not establish
differentiability through a maximum-speed branch or a shock.

## Spatial result

At `N=(24,36,54,81)`, every whole-grid and noncritical error decreased.
Consecutive L1 spatial rates were:

| Formulation | Whole-grid L1 rates | Noncritical L1 rates |
|---|---|---|
| Classical FD | 3.972, 4.093, 4.380 | 5.543, 5.219, 4.889 |
| Dimensional FV | 5.095, 5.030, 5.071 | 5.361, 5.342, 5.135 |

The FD result is consistent with WENO-JS critical-point degradation: its final
whole-grid L2 spatial rate was `3.663`, while the fixed physical region away
from critical points retained approximately fifth-order behavior. FV retained
approximately fifth-order whole-grid behavior for this particular smooth
Burgers problem and cell-average/Rusanov formulation.

This is a bounded numerical observation, not evidence that FV is universally
more accurate. The two methods have different correct discrete states, and
future comparison must remain accuracy-matched rather than treating equal `N`
as equal accuracy.

## Complete-solve result

At exact physical time `T=0.1`, L1 errors were:

| N | FD L1 | FV L1 |
|---:|---:|---:|
| 24 | 2.4258e-5 | 1.1140e-5 |
| 36 | 4.0690e-6 | 1.4755e-6 |
| 54 | 7.3529e-7 | 2.0257e-7 |
| 81 | 1.4255e-7 | 2.6875e-8 |

Final consecutive L1 rates were `4.046` for FD and `4.982` for FV. Final L2
rates were `3.612` and `4.702`, respectively. All terminal errors were far
below the frozen `2e-5` ceiling and all mass checks passed.

## CPU, CUDA, compiler, and movement

Every FD/FV RHS and SSP-RK3-step callable compiled on CPU and CUDA as one graph
with zero graph breaks. CPU compiled/eager differences were zero. CUDA
compiled/eager differences were at most `8.216e-15`; CPU/CUDA eager differences
were at most `1.1103e-16`.

Forge CUDA status is `admitted`, not merely visible. The admitted environment
was the RTX 5070 Ti, compute capability 12.0, driver 580.173.02, CUDA 13.0, and
PyTorch 2.13.0+cu130. MPS is `host_confirmed_absent` on Forge.

Four profiler probes—FD/FV on CPU/CUDA—reported no copy or movement events.
Their `aten::to` dispatches allocated zero CPU or device memory, and every
output remained on the input device and dtype. Static inspection found no
transfer, scalar-extraction, NumPy, Triton, or custom-extension surface in the
new module.

## Claim and next boundary

Phase 5B establishes two correct, conservative, differentiable, compilable,
device-resident smooth Burgers JS5 implementations under the frozen matched
contract. It also establishes a material accuracy difference on this case that
must be respected by the later accuracy-to-time comparison.

It establishes no nonlinear-shock behavior, performance result, FD/FV winner,
best-practical implementation, multidimensional result, dynamic-alpha policy,
mixed-precision result, or publication claim. Phase 5C may freeze a nonlinear
performance protocol, but its primary comparison must be achieved error versus
time and memory—not equal-grid timing alone.
