# FD/FV Phase-3 scalar implementation and qualification protocol

Status: frozen before canonical finite-volume implementation.

Freeze date: 2026-08-27 UTC.

## Purpose

Phase 3 implements exactly the Phase-2 contract
`fv_dimensional_js5_global_lf_periodic_v1` in readable ordinary PyTorch and
qualifies it before any timing. It asks whether the first scalar FV seed is
mathematically faithful, differentiable, device-preserving, and compilable.

Phase 3 does not compare performance with finite difference, choose a winner,
optimize representations, extend to arbitrary order or Euler, change DVEB, or
add a general solver framework.

Correctness > performance > convenience remains binding. A failed
configuration is recorded, not repaired by changing a threshold after the
implementation is observed.

## Frozen implementation surface

The canonical module is `src/gradflow/fv_weno5.py`. Its small public scientific
surface is:

```python
left, right = gradflow.fv_weno5_face_states(cell_averages, axis=-1)
face_flux = gradflow.fv_global_lax_friedrichs_flux(
    left, right, flux, alpha
)
rhs = gradflow.fv_weno5_rhs(cell_averages, dx, flux, alpha, axis=-1)
```

The functions accept only caller-owned `torch.Tensor` state with float32 or
float64 dtype. Leading dimensions are batches and `axis` identifies the unique
periodic cell dimension. The result preserves state shape, dtype, and device.
The functions perform no implicit host/device movement, scalar extraction,
NumPy conversion, custom operation, C++/CUDA/Triton call, or hidden boundary
storage.

The module may reuse GradFlow's existing exact-generated order-five
cell-average reconstruction algebra because Phase 2 proved why its coefficient
tables are common to the FV physical-cell-average and FD auxiliary-flux
derivations. The FV module must nevertheless state physical cell-average
semantics and express state reconstruction, numerical flux, and conservative
divergence explicitly. It must not call the FD `rhs`, split physical fluxes, or
describe cell averages as nodal values.

`alpha` is required and may be a Python scalar or a scalar tensor already on
the state device. Automatic wave-speed estimation is excluded from this seed
so its future preparation cost and policy cannot be hidden. `flux(left)` and
`flux(right)` must preserve shape, device, and dtype.

SSP-RK3 uses the existing algebraically generic `gradflow.ssp_rk3_step`; no
second time-integrator implementation is added solely to attach an FV name.

## Frozen oracle parity

The independent Phase-2 Fraction record is the implementation oracle. On its
eight-cell deterministic state, float64 left/right face states, positive- and
negative-speed numerical fluxes, and RHS values must agree with the exact
fractions at `rtol=0`, `atol=2e-13`.

A constant `7/3` state must reconstruct and remain stationary within:

- float64: `rtol=0`, `atol=5e-15` for faces and RHS;
- float32: `rtol=0`, `atol=2e-6` for faces and RHS.

The implementation must reject integer state, fewer than five cells, invalid
axis/bias, mismatched flux shape/device/dtype, nonpositive or nonscalar alpha,
and tensor `dx` on a different device. Python `dx` and scalar tensor `dx` are
legal and must be positive.

## Smooth spatial gate

For periodic positive and negative linear advection on `[0,1)`, initialize the
physical cell averages of

```text
u(x) = sin(2*pi*x) + 0.15*cos(6*pi*x).
```

Use `N = (32, 48, 72, 108)`, `dx=1/N`, `f(u)=c*u`, `alpha=abs(c)`, and
`c in {1,-1}`. Compare the FV RHS with the exact cell-average time derivative

```text
-c * (u(x_{i+1/2}) - u(x_{i-1/2})) / dx.
```

The float64 L2 error must decrease at every refinement and at least one
consecutive rate must be `>=4.7`. Fifth-order spatial convergence is the claim;
the exact polynomial proof remains primary near roundoff.

## Complete-solve smooth gate

Advance the same field with `c=1`, SSP-RK3, unique periodic cells, and

```text
N          = (24, 36, 54, 81)
final_time = 0.01
nominal_dt = 0.2 * dx**(5/3)
```

Shorten only the final timestep. Compare against analytic cell averages of the
field translated by `final_time`. L1 and L2 errors must decrease at every
refinement; at least one consecutive L2 rate must be `>=4.0`. The total mass at
the final time must satisfy

```text
abs(dx*sum(final-initial))
<= 32*eps*dx*sum(abs(initial)) + 1e-15.
```

This is a correctness experiment, not a runtime endpoint.

## Discontinuity gate

For positive unit-speed periodic advection, initialize exact physical cell
averages of the indicator of `[0.2,0.6)` on `[0,1)`. Run SSP-RK3 to time `0.2`
with `N=(64,128,256)` and `dt <= 0.2*dx`, shortening the final step.

Against exact translated cell averages:

- L1 error must decrease at every refinement;
- every value must be finite;
- the final minimum must be `>=-0.1` and maximum `<=1.1`; and
- mass change must satisfy the smooth complete-solve conservation bound.

This gate records shock/discontinuity behavior for linear transport. It does
not qualify nonlinear scalar shocks, positivity-preserving FV, or Euler.

## Differentiation gate

On a deterministic smooth float64 state:

1. `fv_weno5_rhs` must pass `torch.autograd.gradcheck` with
   `eps=1e-6`, `atol=2e-5`, and `rtol=2e-4` for a scalar RHS objective.
2. A fixed three-step SSP-RK3 objective must have finite gradients and agree
   with a centered directional derivative using step `1e-6` within
   `atol=3e-6`, `rtol=3e-5`.

`alpha` is explicit and fixed in both checks so differentiation does not
silently cross a global-max branch. No discontinuity-gradient claim is made.

## Device and compiler gate

Eager CPU execution must pass for float32 and float64. When CUDA is visible:

- CPU/CUDA face and RHS agreement uses `rtol=0`, `atol=2e-4` for float32 and
  `atol=2e-11` for float64;
- representative eager and compiled runs preserve CUDA residency; and
- no transfer occurs inside the resident numerical loop.

If CUDA or Apple MPS is unavailable, it is recorded as untested rather than
simulated or claimed. The local consumer GPU's FP64 limitation remains relevant
only to later timing, not correctness tolerance.

On CPU, and CUDA when visible,
`torch.compile(fullgraph=True, dynamic=False)` must execute the fixed-shape RHS
and one SSP-RK3 step with eager agreement. `torch._dynamo.explain` must report
one graph and zero graph breaks. Compilation latency is recorded only as
present/not timed; no duration enters Phase 3.

Static AST inspection must find no `.cpu()`, `.cuda()`, `.to()`, `.item()`, or
`.numpy()` call in `fv_weno5.py`. A profiler probe must find no `aten::to`,
`aten::_to_copy`, or device-copy event in the fixed-dtype CPU RHS. Ordinary
tensor allocation and arithmetic are expected.

## Qualification record

`experiments/fd_fv_qualification/qualify_phase_3.py` writes one immutable JSON
record containing:

- source revision and dirty state;
- Phase-2 contract/oracle hashes;
- source hashes for the FV module and qualification code;
- Python, PyTorch, compiler, CPU, CUDA, and MPS availability;
- oracle parity and refusal results;
- spatial and complete-solve errors/rates;
- discontinuity errors, extrema, and conservation;
- autograd and directional-gradient results;
- eager device/dtype agreement;
- compiler graph counts/breaks and eager parity;
- static/profiler transfer evidence; and
- explicit `passed`, `failed`, or `untested_unavailable` status per gate.

The writer refuses to overwrite the canonical record. The verifier checks
schema, source/hash identities, thresholds, statuses, and a separate
`SHA256SUMS` manifest.

## Acceptance and stop condition

Phase 3 passes when all available-environment mandatory gates pass, unavailable
CUDA/MPS gates are explicitly untested, the independent Phase-2 record still
verifies, the existing CPU test suite passes, coherent local commits exist,
and the working tree is clean.

Stop after qualification. Do not benchmark FD versus FV, add a best-practical
variant, extend FV beyond WENO-JS5/scalar periodic problems, add Euler, create
automatic selection, optimize the representation, change DVEB, or push without
new explicit authorization.

Passing Phase 3 establishes a correct and differentiable scalar PyTorch FV
seed. It does not establish performance, general finite-volume capability,
FD/FV superiority, arbitrary-order FV, production readiness, novelty, or
publishability.
