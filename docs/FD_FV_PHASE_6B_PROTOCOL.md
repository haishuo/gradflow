# FD/FV Euler Phase-6B matched correctness protocol

Status: frozen before the matched production FV Euler implementation.

Freeze date: 2026-08-28 UTC.

## Purpose

Phase 6B asks whether the matched dimension-by-dimension FV formulation frozen
in Phase 6A can be implemented in ordinary PyTorch and satisfy the same
scientific capability contract already met by the registered classical FD
formulation.

This is correctness qualification only. It contains no performance timing,
representation optimization, best-practical method, DVEB work, mixed
precision, multidimensional extension, or publication claim.

## Immutable dependencies

The complete `FD_FV_PHASE_6A_PROTOCOL.md` and immutable Phase-6A contract and
projection records govern state semantics, formulation identities,
mathematics, inherited oracle identities, and thresholds. Phase 6B must verify
them before executing production code.

The existing FD evidence is the immutable Euler boundary/shock Phase-B record,
supplemented by the deferred Forge CUDA gates. Phase 6B does not rerun its full
18-run shock campaign merely to create a newer timestamp. It does rerun common
local smooth, conservation, compiler, device, and derivative probes so the two
registered sources share current-environment evidence.

## Public FV surface

Phase 6B may add exactly this narrow API:

```python
euler1d_fv_rhs(cell_averages, dx, boundary=...)
euler1d_fv_rhs_with_boundary_fluxes(cell_averages, dx, boundary=...)
euler1d_fv_cfl_timestep(cell_averages, dx, cfl=0.1)
euler1d_fv_ssp_rk3_step(cell_averages, dx, dt, boundary=...)
```

The state shape is `(3,cells)` in conservative order `(rho,rho*u,E)`. Values
are physical cell averages, not point samples. Supported boundaries are
exactly `periodic` and `transmissive`. No caller-visible ghosts, automatic
point/average conversion, device selection, fallback, positivity clipping, or
general solver registry is introduced.

The formulation identifier is

```text
fv_dimensional_characteristic_js5_global_matrix_lf_euler1d_v1.
```

## Shared-algebra rule

The FV path must reuse:

- the existing Euler state validation, boundary normalization, ghost filling,
  and on-device CFL calculation;
- the exact-generated order-five `WENOJS` reconstruction object;
- the existing face-frozen Roe matrices and line-global characteristic-family
  speeds; and
- one reconstruction helper for both face biases.

It may not duplicate coefficient tables, smoothness formulas, Roe
eigenvectors, or CFL mathematics. Refactoring shared utilities is permitted
only with equivalence tests protecting the qualified FD path.

For each physical face, the left Roe matrix projects every required
cell-average stencil sample. Generated WENO-JS reconstructs left and right
characteristic states using exact reflected offsets. The matching right Roe
matrix returns conservative face states. The numerical flux is exactly the
Phase-6A characteristic matrix global-LF flux. Physical flux evaluation and
flux differencing preserve state dtype and device.

## Frozen local matrix

Both FD and FV use float64 and WENO-JS5.

### Projection identity

Load the exact Phase-6A projection arrays. The harness-created FD point and FV
cell-average entropy states must match their respective arrays exactly. Shock
initial arrays must match the stored Shu--Osher arrays and the registered,
cell-face-aligned Sod constants exactly; terminal Sod projections come from
the stored exact arrays. No production numerical operator is used to generate
an oracle.

### Uniform states

For 19 cells and both boundaries, maximum absolute RHS must be at most
`2e-12`.

### Smooth spatial convergence

Use the exact Phase-6A entropy-wave states and spatial RHS at
`N=(24,36,54,81)`, time zero. L1, L2, and Linf errors must decrease. At least
one consecutive L2 rate must be `>=4.0` before the `1e-11` float64 floor.

### Smooth complete-solve convergence

Advance the entropy wave to `T=0.1` with periodic boundaries, SSP-RK3, the
on-device CFL formula at CFL `0.1`, and exact final-step shortening. Compare
each discrete state with its method-appropriate exact Phase-6A projection.
L1/L2 errors must decrease, and at least one L2 rate must be `>=2.5` before the
`1e-11` floor. The third-order threshold reflects SSP-RK3 with `dt=O(dx)`; it
does not relabel the spatial scheme as third order.

### Conservation

For a 43-cell smooth positive state and both boundaries, require componentwise

```text
dx*sum(rhs) + right_boundary_flux - left_boundary_flux
```

to have roundoff-scaled ratio at most `64`. Periodic complete solves also use
the prospective accumulated-roundoff policy established in Phase 5CR rather
than a single-update bound applied to an entire integration.

### Differentiation

For each formulation and both boundaries, a three-stage fixed-step objective
on 19 smooth cells must produce finite gradients. Its directional derivative
must agree with centered differences within `2e-5` relative or `2e-7`
absolute. No shock-gradient claim is made.

### Compiler and device

For each formulation and boundary:

- eager CPU must pass first;
- fixed-shape `torch.compile(fullgraph=True)` must produce one graph and zero
  graph breaks;
- compiled/eager maximum absolute difference must be at most `5e-11`;
- float64 CPU/CUDA maximum absolute difference must be at most `5e-11` when
  Forge CUDA is admitted; and
- outputs must retain input shape, dtype, and device.

The qualification runner requires Forge CUDA visibility and fresh admission;
it does not convert a hidden process into an `unavailable` hardware claim.

Static inspection and CPU/CUDA profiler probes must find no host/device copy,
NumPy conversion, scalar extraction, custom operator, handwritten CUDA, or
handwritten Triton inside the FV reconstruction/RHS/RK numerical path. Explicit
test-harness input and final-output transfers are outside that path.

## Frozen shock matrix

The new FV path runs Sod and Shu--Osher at `N=(200,400,800)` in float64 on CPU,
with transmissive boundaries, CFL `0.1`, and exact final-step shortening.
Density and pressure are checked after every SSP-RK3 stage. A nonphysical stage
is a recorded failure; the run does not clip, retry, or substitute a method.

### Sod evaluation

Compare against the exact conservative cell averages in the Phase-6A record.
Primitive metrics apply the same conservative-to-primitive conversion to
actual and exact averages. Require the inherited density, velocity, and
pressure L1 ceilings, monotonic refinement, finest/coarsest ratio at most
`0.75`, positive stage states, and contact/right-shock locations within three
cells. Energy L1 is recorded.

### Shu--Osher evaluation

Compare against the conservatively restricted Phase-6A 12,800-cell reference.
Require the inherited primitive L1 ceilings, finest/coarsest density ratio at
most `0.8`, density correlation at least the inherited minimum, density
total-variation ratio inside the inherited interval, and positive stage
states. Conserved errors are recorded alongside derived primitive errors.

The FD shock decision is inherited without converting its point-reference
errors into FV-average errors. Cross-method error-to-time comparison is
deferred to Phase 6C.

## Qualification record

The committed runner must precede execution and refuse a dirty source tree or
existing output directory. It records:

- clean source revision and source/protocol hashes;
- Phase-6A and inherited FD record identities;
- all raw spatial, solve, conservation, derivative, compiler, device,
  movement, and shock metrics;
- stage minima, step counts, exact final times, and failures;
- environment and admitted-device identity;
- explicit top-level gate derivation; and
- `performance_measurements_collected=false`.

An independent verifier recomputes array identities, errors, convergence,
threshold decisions, conservation, shock metrics, hashes, and the top-level
decision without rerunning timed or production work.

## Stop condition

Stop after the implementation either passes or its immutable failure is
preserved, the verifier and complete configured suite run, coherent local
commits exist, and the worktree is clean.

Do not begin Phase 6C, measure execution time, add WENO-Z/HLLC, alter the FD
formulation, optimize after observing behavior, change DVEB, add arbitrary
order FV, or push without explicit authorization.
