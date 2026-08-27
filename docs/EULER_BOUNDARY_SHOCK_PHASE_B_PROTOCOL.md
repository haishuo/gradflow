# Euler boundary/shock Phase-B protocol

Status: frozen before canonical Phase-B implementation.

This document refines `EULER_BOUNDARY_SHOCK_PROTOCOL.md` after Phase A fixed
the oracle arrays and thresholds. It changes no Phase-A number.

## Public implementation surface

Phase B adds narrow scientific functions, not a general equation registry:

```python
euler1d_rhs(state, dx, order=..., boundary=...)
euler1d_rhs_with_boundary_fluxes(state, dx, order=..., boundary=...)
euler1d_cfl_timestep(state, dx, cfl=0.1)
euler1d_ssp_rk3_step(state, dx, dt, order=..., boundary=...)
```

The conservative state shape is `(3, points)` with `(rho, rho*u, E)`. Public
states contain physical point samples only. Supported boundaries are exactly
`periodic` and `transmissive`; every other value is rejected. The functions
preserve float32/float64 dtype, device, layout semantics, and autograd.

The flux-diagnostic function returns the RHS and physical left/right numerical
fluxes with shape `(3, 2)`. The ordinary RHS calls the same implementation and
discards only the diagnostic output.

Phase B does not add a new `Solver` equation registry, adaptive CUDA host
control loop, named scenario API, positivity method, or backend selector.

## Shared implementation rule

The one-dimensional path reuses the existing generated `WENOJS` instances,
Roe characteristic matrices, flux construction, epsilon, smoothness scaling,
and nonlinear reconstruction. Boundary-aligned slicing is new; reconstruction
or eigensystem algebra may not be copied into an independently evolving path.

The existing 2-D/3-D periodic implementation remains numerically unchanged.
A periodic-overlap test must compare the new ghosted line result with the
already qualified duplicated-endpoint line implementation for every order.

## Frozen smooth and local gates

For every order 5, 7, 9, 11, 13, and 15:

- uniform states must produce a maximum absolute RHS at most `2e-12` in
  float64 and `2e-5` in float32 for both boundaries;
- the boundary-flux conservation ratio must be at most the Phase-A value 64;
- the periodic entropy-wave L2 error must decrease on point counts
  `24, 36, 54, 81`; and
- at least one observed rate before the roundoff floor must reach `order-2`.

The entropy wave uses point locations `(i+1/2)/N`, density
`1 + 0.1*sin(2*pi*x)`, velocity `0.7`, pressure `1`, and its exact spatial
Euler RHS. This is an RHS convergence test, not a time-integration test.

Float32/float64 CPU/CUDA agreement uses 37 smooth points for both boundaries.
Frozen absolute bounds are `3e-4` and `5e-11`. CUDA skips with an explicit
reason when unavailable.

Orders 5, 11, and 15 additionally require:

- eager and fixed-shape `torch.compile(fullgraph=True)` RHS execution for both
  boundaries;
- one captured graph and zero graph breaks where graph explanation is
  available; and
- finite autograd directional derivatives agreeing with centered finite
  differences to relative error `2e-5` and absolute error `2e-7` in float64.

Static inspection must find no `.cpu()`, `.cuda()`, `.to()`, `.item()`,
`.numpy()`, custom operator, handwritten CUDA, or handwritten Triton inside
the boundary/RHS/RK numerical path.

## Frozen shock execution

All shock runs use float64 CPU, CFL `0.1`, exact final-step shortening, and
SSP-RK3. Physical state is checked after every stage. Failure produces a
record; it does not trigger clipping or a different scheme.

Orders 5, 11, and 15 run Sod and Shu--Osher on 200, 400, and 800 points. The
remaining orders 7, 9, and 13 run both problems at 200 points to record
admissibility; they are not advertised for the nonperiodic path merely because
the coefficient generator can construct them.

Sod is compared directly with the exact primitive solution sampled at the
same point locations. L1 thresholds and refinement ratios are exactly those
in the Phase-A `thresholds.json`.

Two discontinuous Sod wave locations are measured without reading GradFlow
output to choose a method:

- contact: maximum absolute density difference between adjacent points within
  `0.05` of the exact contact; and
- right shock: maximum absolute density difference within `0.05` of the exact
  shock.

The inferred interface coordinate is the midpoint of those adjacent samples.
Each must be within three grid spacings of its exact self-similar location.
Rarefaction head/tail accuracy is covered by full-field exact errors rather
than an additional feature detector.

Shu--Osher is compared by linear interpolation of the frozen 12,800-cell
primitive reference onto the GradFlow points. L1, 200/800 density ratio,
density correlation, and density total-variation ratio use the exact Phase-A
thresholds and the fixed window `[-3,3]`.

## Evidence record

The refusal-to-overwrite qualification record contains:

- source commit and dirty state;
- Phase-A manifest, threshold, and array hashes;
- coefficient payload identity;
- smooth, uniform, periodic-overlap, and conservation results;
- shock errors, wave locations, admissibility minima, and step counts;
- autograd, device, graph, and transfer/static-inspection evidence;
- full environment identity; and
- explicit claim-boundary fields.

The recorder is committed and tested before the record is generated from a
clean source revision. Phase B stops after the record and interpretation are
committed, the complete suite passes, and the worktree is clean.

No performance measurement, DVEB work, Navier--Stokes, stabilization,
commercial API expansion, or publication claim occurs in Phase B.
