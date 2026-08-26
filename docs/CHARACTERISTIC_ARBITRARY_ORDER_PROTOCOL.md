# Characteristic arbitrary-order WENO-JS protocol

Status: frozen before implementation on branch
`codex/characteristic-arbitrary-order`.

## Question

Can GradFlow apply its exact generated finite-difference WENO-JS mathematics
to the existing characteristic compressible-Euler path, expose the qualified
orders through `Solver`, and preserve the established Shu WENO-5 result?

This is a correctness and API-migration trunk. It is not a performance
campaign.

## Frozen mathematical contract

The migrated path keeps the existing Euler formulation:

- ideal-gas compressible Euler with gamma `1.4`;
- Roe characteristic matrices frozen independently at each face;
- characteristic-wise, per-line global Lax--Friedrichs speeds enlarged by
  `1.1`;
- Jiang--Shu nonlinear power two;
- smoothness indicators scaled by 12, matching the preserved Shu algebra;
- fixed Euler epsilon `1e-6` for every order; and
- conservative finite-difference flux differencing.

For design order `p=2r-1`, each face projects the required physical split-flux
samples through that face's left Roe matrix. The generated candidate
polynomials, positive optimal weights, and exact-factorized smoothness
indicators then reconstruct the positive family from the left and the negative
family from the right. The reconstructed characteristic flux is transformed
back with the same face's right Roe matrix.

The grid stores both endpoints in every periodic direction. Before each RHS
or stage, the final stored endpoint is treated as authoritative and copied to
the first endpoint, preserving the ancestral convention. Reconstruction uses
the unique periodic cells; the returned RHS and state restore the duplicated
endpoint.

## Public surface

The accepted system surface is deliberately narrow:

```python
solver = gradflow.Solver(
    equations="euler",
    dimension=3,
    weno=("JS", order),
    flux_split="global_lf",
    boundaries="periodic_duplicated",
    dtype=torch.float32,  # or torch.float64
    spacing=...,
)
```

Orders 5, 7, 9, 11, 13, and 15 are candidates for qualification. An order is
not advertised through `Solver` merely because the scalar coefficient
constructor can create it. Each spatial direction must contain at least
`order` unique periodic cells.

Direct eager PyTorch is the implementation under test. Fixed-step execution
must preserve the caller's device, dtype, and autograd graph without hidden
transfers. CPU `final_time` control retains its existing declared host control
behavior.

The existing DVEB artifact is mathematically fixed to 3-D Euler
characteristic JS-WENO-5 in float32. Native eligibility must therefore reject
higher order and float64 explicitly. `backend="auto"` may fall back to direct
PyTorch with an inspectable reason; it must never substitute WENO-5 for a
higher-order request.

## Qualification gate

### WENO-5 preservation

- Generated characteristic order five must agree with the preserved bakeoff
  implementation for the full RHS and one SSP-RK3 vortex step.
- Bounds are `2e-12` absolute in float64 and `5e-6` absolute in float32.
- Existing CFL, physical-state, endpoint, and fixed-step behavior must remain
  intact.

### Higher-order mathematics

- Orders 5, 7, 9, 11, 13, and 15 must show monotonically decreasing spatial
  L2 error on a smooth periodic 3-D Euler entropy-wave family.
- At least one successive rate must reach `order-2` before the float64 floor.
  Exact scalar polynomial reproduction remains the independent coefficient
  gate; the entropy wave tests the characteristic system assembly.
- A uniform physical state must be preserved to `2e-12` in float64 and
  `2e-5` in float32.
- The sum of the RHS over unique periodic cells must satisfy a declared
  roundoff-scaled conservation bound for every conserved component.

### Differentiability and execution

- Fixed-step `Solver` execution for orders 5, 11, and 15 must produce finite,
  nonzero input gradients in float64.
- CPU/CUDA agreement is required for every qualified order in float32 and
  float64, with absolute bounds `3e-4` and `5e-11`, respectively.
- Orders 5, 11, and 15 must execute eagerly and under fixed-shape
  `torch.compile(fullgraph=True)` on CPU and CUDA.
- Dynamo explanation must record one graph and zero graph breaks for the
  compiled characteristic RHS probes.
- Static inspection must find no `.cpu()`, `.cuda()`, `.to()`, `.item()`,
  `.numpy()`, custom operator, handwritten CUDA, or handwritten Triton in the
  numerical loop.

CUDA checks skip with their reason when CUDA is unavailable. MPS is recorded
as untested unless Apple Silicon is physically available.

## Evidence and reproducibility

The qualification recorder will emit a refusal-to-overwrite JSON record with:

- source commit and dirty-state identity;
- exact coefficient payload identity;
- environment and device identity;
- WENO-5 preservation errors;
- convergence, constant-state, and conservation results;
- device agreement, autograd, and graph behavior; and
- explicit claim-boundary fields.

The record and its SHA-256 will be committed only after the protocol and
implementation commits exist.

## Explicit exclusions

This trunk does not add Navier--Stokes viscosity, nonperiodic boundaries,
characteristic boundary closures, local LF or Roe flux splitting, WENO-Z,
adaptive epsilon, unstructured or curvilinear grids, arbitrary equations,
DVEB code generation, backend calibration, timing, optimization, or
publication claims.

Failure of a higher order is a result to record. Thresholds and mathematical
policies will not be changed after observing the qualification output merely
to make an order pass.
