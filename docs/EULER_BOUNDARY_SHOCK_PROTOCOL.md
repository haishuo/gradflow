# One-dimensional Euler boundary and shock protocol

Status: frozen planning boundary before numerical implementation.

## Question

Can the generated characteristic WENO-JS mathematics be applied to a bounded
one-dimensional compressible-Euler problem with explicit nonperiodic boundary
closures, preserve discrete conservation including boundary flux, reproduce
an exact Riemann solution under refinement, and resolve a standard
shock--entropy interaction without changing the scheme after observing the
answer?

This is a correctness trunk. It includes no timing, backend selection, DVEB
development, Navier--Stokes, product UI, or publication claim.

## Phase A: oracle and threshold freeze

No canonical boundary implementation begins until an oracle-only preparation
commit records:

- an independently derived exact ideal-gas Euler Riemann solver for the Sod
  problem, tested against its own wave relations and limiting states;
- the exact input grids and sample convention;
- a preserved high-resolution reference for the Shu--Osher interaction from
  an independent implementation or separately derived reference procedure;
- SHA-256 identities for reference arrays and their construction records; and
- numerical acceptance thresholds selected from oracle resolution/error
  analysis without running the GradFlow implementation under test.

This two-stage freeze prevents post-hoc tolerances and prevents a visually
plausible shock plot from serving as the oracle.

## Frozen equation and reconstruction

- one-dimensional ideal-gas compressible Euler;
- conservative state `(rho, rho*u, E)` and gamma `1.4`;
- uniform Cartesian physical point samples with no public ghost cells;
- face-frozen Roe characteristic projection;
- characteristic-wise global Lax--Friedrichs splitting with the preserved
  `1.1` enlargement;
- generated Jiang--Shu orders 5, 7, 9, 11, 13, and 15;
- nonlinear power two, epsilon `1e-6`, and the preserved 12-scaled
  smoothness indicators;
- conservative flux differencing; and
- SSP-RK3 with CFL `0.1`, shortening only the final step to reach the declared
  final time.

No positivity limiter, adaptive epsilon, WENO-Z substitution, local flux
split, filter, or artificial viscosity is introduced in this trunk. If a
qualified order produces a nonphysical state on a frozen problem, that is a
result. Any stabilization proposal begins a separate formulation and gate.

## Grid and boundary contract

The public state contains only physical point samples. For design order
`2r-1`, boundary handling creates the required `r` ghost samples internally
and reapplies the boundary at every SSP-RK3 stage.

Two boundary policies are included:

- `periodic`: ghost samples wrap from the opposite physical side, with no
  duplicated public endpoint; and
- `transmissive`: every left/right ghost state is the constant extrapolation
  of the first/last physical state.

Boundary construction must preserve dtype, device, layout, and autograd. It
must use ordinary tensor operations without a hidden host transfer or scalar
extraction in the numerical loop.

The numerical fluxes at the two physical domain faces must remain available
to the qualification harness. At every RHS evaluation,

```text
dx * sum(rhs over physical points) + (right boundary flux - left boundary flux)
```

must satisfy a roundoff-scaled bound component by component. This is the
nonperiodic conservation gate; total state is not expected to remain constant
when flux crosses an open boundary.

## Frozen problems

### Smooth periodic entropy wave

A constant-pressure, constant-velocity density wave on `[0, 1)` tests the new
one-dimensional system assembly and all orders. Its exact translation at a
fixed time supplies L1/L2/Linf convergence, conservation, and periodic
boundary evidence without a shock.

### Sod shock tube

On `[0, 1]`, with the discontinuity at `x=0.5`, gamma `1.4`, and transmissive
boundaries, primitive initial states are:

```text
left:  (rho, u, p) = (1.0,   0.0, 1.0)
right: (rho, u, p) = (0.125, 0.0, 0.1)
```

The final time is `0.2`. The exact Riemann oracle is sampled at exactly the
same physical points. Orders 5, 11, and 15 are the representative shock gates;
the remaining qualified orders are recorded but cannot be advertised for
nonperiodic Euler if they fail admissibility or refinement.

### Shu--Osher shock--entropy interaction

On `[-5, 5]`, with the interface at `x=-4`, gamma `1.4`, and transmissive
boundaries, primitive initial states are:

```text
left:  (rho, u, p) = (3.857143, 2.629369, 10.33333)
right: (rho, u, p) = (1 + 0.2*sin(5*x), 0.0, 1.0)
```

The final time is `1.8`. Orders 5, 11, and 15 are compared on the frozen grid
sequence against the independent high-resolution reference. This problem
tests shock interaction with smooth small-scale structure; it does not have a
simple exact terminal solution, so its acceptance thresholds are deferred
only to the Phase-A oracle freeze, not to implementation results.

The initial conditions follow the standard Sod survey problem and the
Shu--Osher shock-capturing test. Bibliographic identities are:

- G. A. Sod, *A Survey of Several Finite Difference Methods for Systems of
  Nonlinear Hyperbolic Conservation Laws*, JCP 27 (1978),
  <https://doi.org/10.1016/0021-9991(78)90023-2>.
- C.-W. Shu and S. Osher, *Efficient Implementation of Essentially
  Non-oscillatory Shock-Capturing Schemes, II*, JCP 83 (1989),
  <https://doi.org/10.1016/0021-9991(89)90222-2>.

## Qualification categories

### Correctness and admissibility

- Exact uniform states produce zero RHS for both boundary policies.
- Every accepted run remains finite with positive density and pressure at
  every recorded stage.
- Smooth periodic errors decrease monotonically and reach the order-specific
  frozen convergence band before roundoff.
- Sod density, velocity, pressure, and energy errors decrease under the frozen
  refinement sequence against the exact solution.
- Sod wave locations and plateau states meet the Phase-A bounds.
- Shu--Osher density error and resolved-structure metrics meet the Phase-A
  bounds without post-hoc smoothing.
- The boundary-flux conservation identity passes for every tested RHS.

### Cross-implementation evidence

- Generated WENO-5 interior flux/RHS probes agree with the already qualified
  order-five characteristic algebra where their contracts overlap.
- Float64 CPU is the primary oracle execution policy.
- Float32 and float64 CPU/CUDA agreement is recorded for every accepted order.
- Fixed-step gradients are finite and agree with centered finite differences
  on a smooth boundary-sensitive objective before shock gradients are claimed.

### Compiler and transfer behavior

- Eager execution passes before compilation is attempted.
- Representative orders 5, 11, and 15 capture as one fixed-shape graph with
  zero graph breaks on available CPU/CUDA environments.
- Static and runtime checks find no hidden device/host transfers or scalar
  extraction inside boundary filling, RHS, or SSP-RK3 stages.

## API and architecture boundary

This trunk may expose a narrow experimental 1-D Euler problem surface, but it
must not prematurely freeze the commercial equation registry. Shared pieces
may be extracted from `euler3d.py` only when equivalence tests protect the
existing 3-D results. Copying and independently evolving the reconstruction
algebra is not acceptable.

The public state excludes ghost cells so boundary storage remains an engine
detail. Equation, scenario, boundary, discretization, and backend identifiers
remain separate as required by `PROBLEM_MODEL.md`.

## Evidence record and stop condition

The qualification recorder must refuse to overwrite results and record source
revision/dirty state, coefficient payload, oracle hashes, complete problem
definitions, grids, errors, conservation, admissibility, device agreement,
autograd, graph behavior, and environment/device identities.

Stop after:

- the Phase-A oracle record and thresholds are committed;
- the implementation and public-boundary tests pass or failures are preserved;
- the qualification JSON and SHA-256 are committed;
- existing scalar, 3-D Euler, and DVEB integration tests still pass; and
- the working tree is clean.

Do not time the new path, optimize representations, change DVEB, add a
positivity method, or begin Navier--Stokes in this trunk.
