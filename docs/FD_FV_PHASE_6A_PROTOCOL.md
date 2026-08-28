# FD/FV Euler Phase-6A contract and oracle protocol

Status: frozen before any production finite-volume Euler implementation or
Euler FD/FV timing.

Freeze date: 2026-08-28 UTC.

## Question

Phase 6 asks whether conclusions from the scalar FD/FV studies survive a
nonlinear hyperbolic system:

> For the same ideal-gas Euler problem, requested accuracy or shock-quality
> contract, and execution boundary, where do qualified classical
> finite-difference and dimension-by-dimension finite-volume WENO methods each
> minimize time or memory?

Phase 6A freezes only the mathematics, projections, inherited references,
metrics, and future admission gates. It performs no production solve and
collects no performance timing.

## Governing rules

`FD_FV_EXPERIMENTAL_CONSTITUTION.md` remains normative. In particular:

- correctness precedes performance;
- both methods start from the same continuous physical state;
- FD stores point values and FV stores physical cell averages;
- equal-grid timing is secondary to achieved-error or shock-quality timing;
- matched-component and best-practical lanes remain distinct;
- failed runs, nonphysical states, compiler failures, and allocation failures
  remain results; and
- no method may silently substitute another order, flux, precision, boundary,
  or backend.

The first Euler comparison is WENO-JS5 in one space dimension. Existing
higher-order FD results are not used to give FD an asymmetric maturity
advantage. Higher order and multidimensional extension require later frozen
phases.

## Inherited independent evidence

Phase 6A reuses, without modifying, the independently prepared Euler
boundary/shock assets:

| Asset | SHA-256 |
|---|---|
| `experiments/euler_boundary_shock/sod_exact.py` | `0eb78a61f391eb0640564decfde1f9242ea1831516f948a06bd9b2720fa0e758` |
| `experiments/euler_boundary_shock/fv_reference.py` | `0d1a7aba657e953d169f72988a967df2497aaffb6f003c678a9c763d1a7ae220` |
| Phase-A manifest | `c99f4b9687f818af486f8fb5905e5363ca503cb17747626332fe4952e2e056fe` |
| Phase-A thresholds | `7c3d3c057d9b291a197a8d0c14b1cdeee79b272a39522ed351ff84909513486e` |
| Exact Sod array | `d7aa679fb05021edad4b494ac1ff3f33bfda07a9fd09b9c71c44f554d16b6858` |
| Shu--Osher FV WENO-Z/HLLC reference | `67d551dd2560c7ddead29b9c805082ef896bc2ff8bcd5a5e2cfc9856f8b02f65` |
| Qualified FD Euler record | `95d3da968fc063d204e13effc8d6190e027e4550a4fe2063462ccbbf170c6b5d` |

The Sod oracle is an exact self-similar ideal-gas Riemann solution checked by
wave relations and Rankine--Hugoniot residuals. The Shu--Osher record was
created by a separate NumPy finite-volume WENO-Z/HLLC implementation at
12,800 cells and resolution-checked before the GradFlow bounded FD path
existed. It is a high-resolution numerical reference, not an exact solution.
Its FV origin is disclosed; each target method receives the projection
appropriate to its own state semantics.

Phase 6A adds point/cell-average projection records but does not regenerate or
alter these inherited authorities.

## Continuous equations and physical variables

The one-dimensional ideal-gas Euler equations are

```text
U_t + F(U)_x = 0,
U = (rho, rho*u, E),
F = (rho*u, rho*u^2 + p, u*(E+p)),
p = (gamma-1)*(E - 0.5*rho*u^2),
gamma = 1.4.
```

Every accepted state must be finite with strictly positive density and
pressure. No positivity clipping, retry, artificial viscosity, adaptive
epsilon, or output-dependent scheme replacement is permitted in the matched
lane. A proposed stabilization is a different formulation requiring a new
identifier and prospective gate.

## Discrete state and grid convention

For a domain `[a,b]` with `N` uniform cells, `dx=(b-a)/N` and
`x_i=a+(i+1/2)dx`.

The classical FD state is `U(x_i)`. Its entries are physical point values;
`dx*sum(U_i)` is a quadrature, not a declaration that the entries are cell
averages.

The FV state is

```text
Ubar_i = (1/dx) * integral over cell i of U(x) dx.
```

Public arrays contain no ghost state. Periodic or transmissive ghost filling
is reapplied at every SSP-RK3 stage and remains an engine detail. Neither
method may initialize itself from the other method's discrete array.

## Matched-component lane

The registered seed formulations are:

```text
fd_classical_characteristic_js5_global_lf_euler1d_v1
fv_dimensional_characteristic_js5_global_matrix_lf_euler1d_v1
```

They share:

- ideal-gas Euler with `gamma=1.4`;
- WENO-JS order five, nonlinear power two, epsilon `1e-6`, and the preserved
  12-scaled Jiang--Shu indicators;
- face-frozen Roe left/right characteristic matrices;
- line-global characteristic-family LF speeds enlarged by `1.1` and clamped
  below by `1e-15`;
- conservative interface-flux differencing;
- unique physical samples/cells with periodic or transmissive boundaries;
- SSP-RK3, CFL `0.1`, and exact final-step shortening; and
- float64 as the qualification dtype.

The mathematically necessary distinction is explicit:

- FD projects stencil point states and physical fluxes through each face's
  frozen Roe matrix, forms `0.5*(L F +/- alpha L U)`, and reconstructs the
  positive and negative split characteristic fluxes.
- FV projects cell-average conservative states through each face's frozen Roe
  matrix, reconstructs left and right characteristic states, maps them back,
  and evaluates

  ```text
  Fhat = 0.5*(F(UL)+F(UR))
         - 0.5*R*diag(alpha)*L*(UR-UL).
  ```

The FV flux is a characteristic matrix global-LF flux. It is not relabeled
HLLC or scalar Rusanov. This choice holds the Roe basis and characteristic LF
dissipation policy as close as the different FD/FV reconstruction objects
permit. The methods are not claimed to be algebraically identical.

## Best-practical lane

Phase 6A registers the lane but admits no implementation. Candidate FD and FV
methods must receive a maturity audit and independent justification before
their identities are frozen. The inherited primitive-variable WENO-Z/HLLC
code is an oracle generator, not automatically the GradFlow best-practical FV
competitor. No matched-component result may be pooled with a later
best-practical result.

## Frozen physical problems and projections

### Smooth periodic entropy wave

On `[0,1)`,

```text
rho(x,t) = 1 + 0.1*sin(2*pi*(x-0.7*t)),
u = 0.7,
p = 1.
```

This is an exact Euler solution. Phase 6A records exact point values, exact
cell-average conserved values, pointwise spatial RHS, and cell-average spatial
RHS on `N=(24,36,54,81)`. The cell averages and their time derivatives are
analytic integrals, not quadrature approximations.

### Sod shock tube

On `[0,1]`, with interface `0.5`, transmissive boundaries, and final time
`0.2`:

```text
left  (rho,u,p) = (1,0,1),
right (rho,u,p) = (0.125,0,0.1).
```

Use `N=(200,400,800)`. The interface is a cell face for every grid. FD uses
the exact solution at cell-center coordinates. FV uses exact conservative
cell averages obtained by fixed Gauss--Legendre integration split at every
known self-similar wave location. A 32-versus-64-point quadrature comparison
must be at most `5e-13`, and the domain-integrated conserved state must agree
with the exact boundary-flux balance within `5e-13`.

Primitive FV errors are computed by applying the ideal-gas conversion to the
actual and exact conservative cell averages. They are not compared with
pointwise primitive values at cell centers.

### Shu--Osher shock--entropy interaction

The domain, initial states, boundary, and final time `1.8` are inherited from
the boundary/shock record. Use `N=(200,400,800)`.

- FD evaluates the frozen 12,800-cell primitive reference by the existing
  fixed linear interpolation to FD point coordinates.
- FV conservatively restricts the frozen 12,800-cell conserved reference by
  exact block averaging. All target sizes divide 12,800.

Derived primitive FV metrics convert the restricted conservative averages to
primitive variables. The two projections are recorded separately and never
subtracted from one another as though they represented the same discrete
quantity.

For initial data, constant regions are exact. The right entropy-wave density
cell average is integrated analytically. The `x=-4` interface is a cell face
for every frozen size.

## Phase-6B correctness admission

Phase 6B may implement only the matched FV formulation. Before any timing,
both registered formulations must pass the same applicable gates:

1. exact Phase-6A initial and terminal projection identities;
2. exact uniform-state preservation;
3. periodic entropy-wave spatial and complete-solve error decreasing on the
   frozen sequence, with at least one L2 rate `>=4.0` before a declared
   `1e-11` float64 floor;
4. componentwise boundary-flux conservation ratio at most `64`;
5. positive density and pressure after every SSP-RK3 stage;
6. the inherited Sod primitive L1, refinement, and wave-location ceilings;
7. the inherited Shu--Osher error, refinement, correlation, and
   total-variation ceilings;
8. eager float64 CPU execution;
9. CPU/CUDA maximum absolute agreement at most `5e-11` on frozen smooth probes;
10. fixed-shape `torch.compile(fullgraph=True)` with one graph, zero breaks,
    and eager/compiled disagreement at most `5e-11`;
11. finite smooth-state directional derivatives agreeing with centered
    differences within `2e-5` relative or `2e-7` absolute; and
12. static and runtime evidence of no hidden host/device transfer or scalar
    extraction inside reconstruction, RHS, or SSP-RK3 stages.

The inherited thresholds are ceilings, not targets to tune toward. Failure of
either method blocks the matched performance campaign. A prospective
resolution may diagnose a failed gate but may not edit its original record.
Shock gradients are outside the differentiability claim.

## Future Phase-6C performance constitution

Only after Phase 6B passes may a separate committed protocol freeze timing.
Its primary outcomes are achieved smooth error or declared shock quality
versus complete-solve time and peak memory. Equal-grid RHS/step cost remains a
causal diagnostic.

CPU eager/compiled, CUDA eager/compiled, resident, prepared-transfer, cold,
and genuinely packaged AOT endpoints must remain separate. Compilation counts
in cold latency. Native or DVEB endpoints enter only when they implement the
exact registered formulation and pass the same output gates. DVEB is judged
against a credible native ceiling as well as higher-level deployment
baselines; GradFlow does not require DVEB support to proceed.

## Phase-6A artifacts and stop condition

The Phase-6A generator must:

- refuse an existing output directory and a dirty source tree;
- record source revision and hashes;
- verify every inherited hash;
- emit deterministic formulation, projection, evaluation, and gate contracts;
- emit smooth exact projections, Sod exact point/cell-average projections,
  and Shu--Osher conservative restrictions;
- record quadrature and conservation cross-checks;
- hash every artifact; and
- provide an independent verifier that performs no GradFlow production solve.

Stop after the protocol, immutable oracle records, verifier, results document,
tests, coherent local commits, and a clean tree. Do not implement FV Euler,
collect timing, optimize either formulation, change DVEB, add mixed precision,
begin multidimensional Euler, or claim publication novelty in Phase 6A.
