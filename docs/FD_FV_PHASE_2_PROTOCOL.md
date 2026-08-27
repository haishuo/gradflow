# FD/FV Phase-2 mathematical-contract and oracle protocol

Status: frozen before Phase-2 oracle implementation.

Freeze date: 2026-08-27 UTC.

## Purpose

Phase 2 freezes the mathematical identity of the first finite-volume seed and
constructs independent, inspectable oracles for its coefficients, projections,
and invariants. It does not add an executable finite-volume method to
`src/gradflow`, advance a PDE in time, collect performance data, or compare FD
and FV speed.

The governing order remains correctness > performance > convenience. Phase 3
may implement only the contract that passes this phase; a contract change after
seeing implementation behavior requires a named constitutional amendment.

## Frozen classification and boundary

The seed is a **one-dimensional, dimension-by-dimension finite-volume
WENO-JS5 reconstruction** on a uniform periodic Cartesian mesh. Its persistent
state is the physical cell average, not a point sample and not the auxiliary
finite-difference flux used to derive classical FD-WENO.

Phase 2 includes:

- scalar conservation laws `u_t + f(u)_x = 0`;
- unique periodic cells with no duplicated endpoint;
- left and right cell-average-to-face WENO-JS5 reconstruction;
- a global Lax--Friedrichs/Rusanov face flux;
- analytic and exact-rational projection oracles; and
- the future SSP-RK3 time-integration contract, without executing it.

It excludes Euler, characteristic reconstruction, nonperiodic boundaries,
multidimensional face quadrature, genuinely multidimensional FV, AFD, WENO-Z,
positivity limiting, adaptive epsilon, arbitrary order, production API, GPU
execution, compilation, optimization, and timing.

## Grid and state semantics

For domain `[a,b)` with `N` cells,

```text
dx       = (b-a)/N
face i   = a + i*dx
cell i   = [a+i*dx, a+(i+1)*dx]
center i = a + (i+1/2)*dx
ubar_i   = (1/dx) integral(cell i) u(x) dx
```

Indices are periodic modulo `N`. Face `i+1/2` in the algebra below means the
right face of cell `i`; the stored face-flux array at index `i` follows that
convention. A cell-average array may never be described as nodal data.

Smooth initial and exact terminal states are projected by analytic integration
where available. Otherwise, the later experiment must use independently
converged quadrature whose error is negligible relative to the declared PDE
error. Center sampling is not an FV initialization substitute.

## WENO-JS5 face reconstruction

At the right face of cell `i`, the left-biased candidates are

```text
q0 =  (1/3) ubar[i-2] - (7/6) ubar[i-1] + (11/6) ubar[i]
q1 = -(1/6) ubar[i-1] + (5/6) ubar[i]   +  (1/3) ubar[i+1]
q2 =  (1/3) ubar[i]   + (5/6) ubar[i+1] -  (1/6) ubar[i+2]
```

with optimal weights `(1/10, 3/5, 3/10)`. The right-biased reconstruction is
the exact reflection about the face: candidate offset `j` becomes `1-j` while
candidate identities and optimal weights retain their order.

The standard Jiang--Shu indicators are

```text
beta0 = (13/12)(u[i-2]-2u[i-1]+u[i])^2
      + (1/4)(u[i-2]-4u[i-1]+3u[i])^2
beta1 = (13/12)(u[i-1]-2u[i]+u[i+1])^2
      + (1/4)(u[i-1]-u[i+1])^2
beta2 = (13/12)(u[i]-2u[i+1]+u[i+2])^2
      + (1/4)(3u[i]-4u[i+1]+u[i+2])^2
```

For the matched-component seed, runtime weights intentionally preserve the
existing GradFlow/Gottlieb scaling:

```text
B_k     = 12 * beta_k
epsilon = 1e-29
alpha_k = d_k / (epsilon + B_k)^2
omega_k = alpha_k / sum_j alpha_j
```

The nonlinear power is two. This is a comparison control, not a claim that the
epsilon policy is universally optimal for FV. A conventional or scale-aware FV
epsilon may enter only the later best-practical lane as a separately named
formulation. Weight evaluation must use a normalization that is algebraically
equivalent and finite on constant float32 states.

The optimal linear combination must equal the five-cell face reconstruction

```text
(1/30)u[i-2] - (13/60)u[i-1] + (47/60)u[i]
               + (9/20)u[i+1] - (1/20)u[i+2].
```

## Numerical flux and semidiscrete operator contract

For reconstructed states `uL` and `uR` at a face and a declared scalar
`alpha >= max |f'(u)|` over the applicable state/domain contract,

```text
Fhat(uL,uR) = 1/2 * (f(uL) + f(uR) - alpha*(uR-uL))
rhs[i]      = -(Fhat[i] - Fhat[i-1]) / dx.
```

`alpha` is global and fixed for one RHS evaluation in the matched seed. Its
estimation cost and policy will later be included in the complete-solve
endpoint. For linear advection `f(u)=c*u`, the frozen policy is `alpha=abs(c)`.
For positive `c`, the face flux reduces to `c*uL`; for negative `c`, to
`c*uR`.

The future complete-solve seed uses the same three-stage SSP-RK3 algebra as the
qualified FD seed and shortens only the last timestep to reach the exact final
physical time. Phase 2 records but does not execute that integrator.

## Independent oracle construction

Phase-2 oracle code must use only the Python standard library and must not
import `torch`, NumPy, `gradflow`, or its coefficient generator. It will contain
two independent lines of evidence:

1. exact-rational moment integration and exact linear solves derive candidate
   and full-stencil coefficients from cell averages; and
2. literal published WENO-JS5 coefficients and quadratic indicator matrices
   serve as a separately inspectable expected table.

The derivation must prove:

- each three-cell candidate reproduces face values of polynomials through
  degree two from exact cell averages;
- the five-cell optimal reconstruction reproduces through degree four;
- the candidate combination gives the full reconstruction;
- weights are positive and sum to one;
- smoothness matrices are symmetric, positive semidefinite, and annihilate
  constants; and
- right reconstruction is the exact face reflection of the left.

Projection cases include exact rational polynomial cell averages and analytic
periodic Fourier-mode averages. The frozen record will distinguish face values,
cell averages, and center samples and demonstrate that the latter two are not
interchangeable at finite resolution.

Semidiscrete oracle cases include constant preservation, exact periodic flux
telescoping, sign-correct linear-advection upwinding, and deterministic
nonconstant face/RHS values computed independently of future PyTorch code.

## Frozen artifacts and schemas

Phase 2 produces:

- `experiments/fd_fv_contract/fv_js5_oracle.py`;
- `experiments/fd_fv_contract/derive_phase_2.py`;
- `experiments/fd_fv_contract/verify_phase_2.py`;
- `experiments/fd_fv_contract/results/phase_2_20260827/contract.json`;
- `experiments/fd_fv_contract/results/phase_2_20260827/oracle_cases.json`;
- `experiments/fd_fv_contract/results/phase_2_20260827/SHA256SUMS`;
- `tests/test_fd_fv_phase_2_oracles.py`; and
- `docs/FD_FV_PHASE_2_RESULTS.md`.

`contract.json` records formulation identity, grid/state semantics, exact
coefficients, indicator matrices, epsilon/weight policy, numerical flux, time
integrator, boundary, precision qualification policy, and explicit exclusions.
`oracle_cases.json` records exact fractions as canonical strings, analytic
floating-point cases in hexadecimal representation where necessary, source
hashes, and invariant outcomes. The derivation refuses to overwrite an
existing frozen record unless an explicit output path is supplied.

## Acceptance gate

1. Exact derivation equals the literal candidate, optimal, full-stencil, and
   smoothness data in this protocol.
2. Polynomial reproduction, symmetry, semidefiniteness, constant nullspace,
   reflection, and weight invariants pass using exact arithmetic.
3. Analytic Fourier cell averages agree with high-accuracy direct integration
   to a declared oracle-only tolerance and differ from center samples at a
   nondegenerate frozen case.
4. Exact Fraction-based WENO reconstruction preserves constants and produces
   the frozen deterministic nonconstant face values.
5. Exact periodic Rusanov flux differences telescope to zero and choose the
   correct upwind reconstruction for both velocity signs.
6. The verifier validates schema identity, artifact hashes, and every recorded
   invariant without importing production GradFlow code.
7. Existing Phase-1 and Phase-C records still verify; the repository tree is
   clean after coherent local commits.

## Stop and claim boundary

Stop when the contract, independent oracle code, frozen records, hashes,
documentation, and tests pass. Do not add `src/gradflow/fv*.py`, expose a public
FV API, time an operator, compile an FV graph, extend to Euler/order seven or
higher, optimize, or push without explicit authorization.

Passing Phase 2 establishes an auditable specification and independent test
oracle for the Phase-3 scalar implementation. It does not establish that the FV
scheme has been implemented, is correct in PyTorch, is faster or slower than
FD, or contributes a publishable result.
