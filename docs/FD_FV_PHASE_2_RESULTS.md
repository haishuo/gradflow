# FD/FV Phase-2 mathematical-contract results

Phase 2 completed on 2026-08-27 under the preimplementation protocol frozen at
commit `4638a4ab592338ce24e268ef549c7d960e03605d`.

No finite-volume source was added to `src/gradflow`, no PDE was advanced in
time, and no performance measurement was collected. The result is an exact,
independent specification and oracle package for the Phase-3 scalar FV-JS5
implementation.

## Frozen seed

The first comparator is now identified unambiguously as:

```text
fv_dimensional_js5_global_lf_periodic_v1
```

It is a one-dimensional, dimension-by-dimension finite-volume WENO-JS5 method
for scalar conservation laws on unique periodic uniform cells. The persistent
state contains physical cell averages. Left and right face states are
reconstructed from those averages and supplied to a global
Lax--Friedrichs/Rusanov numerical flux. The future complete solver uses SSP-RK3,
but the integrator was recorded rather than executed in this phase.

The matched-component weight policy is deliberately identical to the existing
GradFlow/Gottlieb convention:

```text
B_k     = 12*beta_k
epsilon = 1e-29
alpha_k = d_k/(epsilon+B_k)^2
```

This prevents the first comparison from silently changing nonlinear-weight
behavior. It does not assert that this is the best FV epsilon policy. A
conventional or scale-aware alternative belongs to the later best-practical
lane and must carry a different formulation identity.

## Exact derivation result

An independent standard-library implementation integrated exact cell-average
moments, solved the reconstruction systems with `fractions.Fraction`, and
recovered the literal WENO-JS5 tables.

The left candidates at the right face of cell `i` are:

```text
q0 =  (1/3)u[i-2] - (7/6)u[i-1] + (11/6)u[i]
q1 = -(1/6)u[i-1] + (5/6)u[i]   +  (1/3)u[i+1]
q2 =  (1/3)u[i]   + (5/6)u[i+1] -  (1/6)u[i+2]
```

The exact optimal weights are `(1/10, 3/5, 3/10)`. Their combination equals

```text
(1/30)u[i-2] - (13/60)u[i-1] + (47/60)u[i]
               + (9/20)u[i+1] - (1/20)u[i+2].
```

All three exact smoothness matrices matched the literal Jiang--Shu quadratic
forms. Every principal minor was nonnegative, each matrix was symmetric, and
the constant vector was in its nullspace.

The mathematical lineage is Jiang and Shu's efficient WENO construction and
Shu's later FD/FV survey; Phase 2 independently derives the particular
cell-average reconstruction rather than treating either publication as an
executable oracle:

- [Jiang and Shu 1996](https://doi.org/10.1006/jcph.1996.0130)
- [Shu 2016](https://doi.org/10.1016/j.jcp.2016.04.030)

The exact gates passed:

- nine candidate checks: each of three candidates reproduced monomials of
  degrees zero through two;
- five full-stencil checks: the optimal reconstruction reproduced monomials of
  degrees zero through four;
- positive optimal weights summing exactly to one;
- exact left/right face reflection; and
- derived coefficients and matrices equal to the literal independent tables.

## Why the FD and FV tables overlap without becoming the same method

The coefficient overlap is expected and scientifically useful. FV reconstructs
a face point value from physical cell averages. Classical conservative FD-WENO
derives the same algebra by interpreting nodal physical-flux samples as cell
averages of an auxiliary numerical flux. GradFlow's exact FD coefficient
generator already follows that auxiliary-cell-average construction.

This does **not** make the methods interchangeable:

- the FV persistent array is `ubar_i`, the average over a physical cell;
- the FD persistent array is `u(x_i)`, a physical point value;
- their initialization and exact terminal projections therefore differ;
- for nonlinear equations, FV reconstructs states and then evaluates a
  numerical flux, whereas classical FD reconstructs split flux samples; those
  operations do not commute; and
- multidimensional FV introduces face averages/quadrature choices absent from
  this one-dimensional seed.

For linear advection with `alpha=abs(c)`, the FV Rusanov flux reduces exactly
to `c*uL` for positive velocity and `c*uR` for negative velocity. In that
special case the reconstruction arithmetic can look identical if supplied the
same array, but supplying the same array would violate the point-versus-average
state contract.

## Projection oracle

The domain convention is `[a,b)` with `N` cells:

```text
cell i = [a+i*dx, a+(i+1)*dx]
ubar_i = integral(cell i, u(x) dx)/dx.
```

An analytic periodic Fourier projection was checked independently against
4,096-panel composite Simpson integration in every cell. The maximum difference
was `2.220446049250313e-16`, below the frozen `2e-15` oracle tolerance.

For the nondegenerate eight-cell frozen case, the maximum difference between a
true cell average and the cell-center sample was
`8.810876102370824e-02`. This is direct evidence that center sampling cannot be
used as FV initialization merely because it is convenient.

## Semidiscrete invariants

The independent Fraction oracle uses

```text
Fhat = 0.5*(f(uL)+f(uR)-alpha*(uR-uL))
rhs[i] = -(Fhat[i]-Fhat[i-1])/dx.
```

It passed:

- exact preservation of a constant `7/3` state;
- exact positive-velocity selection of the left reconstructed state;
- exact negative-velocity selection of the right reconstructed state;
- exact periodic telescoping, `dx*sum(rhs) = 0`, for both signs; and
- frozen nonconstant left/right face and RHS values for future implementation
  parity.

Those deterministic values are stored as exact canonical fractions, not
rounded decimal expectations.

## Independence and evidence identity

The oracle and record generator use only the Python standard library. Static
verification rejects imports of `torch`, NumPy, or `gradflow`. The record
contains source hashes for the protocol, exact oracle, and derivation script.

Frozen artifacts:

- `experiments/fd_fv_contract/fv_js5_oracle.py`;
- `experiments/fd_fv_contract/derive_phase_2.py`;
- `experiments/fd_fv_contract/verify_phase_2.py`;
- `experiments/fd_fv_contract/results/phase_2_20260827/contract.json`;
- `experiments/fd_fv_contract/results/phase_2_20260827/oracle_cases.json`; and
- `experiments/fd_fv_contract/results/phase_2_20260827/SHA256SUMS`.

The generator refuses to overwrite an existing frozen record. The verifier
recomputes the contract and all oracle cases, validates the stored hashes, and
checks the recorded invariants.

## Claim and next-phase boundary

Phase 2 establishes that GradFlow has an auditable, independently verified
mathematical target for one scalar FV-JS5 implementation. It does not establish
that the future PyTorch code is correct, differentiable, compilable, efficient,
or competitive.

Phase 3 may now implement only this scalar seed and test it against these
oracles. It must pass eager correctness, smooth convergence, conservation,
discontinuity behavior, CPU/CUDA agreement, full-graph compilation, transfer
inspection, and gradient checks before Phase 4 can freeze or collect timing.
