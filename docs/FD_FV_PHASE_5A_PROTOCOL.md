# FD/FV nonlinear Phase-5A mathematical and oracle protocol

Status: frozen before production implementation, numerical qualification, or
timing.

Freeze date: 2026-08-28 UTC.

## Identity and relationship to the existing roadmap

This is **FD/FV nonlinear Phase 5A**, the next scalar research boundary after
the completed linear Phase-4 study. The original experimental constitution
used the shorter label “Phase 5” for a later Euler qualification. This additive
name does not renumber, replace, or reinterpret any completed record. Euler
remains later work.

Phase 5A freezes one nonlinear continuous problem, the mathematically distinct
FD and FV projections, an independent pre-shock exact oracle, and the gates for
the later Phase 5B implementation. It collects no performance measurements.

## Continuous problem

On the unique periodic domain `[0,1)`, solve inviscid Burgers' equation

```text
u_t + (u^2/2)_x = 0,
u(x,0) = 1/2 + (1/5) sin(2*pi*(x-7/100)).
```

The data remain in `[0.3,0.7]`. The first classical shock time is

```text
t_shock = -1/min_x u0'(x) = 5/(2*pi) ~= 0.7957747154594768.
```

The frozen final time is `T=1/10`, approximately 12.57% of the shock time.
The characteristic map is strictly increasing by a margin
`1 - 0.04*pi > 0.874`, so the entire experiment is smooth and the classical
solution is unique. No shock-capturing conclusion follows from this phase.

## Independent exact solution

For an Eulerian coordinate `x`, the exact characteristic foot `xi` is the
unique lifted real solution of

```text
x = xi + t*u0(xi).
```

Then `u(x,t)=u0(xi)`. The independent oracle brackets the root using the known
range `[0.3,0.7]` and applies deterministic bisection. It uses only the Python
standard library and imports neither PyTorch, NumPy, GradFlow, nor any WENO
coefficient code.

For FV, exact cell averages are not approximated by center samples. Let

```text
U0(xi) = xi/2 - (1/(10*pi))*cos(2*pi*(xi-7/100)),
H_t(xi) = U0(xi) + (t/2)*u0(xi)^2.
```

Because `dx/dxi = 1+t*u0'(xi)`, the exact average over Eulerian cell `[a,b]`
is

```text
ubar(t) = (H_t(xi_b)-H_t(xi_a))/(b-a),
```

where `xi_a` and `xi_b` are the lifted characteristic feet of the two faces.
This is an independent conservation-law projection, not a quadrature through
the production solver.

## Discrete formulations in the later matched lane

The frozen pair is:

- `fd_classical_js5_burgers_global_lf_periodic_v1`: persistent nodal point
  values at `x_i=i/N`; classical conservative split-physical-flux FD WENO-JS5;
- `fv_dimensional_js5_burgers_global_lf_periodic_v1`: persistent physical cell
  averages over `[i/N,(i+1)/N]`; left/right state reconstruction followed by a
  Rusanov face flux.

Both use the physical flux `f(u)=u^2/2`, unique periodic grids, WENO-JS5,
smoothness scale `12`, epsilon `1e-29`, nonlinear power two, and SSP-RK3.
The global LF speed is the same fixed physical bound `alpha=0.7` in both
methods and every RK stage. This is exact for the frozen smooth solution's
maximum characteristic speed and isolates formulation structure. A dynamic or
method-specific alpha estimator belongs to a separately identified
best-practical lane and its cost must be included there.

FD reconstructs split physical-flux samples. FV reconstructs states and then
evaluates the two-state numerical flux. These operations do not commute for
Burgers, so unlike the linear seed the methods are genuinely nonlinear and
must not share a discrete input array.

## Time integration and correctness sizes

For `dx=1/N`, the later complete-solve qualification uses

```text
nominal_dt = 0.2 * dx^(5/3) / 0.7
steps      = ceil(T/nominal_dt)
dt         = T/steps.
```

The fixed step reaches the exact final physical time and makes third-order
temporal error asymptotically commensurate with fifth-order spatial error. The
initial qualification sequence is `N=(24,36,54,81)`. It is a correctness
sequence, not a timing matrix; a later performance protocol must freeze its
own sizes before measurement.

## Phase-5B admission gate frozen here

Before nonlinear timing, each formulation must pass:

1. exact-oracle initialization and terminal projection at its own state
   semantics;
2. constant preservation and conservation under the semidiscrete RHS and full
   solve;
3. decreasing L1 and L2 complete-solve errors over the frozen sequence;
4. expected WENO-JS behavior at smooth critical points reported explicitly,
   including a noncritical spatial-order diagnostic rather than silently
   weakening or mislabeling the known JS critical-point behavior;
5. agreement with the characteristic point/cell-average oracle at `T=0.1`;
6. float64 CPU/CUDA parity, eager/compiled parity, finite resident outputs, one
   full graph with zero breaks, and no hidden host/device transfers;
7. a fixed-step float64 directional-derivative check against an independent
   centered finite difference, away from discontinuities; and
8. no performance measurements before every applicable gate passes.

The exact numerical tolerances and critical/noncritical diagnostic masks must
be frozen in the Phase-5B protocol before running the production
implementation. They may not be selected after observing timing.

## Infrastructure correction

All device language and machine-readable records are governed by
`docs/EXECUTION_INFRASTRUCTURE_ADMISSION.md`. In particular, process-local
CUDA invisibility on Forge is `process_hidden_host_present`, not evidence that
Forge lacks a GPU. A Phase-5B CUDA stratum requires a fresh formulation-specific
admission in a device-visible process even though Forge host presence is
already established.

## Phase-5A artifacts and acceptance

Phase 5A produces a standard-library oracle, immutable contract and oracle-case
records, hashes, an independent verifier, and tests. Acceptance requires:

- the recorded shock-time and monotonicity margins agree with the analytic
  formulas;
- characteristic residuals satisfy the frozen oracle tolerance;
- the primitive-based FV projection agrees with independently converged
  composite Simpson integration;
- exact projected FV mass equals the analytic periodic mean to the frozen
  tolerance at `t=0` and `T`;
- finite-resolution FV averages demonstrably differ from center samples;
- the oracle and verifier have no production numerical dependency;
- the infrastructure status classifier covers every required status; and
- no production Burgers solver, compiler campaign, timing, WENO-order sweep,
  Euler extension, or publication claim is added.

Stop after these artifacts verify, bounded tests pass, coherent local commits
exist, and the working tree is clean. Do not push without explicit
authorization.
