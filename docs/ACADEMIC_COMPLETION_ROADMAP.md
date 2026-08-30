# GradFlow Academic completion roadmap

Status: **active bounded completion plan**.

Date: 2026-08-30 (UTC)

## Objective

The first GradFlow academic artifact is an empirical systems-and-numerics
study of exact-generated finite-difference WENO-JS in maintainable ordinary
PyTorch. It is not the commercial product, a new WENO formula, or a claim that
PyTorch universally defeats native code.

The governing order remains:

> correctness > performance > convenience

The completed GPU-native G0--G6 investigation is now a closed supporting
study. It supplies a native WENO-5 schedule control and identifies which
hardware-first changes preserved or destroyed the requested mathematics. It
does not authorize another CUDA optimization branch before the academic core
is complete.

## Work already sufficient for the first paper

- exact-rational WENO-JS coefficient generation through order 15;
- scalar periodic and 3-D Roe-characteristic Euler qualification for orders
  5, 7, 9, 11, 13, and 15;
- smooth convergence, conservation, critical-point characterization,
  compilation, device agreement, and basic differentiation gates;
- independently checked one-dimensional Euler Sod and Shu--Osher behavior
  with periodic and transmissive boundaries;
- systematic close-prior-art review and a narrowed empirical claim;
- a completed scalar mixed-precision search and a completed, predominantly
  negative characteristic-Euler transfer test;
- endpoint-explicit FD/FV WENO-JS5 evidence through prepared AOT execution;
- fixed WENO-5 CPU, ordinary-PyTorch, AOT, DVEB, and native-GPU controls; and
- the closed reckless-to-correct GPU-native formulation study; and
- the completed A1 claim, formulation, coefficient-conditioning,
  roundoff-floor, epsilon-sensitivity, and failure-boundary freeze.

These results are inputs to a paper package, not proof that the package is
already publishable.

## Remaining mandatory gates

### A1. Freeze the final claim and numerical-limit matrix — complete

Consolidate the paper's exact formulation contracts and test only the
remaining order-dependent numerical questions needed to interpret orders
5--15: coefficient conditioning, roundoff floor, epsilon sensitivity,
critical-point behavior, and declared failure boundaries. Do not begin a new
mixed-precision search. The completed Tier-2 negative result is itself the
current characteristic mixed-precision conclusion.

Deliverables:

- one claim table separating established, observed, inferred, and untested;
- one order-by-order numerical-limit table generated from immutable records;
- prospectively frozen thresholds for any genuinely missing cases; and
- an updated literature comparison with OpenSBLI, PyWENO/PyClaw, HOPE,
  JAX-Fluids, and JAX-Shock.

Completed on 2026-08-30. All new numerical-limit executions were finite and
conservative; no canonical source or policy changed. See
`ACADEMIC_A1_RESULTS.md`, `ACADEMIC_A1_CLAIM_MATRIX.md`,
`ACADEMIC_A1_NUMERICAL_LIMITS.md`, and
`ACADEMIC_A1_PRIOR_ART_COMPARISON.md`.

### A2. Run the core arbitrary-order performance matrix — complete

This is the principal unfinished empirical experiment. Compare the same
qualified FD-WENO-JS mathematics across orders 5--15 using ordinary PyTorch
eager/compiled and feasible matched CPU, prepared-AOT, and native controls.
Every endpoint must state whether compilation, transfers, process startup,
and preparation are inside or outside the clock.

The matrix must report:

- order, dimension, grid size, dtype, and state residency;
- cold, warm, prepared-AOT, and start-to-finish endpoints where applicable;
- execution time, peak memory, compile/preparation time, and failures;
- pointwise or norm parity before every performance result; and
- hardware/software identity and raw repeated observations.

The native G-series schedule is a fixed WENO-5 comparison point. Extending it
to arbitrary order is not required. DVEB is optional and cannot block the
paper.

Completed on 2026-08-30. All 90 protocol-eligible workers completed, every
compiled worker captured one graph with zero breaks, three AOT packages
qualified, and both prepared-cache and isolated-cache deployment slices
completed. Correctness failures remain explicit exclusions. See
`ACADEMIC_A2_RESULTS.md`.

### A3. Demonstrate one independently validated differentiation use — complete

Complete one bounded sensitivity or inverse problem in which the target is
independently checkable. Validate gradients against centered finite
differences or another independent derivative construction, characterize the
step-size window and failure modes, and report both numerical and execution
costs.

This gate demonstrates why differentiability is scientifically useful. It is
not a claim that differentiable WENO is unprecedented.

Completed on 2026-08-30. GradFlow recovered an analytic linear-advection speed
through an order-11 WENO-JS solve; centered differences and a derivative-free
minimizer independently agreed, CPU/CUDA eager/compiled gradients passed, and
the large whole-solve compilation cost was retained. See
`ACADEMIC_A3_RESULTS.md`.

### A4. Replicate, audit, and freeze the artifact — local candidate complete,
external gates pending

After A1--A3 stabilize:

- reproduce the primary numerical and performance conclusions on a second
  suitable machine;
- make a value-of-information decision before renting A100/H100-class FP64
  hardware rather than treating it as automatically mandatory;
- obtain an external numerical-CFD/prior-art audit;
- resolve or explicitly flag reference redistribution questions;
- freeze environments, scripts, tables, figures, raw records, and hashes;
- create a citable release candidate and run a clean-room reproduction; and
- freeze paper wording only after those checks pass.

The local release-candidate portion completed on 2026-08-30. Tag
`academic-v0.1.0-rc1` identifies the 1,855-file payload; a clean no-hardlink
clone passed 293 tests with 72 declared skips and all A1--A4 offline verifiers.
Environment, rights, second-machine, and external-review packets are frozen.
See `ACADEMIC_A4_RESULTS.md`.

A4 remains open: the second-machine replication and independent
numerical-CFD/prior-art audit have not occurred. Reference and project-license
questions are explicitly flagged but not legally resolved.

## Explicitly deferred from the first academic artifact

The following work may be valuable later but is not required to finish the
bounded paper:

- G7, warp-distributed CUDA, further occupancy tuning, or another hand-written
  CUDA schedule;
- repairing higher-order characteristic mixed precision after the frozen
  Tier-2 failure;
- another FD/FV phase beyond the completed constitution unless the final
  paper explicitly adopts the FD/FV phase diagram as a central claim;
- additional DVEB development;
- a general PDE/equation catalog, Navier--Stokes, geometry, meshing, or
  turbulence modeling;
- automatic backend placement, UI work, or commercial deployment polish; and
- real-time aerospace or universal-performance claims.

## Stop discipline

New side experiments enter the academic critical path only if they close a
named release gate above. Interesting but nonessential questions are recorded
as future work. A negative result is complete when its frozen question has
been answered; it is not an invitation to optimize until it becomes positive.

Under this discipline, A1--A3 are closed and the remaining route is:

```text
A4 second-machine replication + independent CFD/prior-art audit
```

Design work for A3 may overlap A2, but performance remains downstream of the
relevant correctness gates.
