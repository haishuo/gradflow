# Academic U4-A external-baseline compatibility protocol

Status: **frozen before external timing**.

Date: 2026-08-30 (UTC)

## Purpose

U4-A decides which external systems may enter the first GradFlow paper's
operator-level performance comparison. It is a compatibility audit, not a
benchmark campaign. No runtime observed while installing, generating, or
inspecting an external system is a paper result.

The question is deliberately narrower than whether two projects both use the
name WENO:

> Can an independently maintained external implementation execute the same
> finite-difference Jiang--Shu operator, with the same inputs and measured
> endpoint, without replacing its defining numerical machinery?

## Comparison classes

Every candidate receives exactly one class.

1. `matched_operator_candidate`: its native mathematical machinery can
   express the frozen GradFlow operator. A disclosed adapter may expose
   configuration, state injection, output extraction, or timing boundaries.
2. `building_block_only`: it can generate or execute relevant reconstruction
   pieces but is not a complete matched semidiscrete operator.
3. `application_context_only`: it is a scientifically relevant CFD system,
   but its PDE discretization or solver endpoint is materially different.
4. `excluded`: it cannot provide useful independent evidence for the frozen
   question.

Only the first class may enter a table labeled as a direct performance
comparison. The other classes remain important prior art and may motivate a
separate time-to-validated-solution study.

## Frozen direct-comparison subject

The first target is the scalar periodic operator because it avoids conflating
the WENO representation with different Euler eigensystems:

- finite-difference WENO-JS of order 5, with orders 7--15 admitted only after
  the order-5 lane passes;
- exact GradFlow candidate polynomials, optimal weights, and Jiang--Shu
  smoothness quadratic forms;
- unique periodic nodes on `[0,1)`;
- positive scalar linear advection, `f(u)=u`;
- global Lax--Friedrichs `alpha=1` supplied explicitly;
- GradFlow's 12-scaled smoothness convention and scalar `epsilon=1e-29`;
- nonlinear power two;
- one semidiscrete RHS, with no Runge--Kutta stage, I/O, allocation of the
  persistent state, or host/device transfer inside the resident endpoint; and
- native float64 first, followed by float32 only as a separately qualified
  hardware lane.

Multiplying every smoothness indicator and epsilon by the same positive
constant leaves normalized JS weights unchanged. An external implementation
using standard, unscaled indicators may therefore use `epsilon=1e-29/12`.
That transformation must be explicit; silently retaining another epsilon is
not a match.

The existing characteristic Euler contract is not the initial external
target. It has duplicated periodic endpoints, face-frozen Roe projection,
per-line global characteristic speeds enlarged by 1.1, and a different
epsilon policy. A system comparison may use it only after separately matching
all of those choices.

## Admission matrix

Before timing, every candidate must have a machine-readable declaration for:

- finite-difference versus finite-volume semantics;
- WENO family and order;
- candidate-polynomial, optimal-weight, and smoothness-indicator algebra;
- smoothness scaling, epsilon, and nonlinear power;
- flux splitting and alpha policy;
- grid locations and periodic endpoint convention;
- boundary/halo treatment;
- scalar or characteristic reconstruction and, for systems, the eigensystem
  averaging policy;
- dtype and storage layout;
- spatial endpoint: reconstruction, numerical flux, semidiscrete RHS, RK
  stage, full step, or complete solve;
- state residency, transfers, allocation, process startup, generation,
  compilation, and I/O relative to each clock; and
- source revision, local adapter revision, compiler, dependencies, and
  license.

Unknown is not a match.

## Permitted adapter

An adapter may:

- choose existing options;
- expose an otherwise hard-coded numerical constant;
- provide the frozen initial state and periodic halos;
- isolate the semidiscrete RHS endpoint;
- return the result in a comparison-friendly format; and
- add synchronization and repeated-observation instrumentation.

It may not replace the candidate reconstruction, nonlinear weights,
smoothness construction, flux splitting, divergence, or generated execution
backend with GradFlow code. Every adaptation must be retained as a patch with
a SHA-256 hash and reported as `adapted`, never `stock`.

## Correctness gate before performance

For deterministic nonconstant, constant, and smooth periodic states, an
external RHS must:

- be finite;
- conserve to the dtype-specific roundoff bound;
- have the same sign, shape, point ordering, and periodic convention;
- agree with the canonical float64 RHS at `rtol=0`, `atol=2e-12` for the
  qualification cases; and
- reproduce the expected smooth convergence order on the frozen smooth
  problem.

The pointwise tolerance is a prospective qualification bound, not permission
to tune the implementation toward GradFlow output. A failed lane is retained
as a correctness exclusion and receives no speedup.

## Timing contract reserved for U4-B

U4-A records no comparative runtime. A later U4-B must report, separately:

- resident warm execution;
- process launch-to-answer with already built artifacts;
- build/generation/compilation cost;
- transfer-inclusive execution where meaningful;
- retained raw observations, median, minimum, maximum, and dispersion;
- peak memory; and
- exact hardware and software identity.

Warm and launch-to-answer claims may not be combined. Ahead-of-time build
cost is outside the prepared launch clock but remains reported.

## Stop condition

U4-A closes when the leading candidates have source-pinned compatibility
records, each has a declared comparison class, the first implementation path
for U4-B is selected, an offline verifier checks the evidence, and no external
timing has been interpreted as a result.

Do not push without explicit authorization.
