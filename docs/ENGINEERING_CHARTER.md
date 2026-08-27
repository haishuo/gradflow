# GradFlow engineering charter

## Status and authority

This document governs GradFlow development. It applies to research code,
canonical library code, generated artifacts, native backends, benchmarks,
Python interfaces, and any future user interface.

When goals conflict, the following ordering is law:

> **Correctness > performance > convenience.**

This is a strict precedence relation, not a list of equally weighted goals.
A lower-ranked benefit cannot justify damage to a higher-ranked property.

- If a convenience makes a qualified execution path slower, do not adopt it
  on that path.
- If an optimization changes the requested mathematics or makes the numerical
  result wrong, do not adopt it.
- If a backend is fast but cannot honor the complete problem contract, it is
  ineligible.
- If the evidence is insufficient, report the limitation or refuse the
  request rather than silently guessing.

The applicable numerical and performance contracts must be defined before a
change is judged. Floating-point correctness means agreement with specified
mathematics under a documented, independently justified tolerance; it does
not mean that every backend must be bitwise identical. Performance means the
declared end-to-end endpoint on declared hardware, not a selectively timed
kernel unless kernel time is itself the declared endpoint.

## Mission

GradFlow has two related but distinct goals.

The research goal is to determine whether a direct, maintainable PyTorch
system can construct, verify, differentiate, and efficiently execute
arbitrary-order finite-difference WENO schemes, including realistic
characteristic-system and WENO-15 cases, without requiring bespoke CUDA or
Triton engineering in the canonical scientific source.

The product goal is one scientific engine in which a user can select a
qualified equation family, physical parameters, domain, initial and boundary
conditions, accuracy policy, and requested outputs without needing to manage
compiler or device machinery. A Python library is the foundational product
surface; a guided application may present the same model later.

The academic work explains and tests the machinery. The product makes the
qualified machinery usable. Product requirements do not determine academic
novelty, and a publishable experiment is not automatically a usable product.
They share numerical source, validation, and provenance rather than becoming
two implementations.

## Gate 1: correctness

Correctness is an admission requirement for every canonical feature. A
numerical capability is not eligible for performance claims or product
promotion until it has, as applicable:

1. a precise mathematical specification, including convention choices;
2. an independent reference, oracle, manufactured solution, or exact
   property against which it can be checked;
3. a declared precision policy and justified tolerances;
4. convergence evidence at the expected rate;
5. conservation, invariant, symmetry, and admissibility checks;
6. boundary and endpoint tests, not only interior periodic tests;
7. cross-device and cross-backend agreement;
8. differentiation tests when gradients are claimed;
9. explicit handling of invalid and unsupported states; and
10. a reproducible qualification record tied to source and environment
    identities.

No implementation may silently change equation, WENO family or order, flux
splitting, boundary treatment, time integrator, dtype, device residency,
differentiability, or requested output to make a request run. A fallback is
legal only when it implements the same scientific contract. Otherwise the
request must fail clearly.

Plots that look plausible are not correctness evidence. Agreement among
implementations derived from the same algebra is useful but is not fully
independent validation.

## Gate 2: performance

After correctness, a capability must establish a useful performance envelope.
GradFlow need not win at every grid size, dimension, dtype, or endpoint, but it
must not add machinery that is broadly dominated without a separately stated
research or portability purpose.

Performance studies must:

1. compare mathematically equivalent workloads;
2. declare hardware, software, precision, shapes, memory layout, and output
   requirements;
3. separate preparation, cold invocation, warm invocation, and resident
   execution;
4. charge runtime compilation and required transfers to an endpoint when the
   user pays those costs;
5. exclude ahead-of-time preparation from prepared invocation time while
   still reporting its cost and artifact assumptions;
6. synchronize devices correctly and report the statistic and repetition
   count;
7. include established relevant comparators when they can be run fairly;
8. record memory limits, failures, and selector regret as well as wins; and
9. bind automatic placement decisions to measured, versioned evidence.

An optimization begins a new correctness obligation. It must pass the
scientific gate before its timing can influence a decision. Fast-math modes,
precision changes, approximations, and altered stopping conditions are new
contracts, not invisible optimizations.

Consumer-GPU float64 results must be described in the context of that
hardware's float64 capability. They may establish correctness and a local
performance result, but they cannot substitute for a future data-center-GPU
study when making general float64 claims.

## Gate 3: convenience

Convenience is pursued only on top of correct, competitively executed
capabilities. It must not introduce a regression in the applicable qualified
execution path.

The ordinary user should describe the physical problem and requested result,
not choose tensor layouts, compiler modes, transfer schedules, or launch
geometry. Expert users may request a backend or numerical policy explicitly,
even when it is slower, but an override bypasses only automatic performance
selection. It never bypasses mathematical, admissibility, or memory-safety
validation.

A user interface may hide machinery, but it may not hide uncertainty or alter
semantics. Defaults must be documented and reproducible. Every successful
guided run must be expressible through the same public problem model used by
Python, and its resolved configuration must be inspectable.

The intended product user may be numerics-agnostic, but cannot be made
physics-agnostic by interface design. GradFlow can explain required physical
choices, supply qualified presets, validate units and ranges, and prevent
incompatible combinations. It cannot make an ill-posed physical problem
meaningful.

## Architecture law

The durable dependency direction is:

```text
Python API or guided UI
          |
          v
validated scientific problem model
          |
          v
equation and discretization construction
          |
          v
evidence-bound execution planning
          |
          v
qualified CPU / PyTorch / AOT / DVEB / future backends
          |
          v
results, diagnostics, and provenance
```

The scientific problem model is backend-neutral. Backends implement that
model; they do not define it. The UI and Python interface are two
presentations of the same validated model. Numerical results and provenance
come from the engine, not from duplicated UI calculations.

Equation names, example problems, and execution targets are separate concepts.
For example, Euler is an equation family, the Sod shock tube is a scenario,
and CPU SIMD or CUDA is an execution target. Their separation prevents a
different solver product from emerging for every benchmark or backend.

## Research code, canonical code, and generated artifacts

Exploratory work may be intentionally narrow or disposable, but it must live
under `experiments/` or `legacy/` and must not masquerade as the public API.
Its limitations and relationship to canonical mathematics must be recorded.

Promotion into `src/gradflow/` requires:

1. a stable mathematical contract;
2. the relevant correctness gate;
3. readable scientific source or a reproducible generator;
4. documented public behavior and failure modes;
5. no hidden transfers, conversions, or mathematical substitutions;
6. tests at the public boundary;
7. performance evidence appropriate to the feature's role; and
8. provenance sufficient to reproduce the decision.

Generated C++, CUDA, or packaged compiler output may be a deployment artifact
or optional backend. It is not the canonical mathematical specification.
Artifacts must be versioned and tied to their generator, source inputs,
compiler environment, and qualified problem signature.

## Technical-debt policy

GradFlow does not use “we will fix it later” as an unrecorded development
strategy. A known compromise may be accepted only when it is:

- necessary to answer a bounded research question or unblock a higher-ranked
  requirement;
- isolated from the canonical contract where possible;
- documented with its consequence and affected surface;
- protected by tests that prevent it from expanding silently;
- assigned an objective removal or repayment condition; and
- reviewed before dependent feature work broadens it.

Debt that threatens correctness blocks release and further dependent work.
Debt that threatens competitiveness blocks performance qualification. Debt
that affects convenience may remain only when it does not leak into or
degrade higher-ranked contracts.

Avoiding debt does not mean building a speculative framework. A general
abstraction should normally be extracted only after at least two concrete
uses demonstrate the shared contract. The current Euler vertical slice is
evidence for one use; compressible Navier--Stokes is expected to be the first
major test of whether an equation-level abstraction is genuine.

## Development cadence

Deliberate development is acceptable; unmeasured drift is not. Work should be
organized into bounded trunks with an explicit question, frozen acceptance
gate, preserved evidence, and a clear stop condition. Each completed trunk
must leave at least one durable result: qualified code, a reproducible negative
result, a clarified contract, or retired debt.

Feature count is not the measure of progress. A smaller capability that is
correct, characterized, and maintainable is preferable to a broad surface
whose behavior cannot be defended. At the same time, architectural discussion
must eventually be tested by concrete implementations; documentation alone
does not qualify a capability.

## Decision and claim discipline

Material numerical, API, backend, or measurement decisions must be preserved
in documentation, tests, or versioned records. A result must state what was
tested, what passed, what failed, and what remains untested.

The project will not claim generality from a single equation, boundary,
machine, or precision; real-time capability from a kernel benchmark; absence
of prior art from a preliminary search; or product readiness from a research
prototype. Negative results remain evidence and are preserved.

## Current application

Today the canonical package has a qualified scalar arbitrary-order WENO-JS
seed and one narrow 3-D characteristic Euler path for periodic duplicated
endpoints. It does not yet have a general equation library, Navier--Stokes,
general boundary closures, complex geometry, an automatic resident-backend
planner, or a guided UI.

Those omissions are explicit scope boundaries, not permissions to represent
future interfaces as implemented. GradFlow will complete the bounded academic
program in `ACADEMIC_SCOPE.md` before resuming the broader commercial build.
The immediate numerical work is the correctness-first Euler boundary/shock
trunk. Later commercial work will use compressible Navier--Stokes as the first
major test of the shared equation abstraction.
