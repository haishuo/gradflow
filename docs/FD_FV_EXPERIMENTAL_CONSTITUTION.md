# FD/FV experimental constitution

Status: frozen by FD/FV Phase 1 on 2026-08-27. This document governs future
GradFlow finite-difference/finite-volume comparisons. Changing a constitutional
rule requires a named amendment, rationale, and new commit before affected
results are collected. Phase-specific numerical values belong in preregistered
protocols, not silent edits here.

## Governing law

Correctness > performance > convenience.

- A faster method that fails its mathematical contract is not a performance
  result; it is a failed configuration.
- A convenience layer may not hide a material precision, formulation, boundary,
  or execution change.
- Performance work begins only after the compared configurations pass their
  declared correctness gates.
- Negative results, failures, unsupported cases, and memory exhaustion are data.

The purpose is not to make finite difference win. It is to determine where each
qualified formulation is preferable under explicit requirements.

## Question and permitted conclusion

The experiment asks:

> For a declared PDE, geometry, accuracy target, robustness contract, precision,
> hardware, and execution boundary, when does a qualified finite-difference or
> finite-volume WENO formulation minimize time or memory, and why?

The target result is a conditional phase diagram. A universal “FD is better” or
“FV is better” conclusion is outside scope. The literature already shows that
dimension, grid regularity, Mach regime, FV class, numerical flux, and legitimate
method-specific optimizations can reverse the useful conclusion.

## Formulation registry

Every executable must declare one of these normalized classes in result
provenance:

1. **Classical conservative FD flux reconstruction.** Pointwise state; split
   physical flux reconstruction; conservative difference of interface fluxes.
2. **Alternative FD state interpolation (AFD).** Pointwise state reconstructed
   to interfaces; monotone/approximate-Riemann numerical flux; any required
   high-order flux-derivative corrections.
3. **Dimension-by-dimension FV.** Cell averages; one-dimensional face-state
   reconstruction along each direction; numerical flux; with any average/point
   transformations declared.
4. **Genuinely multidimensional FV.** Multidimensional cell polynomial and face
   quadrature, including any reduction of quadrature or Riemann calls.

“WENO-5,” “finite difference,” and “finite volume” are insufficient identifiers.
The registry must also record WENO family, order, epsilon/scaling, variables,
characteristic policy, flux split or Riemann solver, time integrator, CFL,
boundary treatment, and precision by numerical stage.

The seed matched-component study uses classical conservative FD WENO-JS5 and a
clearly classified FV WENO-JS5. AFD is a separate later lane because matching
its Riemann flux to FV can isolate a useful structural comparison, but it must
not be relabeled as the classical GradFlow FD formulation. Dimension-by-
dimension and genuinely multidimensional FV results are never pooled.

## Two mandatory comparison lanes

### Matched-component lane

Shared choices are held identical wherever the mathematics permits: PDE,
domain, physical initial condition, final physical time, boundary semantics,
nominal WENO family/order, equation of state, numerical-flux family, time-
integration order, CFL policy, precision, and convergence criterion.

Differences required by the discrete state—point values versus cell averages,
for example—are preserved and documented. A false algebraic match is not
preferred over mathematically correct formulations.

### Best-practical lane

Each method may use legitimate, published or independently justified
optimizations: characteristic variables, efficient average-to-point
transformations, reduced Riemann calls, appropriate fluxes, compiler modes,
and backend selection. Both implementations must receive comparable engineering
effort and an explicit maturity audit. Every asymmetric optimization is listed.

The two lanes answer different questions and are always reported separately.

## Continuous problem and discrete initialization

The comparison starts from one continuous mathematical problem. It does not
copy one method's discrete array into the other.

- FD receives the continuous initial state evaluated at its declared nodes.
- FV receives cell averages of that same state, evaluated analytically when
  possible or with a quadrature rule whose independent error is negligible
  relative to the requested solution error.
- Error is evaluated against the same continuous exact solution or independent
  reference. Point samples and cell averages use their appropriate projection.
- A conversion between states is itself an operator with measured error and
  cost; it may not be hidden in setup.

The seed domain is uniform, structured, periodic Cartesian geometry. Scalar
conservation laws precede ideal-gas Euler. This isolates the formulation before
geometry, meshing, turbulence, and viscous-model complications are introduced.

## Correctness gates

No timed cell enters a performance figure until it passes all applicable gates:

1. coefficient and stencil agreement with independent mathematics;
2. smooth spatial and complete-solve convergence at the expected observed order;
3. conservation to a declared roundoff/discretization tolerance;
4. agreement with exact or independent high-accuracy references;
5. declared shock location, oscillation, positivity, and failure checks;
6. CPU/device and eager/compiled agreement under the reference precision;
7. no hidden host/device transfers in a resident numerical loop; and
8. for differentiated results, primal parity plus independently checked
   directional derivatives or parameter sensitivities.

FP64 is the qualification reference. FP32 and mixed precision are performance
configurations that require their own parity gates. Consumer-GPU FP64 results
are hardware-specific observations and must mention the device's FP64 rate;
they do not establish an algorithmic GPU ceiling.

At discontinuities, classical derivative tests may be undefined or unstable.
The protocol must declare the differentiability domain and treat gradient
failure separately from primal failure.

## Outcomes and fairness axes

The primary cross-formulation outcomes are:

- achieved-error versus complete-solve time to the same final physical time;
- achieved-error versus warm device-resident step time;
- achieved-error versus peak memory;
- conservation, positivity/physical admissibility, and robustness; and
- primal and gradient error for the differentiated experiment.

Equal-grid, equal-cell, kernel-only, or nominal-order timing is secondary. It is
useful for causal analysis but cannot substitute for accuracy-to-time because
FD values and FV averages are different discrete states and may need different
resolutions for the same error.

Each result carries five audits:

- **mathematical:** PDE, formulation, order, flux, time integration, boundary,
  CFL, precision;
- **resource:** cells, stored degrees of freedom, bytes, device, CPU cores and
  threads;
- **outcome:** achieved error, conservation, shock/positivity result, final
  physical time;
- **execution:** kernel/step/solve, residency, transfers, compilation,
  preparation, cold/warm state; and
- **capability:** geometry, mesh regularity, adaptivity, physics, and permitted
  method-specific features.

## Timing boundaries

Every performance campaign reports, where applicable:

1. **cold complete solve:** process entry or first library call through host-
   visible answer, including compilation and transfers;
2. **prepared complete solve:** reusable binaries/caches prepared ahead of the
   timed invocation, but input/output transfers included;
3. **warm complete solve:** compiled runtime in the same process through the
   final answer;
4. **device-resident solve/step:** inputs and outputs remain on the device; and
5. **kernel/operator:** a diagnostic lower boundary, never the sole headline.

Synchronization is explicit. Warmups and repetitions are frozen before the
campaign. CPU thread affinity/count, GPU clocks or power policy when observable,
compiler versions, caches, and environment hashes are recorded. Compilation may
be excluded only from a named AOT/prepared endpoint; it remains part of cold
latency. Failed allocations and compile failures remain in the record.

## Initial campaign boundary

The first implementation campaign is deliberately small:

- uniform structured Cartesian grids;
- scalar advection and scalar nonlinear conservation law, then ideal-gas Euler;
- periodic boundaries first, followed by separately qualified shock boundaries;
- WENO-JS5 before any order sweep;
- FP64 correctness before FP32/mixed-precision performance;
- one CPU implementation and the local RTX GPU through ordinary maintainable
  source, with native/generated ceilings used only when mathematically matched;
- spatial-operator characterization and complete method-of-lines solves; and
- one bounded sensitivity/inverse task only after primal parity.

Excluded from this initial campaign are Navier–Stokes, turbulence models,
unstructured grids, production AMR, moving meshes, multiphase flow, wing
geometry, real-time claims, and product UI. Their importance is not disputed;
they would prevent the first causal comparison from being interpretable.

## Hardware escalation

The local CPU and RTX 5070 Ti are the development and first-report machines. A
data-center FP64 GPU is rented only after a value-of-information decision frozen
from local results. The decision must name the unresolved claim, required
precision, smallest sufficient matrix, estimated cost, stopping rule, and why
the result cannot be inferred locally. Hardware is never silently substituted.

## Staged program

1. **Phase 1 — complete:** literature, taxonomy, claim boundaries, constitution.
2. **Phase 2 — mathematical contracts:** derive the seed FV-JS5 formulation,
   point/cell-average projections, independent oracles, and invariant schemas.
3. **Phase 3 — scalar qualification:** implement only the seed scalar FV path;
   convergence, conservation, discontinuity, device, compile, and gradient
   gates; no broad benchmark.
4. **Phase 4 — scalar matched bakeoff:** freeze sizes/endpoints, then measure
   1-D/2-D/3-D and accuracy-to-time/memory in both lanes.
5. **Phase 5 — Euler qualification:** characteristic/component policies,
   positivity/failure contracts, multidimensional smooth and shock references.
6. **Phase 6 — Euler bakeoff and replication:** execute the frozen modern matrix,
   include a feasible external close-system baseline, and reproduce on a second
   machine. Decide data-center GPU value of information.
7. **Phase 7 — order and differentiation extension:** only after JS5 conclusions
   are stable, extend matched generated orders and the independently checkable
   gradient task. This phase may be split if either question becomes large.
8. **Paper freeze:** external CFD review, immutable artifacts, claim table,
   negative results, release/licensing audit, and citable environment.

Each phase needs its own preregistered protocol and stop condition. Later phases
cannot retroactively change earlier thresholds after seeing performance data.

## Method selection and product provenance

A future automatic selector may choose FD or FV only under an explicit user or
application contract for accuracy, conservation, robustness, geometry, memory,
and latency. It may hide routine machinery from an end user, but it must record
the chosen formulation and evidence version in machine-readable provenance.
Backend selection among equivalent implementations is not the same as changing
the discretization. A user override may request a qualified method/backend; an
unqualified mathematical configuration is rejected rather than silently run.

Automatic discretization selection remains outside the first paper and requires
a separate review of autotuning and algorithm-selection literature.

## Amendment rule

A constitutional amendment must:

1. be committed before affected experiments;
2. identify the rule and scientific reason;
3. state whether previous results remain comparable;
4. allocate a new protocol/result-series identity when comparability breaks;
5. never weaken an already-observed gate merely to admit a preferred result.

## Amendment 1 — accelerator visibility vocabulary (2026-08-28)

Future experiments follow `docs/EXECUTION_INFRASTRUCTURE_ADMISSION.md` and
separate physical host inventory from execution-context device visibility.
The bare status `untested_unavailable` is retired for new records in favor of
the explicit statuses defined there. This corrects ambiguous infrastructure
language after Forge's RTX 5070 Ti was found to be hidden by the default
process sandbox. It does not alter any historical record, numerical result, or
comparison; linked supplements remain the evidence for the previously deferred
CUDA strata.

## Amendment 2 — nonlinear scalar and Euler phase labels (2026-08-28)

The original staged program described Euler qualification as Phase 5 and its
bakeoff as Phase 6. The executed Phase-5A--5C sequence instead inserted a
necessary nonlinear scalar Burgers qualification and performance boundary
after the earlier linear study. Those immutable labels are retained.

The Euler program therefore begins at Phase 6: Phase 6A freezes contracts and
oracles, Phase 6B qualifies the matched Euler formulations, and Phase 6C may
time them only after qualification. Multidimensional extension, replication,
external baselines, and hardware escalation follow as separately frozen Phase
6 subphases. This is a scheduling clarification only. It changes no prior
mathematics, threshold, measurement, or conclusion.
