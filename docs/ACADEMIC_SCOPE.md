# Academic GradFlow scope

## Sequence decision

GradFlow will pursue the bounded academic result before the larger commercial
product. This is sequencing, not abandonment of the product architecture.
Canonical work must remain compatible with the engineering charter and target
problem model, but the paper does not need a general PDE catalog, guided UI,
or production aerospace workflow.

The academic result must stand on GradFlow's own readable scientific source,
mathematics, tests, and reproducible records. It must remain coherent if DVEB
is unavailable.

## Candidate paper question

The focused question is:

> Can exact-generated arbitrary-order finite-difference WENO-JS be expressed
> as maintainable, differentiable ordinary PyTorch; extended from scalar
> reconstruction to characteristic Euler systems through WENO-15; and
> executed competitively across CPU and GPU regimes without bespoke kernels in
> the canonical numerical implementation?

The final wording and novelty claim remain subject to a systematic literature
review. This question does not presume a positive answer to every performance
comparison.

## Evidence already established

The current repository has reproducible evidence for:

- exact-rational generation of candidate reconstruction coefficients,
  positive optimal weights, and Jiang--Shu smoothness matrices;
- scalar periodic WENO-JS qualification for orders 5--15;
- one periodic 3-D face-frozen Roe-characteristic Euler path for orders 5--15;
- smooth convergence, conservation, device agreement, differentiation, and
  full-graph compilation on the qualified paths;
- an explicitly characterized WENO-JS critical-point order loss;
- a fixed WENO-5 Fortran/PyTorch/DVEB deployment bakeoff; and
- a device-resident DVEB artifact that establishes a strong native comparison
  point for one float32 3-D workload.

These are bounded results, not yet a complete paper package.

## Required academic work

Before a paper claim is frozen, GradFlow needs:

1. **Systematic prior-art review.** Record databases, queries, dates,
   inclusion criteria, close systems, and the exact claim intersection.
2. **Boundary and discontinuity qualification.** Add independently checked
   nonperiodic Euler boundary behavior and standard shock problems; periodic
   smooth tests alone are insufficient.
3. **Numerical-limit characterization.** Study conditioning, roundoff,
   epsilon sensitivity, critical points, and failure behavior as order rises.
4. **One genuine differentiation use.** Demonstrate a bounded inverse or
   sensitivity problem with an independently checkable target and gradient
   validation.
5. **Arbitrary-order performance matrix.** Compare mathematically identical
   implementations across order, dimension, size, precision, residency, and
   cold/warm/AOT endpoints. Report memory and failures as well as speed.
6. **Independent reference package.** Preserve oracle inputs, hashes,
   generators, exact solutions, and result records required to rerun the
   numerical claims.
7. **Paper artifact and release review.** Resolve redistribution/license
   questions, freeze environments, run clean reproductions, and archive a
   citable release.

The order remains correctness before performance. Literature work may proceed
in parallel because it changes claim scope rather than numerical execution.

## Academic release gate

An academic release candidate requires:

- every claimed formulation to have a frozen mathematical contract;
- independent correctness evidence for smooth, critical, boundary, and shock
  behavior;
- all reported performance points to pass parity first;
- a claim table separating established, observed, inferred, and untested
  statements;
- scripts that reproduce figures and tables from immutable records;
- a clean source revision and environment manifests;
- clear negative results and hardware limitations; and
- redistribution status for every bundled reference and artifact.

DVEB may appear as a comparator or optional backend. Neither its availability
nor a favorable result is an academic-release requirement.

## Explicit commercial deferrals

The focused academic version does not require:

- a general equation catalog or arbitrary symbolic PDE entry;
- compressible Navier--Stokes;
- wing geometry, meshing, turbulence modeling, or complete aerospace CFD;
- schema-generated guided UI;
- a production automatic backend planner;
- real-time claims;
- universal superiority over existing solvers; or
- qualification on every CPU, GPU, or Apple device.

Work on these items resumes after the academic core is defensible. Design
decisions made now must avoid needlessly obstructing them, but speculative
product framework work must not delay the bounded paper.

## Immediate trunk order

1. Freeze and execute the 1-D Euler boundary/shock correctness trunk.
2. Complete the systematic literature and claim matrix.
3. Characterize high-order numerical limits and select any additional WENO
   variants only in response to evidence.
4. Add and validate one differentiable inverse/sensitivity experiment.
5. Freeze and execute the arbitrary-order performance campaign.
6. Assemble the paper artifact, then decide whether a data-center float64 GPU
   addendum is worth its cost.

Commercial equation-library and UI implementation remains behind this list.
