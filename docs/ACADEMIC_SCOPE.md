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

Phase C found direct prior art for arbitrary-order symbolic finite-difference
WENO generation (OpenSBLI and PyWENO/PyClaw), differentiable JAX WENO CFD
(JAX-Fluids and JAX-Shock), and arbitrary-order differentiable PyTorch
finite-volume WENO (HOPE). The focused question is therefore empirical rather
than a method-invention claim:

> How accurately, differentiably, and efficiently can one exact-generated
> Jiang--Shu finite-difference WENO implementation execute as maintainable
> ordinary PyTorch from orders 5 through 15, relative to mathematically
> matched CPU, compiler-generated, and native-GPU baselines?

This question does not presume a positive answer to every performance
comparison. Phase C narrows but does not prove novelty or publishability; an
external prior-art audit remains part of the release gate. See
`LITERATURE_REVIEW_PHASE_C_RESULTS.md`.

FD/FV Phase 1 additionally found that direct comparisons already exist and
that their conclusions depend on formulation class, grid, dimension, and
capability. The permitted extension is a conditional phase diagram under the
frozen `FD_FV_EXPERIMENTAL_CONSTITUTION.md`, not an FD-superiority claim. The
first comparison remains structured Cartesian WENO-JS5; arbitrary order and
automatic discretization selection are later gates.

## Evidence already established

The current repository has reproducible evidence for:

- exact-rational generation of candidate reconstruction coefficients,
  positive optimal weights, and Jiang--Shu smoothness matrices;
- scalar periodic WENO-JS qualification for orders 5--15;
- one periodic 3-D face-frozen Roe-characteristic Euler path for orders 5--15;
- smooth convergence, conservation, device agreement, differentiation, and
  full-graph compilation on the qualified paths;
- one-dimensional periodic/transmissive Euler boundaries, exact Sod
  refinement, and independent-reference Shu--Osher qualification for
  representative WENO-JS orders 5, 11, and 15;
- an explicitly characterized WENO-JS critical-point order loss;
- a fixed WENO-5 Fortran/PyTorch/DVEB deployment bakeoff; and
- a device-resident DVEB artifact that establishes a strong native comparison
  point for one float32 3-D workload; and
- an exhaustive scalar binary32/binary64 WENO-JS precision search whose
  passing indicator/weight-formation split produced order-dependent
  `1.838x--7.058x` warm compiled speedups on the local RTX 5070 Ti while
  retaining binary64 normalization and face-flux arithmetic.

These are bounded results, not yet a complete paper package.

## Required academic work

Before a paper claim is frozen, GradFlow needs:

1. **Systematic prior-art review.** Completed in Phase C. Keep the record
   current and obtain an external subject-matter audit before paper freeze.
2. **Boundary and discontinuity qualification.** Add independently checked
   nonperiodic Euler boundary behavior and standard shock problems; periodic
   smooth tests alone are insufficient.
3. **Numerical-limit characterization.** Extend the completed scalar
   mixed-precision seam into characteristic Euler, and continue conditioning,
   roundoff, epsilon-sensitivity, critical-point, and failure analysis as order
   rises.
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

1. Under the completed FD/FV Phase-1 constitution, freeze the mathematical
   contract and independent oracles for a scalar FV WENO-JS5 seed. **Completed
   in Phase 2.**
2. Implement and qualify that seed for convergence, conservation, discontinuities, device
   agreement, compilation, and gradients before collecting performance data.
   **Implemented in Phase 3, but the frozen qualification failed two gates;
   resolve those gates prospectively before timing.**
3. Freeze and execute the scalar matched-component and best-practical FD/FV
   accuracy-to-time/memory matrix.
4. Extend qualification and the frozen comparison to ideal-gas Euler, then
   reproduce the result on a second machine and make a value-of-information
   decision for data-center FP64 hardware.
5. Only after WENO-JS5 conclusions stabilize, extend the comparison across
   generated order and one independently checkable differentiated task.
6. Obtain an external numerical-CFD prior-art audit and assemble the paper
   artifact.

The former first item, Phase B of the one-dimensional Euler boundary/shock
trunk, passed its frozen gate at source commit `3b64b1a`; see
`EULER_BOUNDARY_SHOCK_PHASE_B_RESULTS.md`.

Commercial equation-library and UI implementation remains behind this list.
