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
  retaining binary64 normalization and face-flux arithmetic;
- a completed characteristic-Euler transfer test showing that the scalar
  mixed-precision seam does not satisfy the tight order-7--15 contract;
- a completed FD/FV Euler WENO-JS5 study through qualified compiler-free
  prepared AOT process entry; and
- a closed G0--G6 reckless-to-correct native-GPU study that isolates an
  approximately twofold WENO-5 face-ownership schedule effect and records
  negative shared-pencil and occupancy interventions; and
- an ordinary-PyTorch scalar replication in which logical face-once ownership
  won every valid 3-D WENO-JS5/15 eager and compiled endpoint, while
  extreme-resolution 1-D compiled cases were excluded for correctness; and
- the completed A2 arbitrary-order performance matrix, including scalar and
  characteristic CPU/CUDA eager/compiled endpoints, fixed-shape prepared AOT,
  prepared- and isolated-cache process entry, memory, compilation, and
  correctness-exclusion records; and
- an independently validated order-11 inverse-advection use in which autograd
  recovered an analytic speed consistently with centered finite differences
  and a derivative-free minimizer, including CPU/CUDA compiler and execution
  costs.

These are bounded results, not yet a complete paper package.

## Required academic work

Before a paper claim is frozen, GradFlow needs:

1. **Systematic prior-art review.** Completed in Phase C. Keep the record
   current and obtain an external subject-matter audit before paper freeze.
2. **Boundary and discontinuity qualification.** Completed for the bounded
   one-dimensional ideal-gas Euler scope, including transmissive boundaries,
   Sod refinement, and independent-reference Shu--Osher checks at
   representative orders. Broader geometries are not a first-paper gate.
3. **Numerical-limit characterization.** The characteristic-Euler
   mixed-precision transfer and critical-point studies are complete, including
   their negative results. A1 has now consolidated coefficient conditioning,
   roundoff floors, epsilon sensitivity, and order-dependent failure records
   without beginning another precision-policy search.
4. **One genuine differentiation use.** Completed in A3 with analytic
   advection observations, centered finite-difference gradient validation, and
   an independent derivative-free parameter recovery.
5. **Arbitrary-order performance matrix.** Completed in A2 across order,
   dimension, size, precision, residency, prepared/isolated cache, warm, AOT,
   memory, compilation, and failure endpoints.
6. **Independent reference package.** Preservation is substantially complete;
   consolidate the oracle inputs, hashes, generators, exact solutions, and
   result records required by the final paper claims into one release index.
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

The long FD/FV and GPU-native branches have reached their declared stop
conditions, and A1--A3 are complete. The remaining first-paper sequence is
now:

1. reproduce the primary result on a second suitable machine, obtain an
   external numerical-CFD/prior-art audit, and freeze the citable artifact.

No additional hand-written CUDA schedule, FD/FV phase, DVEB feature, or
higher-order mixed-precision rescue is required for this sequence. Data-center
FP64 rental follows a value-of-information decision after the local matrix;
it is not assumed mandatory in advance. The detailed gate definitions and
explicit deferrals are in `ACADEMIC_COMPLETION_ROADMAP.md`.

Commercial equation-library and UI implementation remains behind this list.
