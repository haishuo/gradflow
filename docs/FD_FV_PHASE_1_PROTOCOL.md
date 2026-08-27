# FD/FV Phase-1 literature and experimental-constitution protocol

Status: frozen before the formal FD/FV searches and screening.

Freeze date: 2026-08-27 UTC.

## Purpose

Phase 1 asks what controlled comparisons already establish about conservative
finite-difference and finite-volume WENO, and what modern comparison remains
scientifically defensible for GradFlow. It produces literature evidence,
mathematical taxonomy, claim boundaries, and an experimental constitution.
It implements no finite-volume solver and collects no new performance timing.

The review must not treat the presence of a finite-volume method in a
successful production or differentiable code as evidence that finite volume
is universally more accurate, robust, or efficient. It must likewise not
infer finite-difference superiority from regular-grid kernel speed alone.

## Questions

1. Which head-to-head FD/FV WENO comparisons have been published, and which
   mathematical and implementation variables did they control?
2. How do conclusions change with dimension, grid geometry, adaptivity,
   formal order, flux construction, time integration, and error target?
3. Have controlled comparisons covered modern CPUs and GPUs, compilation,
   preparation latency, memory, accuracy-to-time, and differentiation in one
   reproducible study?
4. Which bounded comparison can GradFlow execute without silently favoring
   its existing finite-difference implementation?

## Candidate statements

Each candidate receives one of the Phase-C decision statuses.

| ID | Candidate statement |
|---|---|
| F0 | Direct FD/FV WENO comparisons and comparative surveys already exist. |
| F1 | Neither FD nor FV WENO is universally superior; their applicability and cost depend materially on dimension, grid, formulation, and execution choices. |
| F2 | A controlled structured-grid accuracy-to-time study across FD/FV formulation, dimension, and order on modern CPU/GPU array compilers may remain a contribution. |
| F3 | A controlled comparison of differentiated FD/FV WENO execution and gradient reliability may remain a contribution. |
| F4 | A comparison extending generated Jiang--Shu FD/FV formulations through high odd order may remain a contribution. |
| F5 | Automatic selection of FD versus FV from empirical evidence may be a later systems contribution. |

“First,” “only,” “never compared,” and universal-superiority statements are
prohibited before and after this review unless independently established;
this bounded review cannot prove absence.

## Mathematical taxonomy

Every comparison must identify, rather than collapse, at least these classes:

1. **Classical conservative FD flux reconstruction:** pointwise state,
   reconstruction of split physical fluxes, and conservative flux difference.
2. **Alternative FD state interpolation:** pointwise state reconstructed to
   faces and coupled to a monotone or approximate-Riemann numerical flux.
3. **Dimension-by-dimension FV reconstruction:** cell averages reconstructed
   to face states along coordinate directions, followed by numerical fluxes.
4. **Genuinely multidimensional FV reconstruction:** multidimensional cell
   polynomials and face quadrature, including any method-specific reduction in
   quadrature or Riemann-solver calls.

The review records authors' terminology as well as this normalized taxonomy.
It does not reclassify a method without primary-text or source evidence.

Conservative FD is not described as nonconservative merely because it evolves
point values. FV's cell-integral semantics, geometric flexibility, and local
conservation structure remain materially distinct.

## Frozen search families

Searches use both `WENO` and “weighted essentially non-oscillatory” where a
provider permits it. Provider syntax may change quoting but not the concepts.

```text
S1  WENO AND "finite difference" AND "finite volume" AND (comparison OR versus)
S2  WENO AND "finite difference" AND "finite volume" AND (performance OR efficiency OR cost)
S3  WENO AND "finite difference" AND "finite volume" AND (accuracy OR resolution) AND (runtime OR time OR cost)
S4  WENO AND "finite difference" AND "finite volume" AND (GPU OR CUDA OR accelerator)
S5  WENO AND "finite difference" AND "finite volume" AND (multidimensional OR "three dimensional" OR Cartesian)
S6  WENO AND (adaptive OR nonuniform OR unstructured) AND ("finite difference" OR "finite volume") AND efficiency
S7  WENO AND "finite difference" AND "finite volume" AND (differentiable OR gradient OR adjoint OR PyTorch OR JAX)
S8  ("alternative finite difference WENO" OR AFD-WENO) AND "finite volume" AND comparison
S9  (JAX-Fluids OR HOPE OR OpenSBLI OR PyWENO OR PyClaw) AND ("finite difference" OR "finite volume") AND WENO
S10 WENO AND (survey OR review) AND "finite difference" AND "finite volume"
```

Repository searches add:

```text
R1  FD WENO FV WENO benchmark
R2  finite difference finite volume WENO GPU
R3  differentiable finite volume WENO
R4  differentiable finite difference WENO
R5  JAX Fluids WENO finite volume flux splitting
```

Each search record retains provider, submitted query, date, URL or endpoint
when available, status, returned count when known, and candidates screened.

## Sources and evidence hierarchy

Searches cover Crossref/OpenAlex where accessible, arXiv, publisher indexes,
NASA NTRS, official repositories/documentation, and backward/forward citation
searches from included comparisons. Scopus and Web of Science are recorded as
unavailable unless access exists; Google Scholar is discovery-only.

Feature and result claims prefer:

1. peer-reviewed papers or archival technical reports;
2. official preprints;
3. official software documentation;
4. inspected official source; and
5. secondary sources only to locate primary evidence.

Search snippets do not establish a study field. Unknown remains `unknown`, not
`no`. The record stores metadata and evidence locations, not redistributed
copyrighted papers.

## Eligibility and screening

The main publication window is 1994 through the review date; earlier ENO work
may enter as lineage. A work is included when it materially compares or
defines the relationship between at least two taxonomy classes, or when it
establishes a capability boundary essential to comparison design.

For every included study record:

- publication identity, year, DOI or archival URL;
- taxonomy class for each endpoint;
- WENO family and order;
- equations, dimensions, grid and geometry;
- state semantics and initialization/projection policy;
- reconstruction variables and characteristic policy;
- flux split or Riemann solver;
- time integrator and CFL policy;
- precision, hardware, language, and compiler;
- same-grid, same-DOF, same-error, or other comparison basis;
- kernel, step, resident, or complete-solve timing boundary;
- preparation/compilation and transfer treatment;
- memory and differentiation evidence;
- numerical correctness and independent reference;
- method-specific optimizations admitted or withheld;
- reported conclusions and author-stated limitations; and
- direct relevance to F0--F5.

Plausible exclusions retain a reason. Citation snowballing continues until one
complete pass adds no comparison that changes a candidate decision or the
experimental constitution.

## Fairness audit

No comparison is labeled simply “fair.” It is classified across these axes:

- **mathematical match:** equation, order, reconstruction family, flux,
  boundary treatment, time integrator, CFL, limiter, and precision;
- **resource match:** cells, stored degrees of freedom, memory budget, device,
  thread count, and implementation maturity;
- **outcome match:** achieved error, conservation, shock location/structure,
  positivity, and final physical time;
- **execution match:** kernel versus full solve, device residency, transfers,
  compilation/preparation, and cold versus warm state; and
- **capability match:** geometry, mesh regularity, adaptivity, and method-
  specific optimizations.

The future study will require two distinct lanes:

1. a **matched-component lane** that holds shared algorithmic choices fixed
   as far as the formulations permit; and
2. a **best-practical lane** that permits legitimate method-specific
   optimizations but records them explicitly.

Equal-grid timing is secondary. Accuracy-to-time and accuracy-to-memory are
the primary cross-formulation performance concepts because point values and
cell averages are different discrete states and may reach a target error at
different resolutions.

## Experimental-constitution constraints

Phase 1 may freeze only the constitution, not the later numerical thresholds.
The constitution must specify:

- structured Cartesian grids as the first head-to-head domain;
- scalar conservation laws followed by ideal-gas Euler;
- WENO-JS5 as the seed before an order sweep;
- FP64 reference qualification before FP32 performance endpoints;
- independent continuous mathematics with correct point-value and cell-average
  projections;
- correctness and differentiation gates before timing;
- matched and best-practical lanes kept separate;
- same-grid results reported but never substituted for accuracy-to-time;
- cold, warm, device-resident, and complete-solve timing boundaries;
- CPU and local RTX GPU first, with rented data-center hardware only after a
  frozen value-of-information decision; and
- Navier--Stokes, unstructured grids, production AMR, multiphase flow, and UI
  outside the initial structured comparison.

Automatic selection may eventually choose a discretization only under an
explicit accuracy/capability contract. Unlike CPU versus GPU placement,
switching FD and FV changes the discrete mathematics and must remain visible
in result provenance even if a product UI hides routine machinery.

## Outputs

Phase 1 produces:

- `experiments/fd_fv_review/results/phase_1_20260827/search_log.json`;
- `experiments/fd_fv_review/results/phase_1_20260827/studies.json`;
- `experiments/fd_fv_review/results/phase_1_20260827/claim_matrix.json`;
- `experiments/fd_fv_review/results/phase_1_20260827/SHA256SUMS`;
- `experiments/fd_fv_review/verify_phase_1.py`;
- `docs/FD_FV_PHASE_1_RESULTS.md`; and
- `docs/FD_FV_EXPERIMENTAL_CONSTITUTION.md`.

## Stop condition

Phase 1 stops when all frozen search families and one stable citation-snowball
pass are recorded, included studies have source-grounded fields, exclusions
and access limits are explicit, F0--F5 have bounded decisions, the constitution
is internally consistent with GradFlow's engineering charter, artifacts
verify, coherent local commits exist, and the worktree is clean.

No FV implementation, new numerical result, optimization, or benchmark is
permitted during Phase 1.
