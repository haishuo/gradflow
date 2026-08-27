# FD/FV Phase-1 results

Review date: 2026-08-27 UTC.

Protocol freeze: commit `6fbdf1b7baf79d9360c5b3f5f9f848309f0be234`.

Phase 1 performed literature review and froze the future experimental rules. It
implemented no finite-volume solver and collected no new numerical or timing
result.

## Bottom line

Head-to-head FD/FV WENO comparisons already exist. The literature does **not**
support a universal winner.

Finite-difference WENO often has a structural work advantage on regular
multidimensional grids. Finite-volume WENO can justify additional work through
geometry, nonuniform/adaptive resolution, low-Mach or mapped-grid behavior, and
other formulation-specific capabilities. Moreover, “FV WENO” includes cheap
dimension-by-dimension constructions, genuinely high-order multidimensional
constructions, and optimized variants that reduce quadrature or Riemann calls.
Their cost and even formal accuracy are materially different.

The defensible GradFlow question is therefore not “Is FD or FV better?” It is:

> Under a declared accuracy, robustness, geometry, precision, hardware, and
> execution contract, where does each qualified formulation minimize complete
> time or memory, and where does it cease to be applicable?

The review justifies a bounded modern study of structured Cartesian WENO-JS,
provided it reports accuracy-to-time and accuracy-to-memory, separates matched
components from best-practical implementations, qualifies gradients before
timing them, and makes cold/warm/resident execution boundaries explicit. It does
not establish novelty or publishability by absence.

## What earlier work establishes

### Comparative literature is not new

Shu's 1997 lecture, 2003 comparative review, and 2016 survey explicitly relate
finite-difference and finite-volume WENO. The 2016 survey notes that in one
dimension their code and cost can be almost identical, while genuinely
multidimensional FV generally incurs reconstruction and quadrature work absent
from classical structured FD. These are mathematical and implementation
observations, not a present-day GPU benchmark.

- [Shu 1997, NASA NTRS](https://ntrs.nasa.gov/citations/19980007543)
- [Shu 2003, DOI 10.1080/1061856031000104851](https://doi.org/10.1080/1061856031000104851)
- [Shu 2016, DOI 10.1016/j.jcp.2016.04.030](https://doi.org/10.1016/j.jcp.2016.04.030)

Wang, Feng, and Spiteri show why equal-cell cost is not the whole story: a
finite-volume method on a nonuniform mesh may offset higher per-cell work by
using fewer cells and less memory for a target result. This directly motivates
accuracy-to-time rather than same-grid speed as the primary comparison.

- [Wang, Feng, and Spiteri 2008](https://doi.org/10.1016/j.amc.2007.06.024)

### FV must be classified before comparison

Zhang, Zhang, and Shu compare two two-dimensional Cartesian FV-WENO classes.
Their cheaper Class A is only second-order accurate for nonlinear systems,
whereas Class B preserves high order; nevertheless their shock resolution can
be comparable on the same mesh. Thus a label such as “FV WENO-5” does not freeze
either the mathematics or cost.

- [Zhang, Zhang, and Shu 2011](https://doi.org/10.4208/cicp.291109.080410s)

Teissier, Mäusle, and Müller later show that best-practical high-order FV can
reduce the reconstruction/quadrature growth assumed by standard
dimension-by-dimension implementations. A deliberately naive FV tensor-product
implementation would therefore be a useful matched diagnostic but not an
acceptable sole competitor.

- [Teissier, Mäusle, and Müller 2024](https://doi.org/10.1016/j.jcp.2024.113287)

### Direct results are strong but conditional

Luo, Xuan, and Xu report roughly fourfold lower per-step CPU time for their
fifth-order FD WENO solver in a two-dimensional cavity case, while their FV
gas-kinetic scheme can be more accurate and robust on under-resolved viscous
flows. This is an intentionally different-flux comparison—not a pure
discretization ablation—and demonstrates why both matched and best-practical
lanes are required.

- [Luo, Xuan, and Xu 2013](https://doi.org/10.4208/cicp.110212.021112a)

Balsara, Bhoriya, and Shu provide the closest recent efficiency comparison. On
one Xeon Gold 6248R core with GNU Fortran 9.4, their reported alternative-FD
scheme is approximately 7–14× faster in selected 2-D cases and 5–12× faster in
selected 3-D cases than their divergence-preserving FV scheme. The authors also
state that FV Riemann-call reductions were not used and that ADER-style time
integration could reduce FV cost. The result is important evidence, not a
universal ceiling: it is CPU-only, same-grid, system-specific, and compares AFD
rather than GradFlow's classical split-flux FD.

- [Balsara, Bhoriya, and Shu 2025](https://doi.org/10.1007/s42967-025-00517-y)

Grimm-Strele, Kupka, and Muthsam show the other side of the phase diagram: in
their curvilinear astrophysical tests, the FD method is restricted to smooth
mappings/high-Mach regimes, while FV remains accurate in their low-Mach and
nonsmooth-grid tests. A fast method outside its capability domain is not a win.

- [Grimm-Strele, Kupka, and Muthsam 2014](https://doi.org/10.1016/j.cpc.2013.11.005)

### Accelerator and differentiation evidence exists separately

Native GPU FD-WENO, differentiable JAX CFD, and arbitrary-order differentiable
PyTorch FV-WENO all exist. JAX-Fluids is primarily a finite-volume Godunov
framework, with additional flux-splitting paths requiring exact classification;
HOPE is a genuinely two-dimensional arbitrary-order PyTorch FV shallow-water
core. Neither project's choice of formulation is evidence of universal
superiority, and neither supplies the matched FD/FV study proposed here.

- [JAX-Fluids 2.0](https://doi.org/10.1016/j.cpc.2024.109433)
- [HOPE](https://doi.org/10.5194/gmd-18-8175-2025)
- [GPU-driven FD-WENO shallow-water solver](https://doi.org/10.1016/j.compfluid.2017.11.012)
- [GPU-accelerated FD-WENO DNS](https://doi.org/10.1016/j.compfluid.2022.105744)

The frozen search did not establish one study that jointly controls FD/FV
mathematics, achieved error, memory, CPU/GPU execution, compilation, transfers,
and differentiated-gradient reliability. This is a bounded review finding—not
proof that no such work exists.

## Candidate decisions

| ID | Decision | Consequence |
|---|---|---|
| F0 | Established non-novel | Direct FD/FV WENO comparison cannot be claimed as new. |
| F1 | Narrowed | Seek a conditional phase diagram; prohibit universal superiority claims. |
| F2 | Narrowed candidate | A modern structured-grid accuracy-to-time/memory CPU/GPU study is justified, pending external prior-art audit. |
| F3 | Narrowed candidate | Compare gradient reliability as supporting evidence; differentiable WENO itself is not new. |
| F4 | Narrowed candidate | A generated cross-formulation order study may follow JS5; arbitrary-order WENO is not new. |
| F5 | Insufficient evidence | Automatic discretization selection is a separate future systems question. |

The full rationales and prohibited statements are frozen in
`experiments/fd_fv_review/results/phase_1_20260827/claim_matrix.json`.

## Frozen experimental consequences

The experimental constitution requires:

- classical FD, AFD, dimension-by-dimension FV, and genuinely multidimensional
  FV to remain separate identities;
- one continuous problem with point evaluation for FD and mathematically
  correct cell averages for FV;
- independent correctness, convergence, conservation, robustness, device, and
  gradient gates before performance;
- a matched-component lane and a separately reported best-practical lane;
- achieved-error/time and achieved-error/memory as primary outcomes;
- same-grid and kernel-only timings only as causal diagnostics;
- cold, prepared/AOT, warm, device-resident, and kernel boundaries where
  applicable;
- FP64 qualification before FP32 or mixed-precision performance;
- local CPU and RTX 5070 Ti evidence first, with a frozen value-of-information
  decision before renting data-center FP64 hardware; and
- explicit provenance if a later product selects a discretization automatically.

The first implementation boundary is uniform structured Cartesian scalar
problems, then ideal-gas Euler, using WENO-JS5. Navier–Stokes, AMR, unstructured
geometry, wing workflows, UI, and real-time claims are deliberately deferred.

See [FD_FV_EXPERIMENTAL_CONSTITUTION.md](FD_FV_EXPERIMENTAL_CONSTITUTION.md).

## Review accounting and limitations

The frozen record contains ten OpenAlex query families (up to 100 screened
records each), ten first-page Crossref screens (50 records each), primary-text
publisher/arXiv/NASA/official-project searches, and one stable citation-
snowball pass. Scopus and Web of Science were unavailable. Broad database totals
are recorded as provider output, not relevant-study counts.

Machine-readable study fields use `unknown` when the source did not establish a
detail. A retracted superficially relevant paper is retained only in the
exclusion record. Copyrighted papers are linked, not redistributed.

The chief unresolved literature question is whether an obscure, poorly indexed,
or newer study already combines the full modern matrix. An external numerical-
CFD review remains mandatory before a paper novelty statement is frozen.

## Phase-1 stop condition

Phase 1 is complete when the JSON records and checksums pass
`experiments/fd_fv_review/verify_phase_1.py`, this constitution is committed, and
the working tree is clean. No FV implementation, WENO-15 extension, or new
performance campaign belongs to Phase 1.
