# Phase-C systematic literature and claim-review protocol

Status: frozen before the Phase-C searches and screening.

Review date: 2026-08-27 UTC.

## Question

What published methods and maintained software overlap GradFlow's candidate
academic contribution: exact-generated arbitrary-order finite-difference WENO
in maintainable, differentiable ordinary PyTorch, including characteristic
systems through WENO-15 and competitive CPU/GPU execution without bespoke
accelerator kernels in the canonical implementation?

The review tests an intersection of properties. It must not infer novelty
because no single close work has every property, nor treat finite-volume and
finite-difference WENO as interchangeable.

## Claim candidates under review

The review will classify, narrow, or reject these candidates:

| ID | Candidate statement |
|---|---|
| C0 | A fixed WENO-5 implementation in PyTorch is not itself novel. |
| C1 | Exact or symbolic generation of arbitrary odd-order finite-difference WENO coefficients in an ordinary-PyTorch execution system may be a contribution. |
| C2 | Carrying one generated construction through characteristic compressible Euler and WENO-15 may be a contribution. |
| C3 | Differentiable finite-difference WENO suitable for inverse or sensitivity problems may be a contribution. |
| C4 | Competitive compiler-generated CPU/GPU execution from maintainable tensor source, without bespoke kernels in the canonical method, may be a contribution. |
| C5 | A data-driven selector among equivalent CPU, compiled-tensor, AOT, and native GPU representations may be a separate systems contribution. |

No candidate is a novelty claim before screening completes. “First,” “only,”
“unprecedented,” and proof-of-absence language are prohibited by this phase.

## Sources

Searches will cover the following source classes:

1. Crossref and OpenAlex metadata indexes;
2. arXiv;
3. Semantic Scholar;
4. NASA Technical Reports Server;
5. publisher indexes or primary paper pages surfaced through IEEE Xplore,
   ACM Digital Library, ScienceDirect, SpringerLink, SIAM, AIP, and JCP;
6. official documentation and repositories for software systems; and
7. backward and forward citation searches from included close works.

Google Scholar may be used only as a discovery supplement because its result
ordering and counts are not reproducible. Scopus and Web of Science are not
assumed available; lack of subscription access must be recorded rather than
silently represented as coverage.

## Frozen search families

Each family is run with both the full phrase “weighted essentially
non-oscillatory” and the acronym `WENO` where the source permits it.

```text
S1  WENO AND (PyTorch OR torch)
S2  WENO AND "automatic differentiation"
S3  WENO AND (differentiable OR inverse OR adjoint) AND (PyTorch OR JAX)
S4  "finite difference" AND WENO AND (GPU OR CUDA OR accelerator)
S5  "finite difference" AND WENO AND ("arbitrary order" OR symbolic OR generated OR generation)
S6  WENO AND (code generation OR DSL OR domain-specific language)
S7  WENO AND (TorchInductor OR torch.compile OR XLA OR JIT)
S8  WENO AND (characteristic OR Euler OR Navier-Stokes) AND (PyTorch OR JAX)
S9  "WENO-15" AND (GPU OR PyTorch OR JAX OR generated)
S10 (JAX-Fluids OR PyWENO OR OpenSBLI OR FUN3D OR OVERFLOW) AND WENO
```

Repository searches add these fixed forms:

```text
R1  pytorch weno
R2  torch weno finite difference
R3  jax weno
R4  arbitrary order weno generator
R5  weno cuda
```

Search-provider syntax may change quoting or boolean spelling but not the
concepts. The record must retain the submitted query, provider, date, URL or
endpoint where available, and number of candidates actually screened.

## Eligibility and screening

The publication window is 1996 through the review date. Earlier ENO work may
appear only when required to explain lineage. English abstracts are sufficient
for initial screening; a positive feature classification requires primary
paper text, official documentation, or source inspection.

A work enters the direct set if it materially addresses at least one of:

- finite-difference WENO coefficient or expression generation;
- finite-difference WENO execution on GPU or another accelerator;
- WENO implemented in PyTorch or another differentiable array system;
- automatic differentiation through WENO or a WENO-based solver; or
- automatic selection or compilation of equivalent WENO representations.

A work enters the close-prior-art set when it informs the claim boundary but
differs materially—for example finite-volume JAX CFD, production GPU CFD using
another discretization, or a symbolic WENO generator that emits low-level
code. Foundational WENO mathematics enters a separate lineage set.

Exclude duplicates, incidental mentions, neural networks merely named
“WENO-like,” papers without enough accessible evidence to classify any field,
and applications that use an unidentified third-party solver without changing
or evaluating the relevant method. Every otherwise plausible exclusion gets a
reason.

Screen titles/abstracts first, then inspect primary text or official source for
all direct and close entries. Backward/forward snowballing continues until one
complete pass over the direct set adds no new work that changes a claim field.

## Comparison fields

Every included work is classified with evidence and uncertainty for:

- bibliographic identity, DOI or archival URL, year, and version;
- paper, software, or both; maintenance evidence and last checked date;
- finite difference, finite volume, discontinuous Galerkin, or other spatial
  formulation;
- WENO family, supported orders, and hard-coded versus generated coefficients;
- scalar, componentwise, or characteristic reconstruction;
- equations, dimensions, grids, boundaries, and time integration;
- implementation language and user-facing language;
- CPU, CUDA, other GPU, and multi-device support;
- handwritten kernels, directives, source transformation, JIT, or AOT;
- automatic differentiation and demonstrated differentiable use;
- precision policy;
- correctness/oracle evidence;
- performance endpoints, residency, baselines, and hardware; and
- license, source availability, and direct relevance to C0--C5.

Unknown is a valid value and must not be converted to “no.” Repository
popularity is not scientific evidence. “Maintained” requires an official
release, commit, documentation update, or issue activity within 24 months of
the review date; otherwise the record says the last observed activity.

## Evidence hierarchy

Feature claims prefer, in order:

1. peer-reviewed paper or archival technical report;
2. official project documentation;
3. inspected official source code;
4. author preprint; and
5. secondary discovery material, used only to locate a primary source.

Search-result snippets cannot establish a feature. Absence from documentation
is recorded as unknown unless source inspection or an explicit statement
supports “no.” Short quotations remain within source permissions; the review
normally paraphrases.

## Outputs

Phase C produces:

- `experiments/literature_review/results/phase_c_20260827/search_log.json`;
- `experiments/literature_review/results/phase_c_20260827/studies.json`;
- `experiments/literature_review/results/phase_c_20260827/claim_matrix.json`;
- `experiments/literature_review/results/phase_c_20260827/SHA256SUMS`;
- `experiments/literature_review/verify_phase_c.py`;
- `docs/LITERATURE_REVIEW_PHASE_C_RESULTS.md`; and
- an updated research question and claim boundary where evidence requires it.

The records contain metadata, classifications, evidence locations, queries,
and hashes—not redistributed copies of copyrighted papers.

## Decision and stop condition

Each C0--C5 candidate receives one status:

- `established_non_novel`;
- `supported_candidate_contribution`;
- `narrowed_candidate_contribution`;
- `rejected_candidate_contribution`; or
- `insufficient_evidence`.

“Supported candidate contribution” permits an experiment and cautiously
worded paper claim; it does not prove novelty or publishability.

Phase C stops when the frozen searches and snowball pass are recorded, every
included study has source-grounded fields, exclusions are accounted for, the
claim matrix and limitations are written, artifacts verify, tests pass, local
commits are coherent, and the worktree is clean. Phase C performs no new WENO
implementation, numerical optimization, or performance campaign.
