# Phase-C systematic literature and claim-review results

Review date: 2026-08-27 UTC.

Protocol: `LITERATURE_REVIEW_PHASE_C_PROTOCOL.md` at commit `b9d816b`.

Machine-readable record:
`experiments/literature_review/results/phase_c_20260827/`.

## Result

Phase C changes the academic story materially. GradFlow is **not** the first
system to generate arbitrary-order WENO, generate arbitrary-order
finite-difference WENO, use WENO in PyTorch, differentiate through WENO CFD,
perform characteristic GPU WENO, or generate portable high-performance CFD
from a high-level scientific description.

The strongest direct boundary is OpenSBLI. Its papers and inspected current
source establish a maintained Python/SymPy code-generation framework for
arbitrary odd-order finite-difference WENO-JS, characteristic flux splitting,
compressible Navier--Stokes, curvilinear grids, and heterogeneous CPU/GPU
execution through generated OPS C/C++ code. PyWENO had already symbolically
generated arbitrary-order WENO kernels, and PyClaw reported generated odd
orders 5--17 in 2012.

The strongest differentiable-array boundaries are JAX-Fluids and HOPE.
JAX-Fluids is a maintained, fully differentiable JAX CFD system with fixed,
hard-coded WENO families through WENO9, a primarily finite-volume Godunov
formulation, and a separate characteristic/component flux-splitting path.
HOPE is an arbitrary-order differentiable PyTorch finite-volume shallow-water
core; it demonstrated order 11 and GPU acceleration using convolution and
Einstein summation. JAX-Shock additionally reports WENO5-based inverse
parameter inference.

These results do not make GradFlow redundant. They change its candidate paper
from a method-invention claim into a narrower reproducible numerical-and-
systems study:

> How accurately, differentiably, and efficiently can one exact-generated
> Jiang--Shu finite-difference WENO implementation execute as maintainable
> ordinary PyTorch from orders 5 through 15, relative to mathematically
> matched CPU, compiler-generated, and native-GPU baselines?

This is a candidate research question, not a novelty or publishability claim.

## Review execution

The frozen S1--S10 families were run through OpenAlex, supplemented by
publisher search, arXiv, NASA NTRS, official documentation and repositories,
and one citation-snowball pass. The OpenAlex searches returned 359 records and
all returned metadata was screened; S2 was limited to the first 100 of 620
noisy relevance-ranked matches. Four 100-result Crossref pages were screened
before the service rate-limited the remaining calls. Semantic Scholar's
public endpoint returned HTTP 429 and is explicitly recorded as blocked.

Official source inspection pinned these especially important repositories:

| System | Inspected revision | Revision date | Maintenance classification |
|---|---:|---:|---|
| OpenSBLI | `e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d` | 2026-08-05 | maintained |
| JAX-Fluids | `22fb0652e6b6e1c6b5dcb90cbb5eaa9828ee78d6` | 2026-08-26 | maintained |
| PyWENO | `cfc12766556d8989b03c1051e2dd32510dc33f6e` | 2021-03-08 | not maintained under the frozen 24-month rule |
| torch-cfd | `a5fbf59f2af8b297ed68612dcff54e91c4fa380a` | 2025-11-21 | maintained, but no WENO source found |

The final evidence set contains 16 direct, close, or lineage records and eight
otherwise plausible exclusions with reasons. Search counts, partial failures,
source URLs, and classifications are retained in the JSON records rather than
relying on this narrative.

## Closest prior art

| Work | What it establishes | Material difference from GradFlow's current candidate |
|---|---|---|
| [OpenSBLI](https://doi.org/10.1016/j.cpc.2021.108063) | Symbolically generated arbitrary odd-order FD WENO-JS; characteristic compressible CFD; generated CPU/GPU code; maintained production-scale framework | Emits OPS C/C++ and has no documented automatic differentiation; GradFlow's candidate is direct ordinary PyTorch plus gradient/compiler qualification |
| [PyWENO](https://pyweno.readthedocs.io/) and [PyClaw](https://doi.org/10.1137/110856976) | Symbolic WENO coefficients and low-level code generation; PyClaw used generated orders 5--17 | Reconstruction toolkit / finite-volume wave propagation; not one direct tensor/autograd FD execution path |
| [JAX-Fluids](https://arxiv.org/abs/2402.05193) | Fully differentiable JAX compressible CFD, WENO through order 9, characteristic flux splitting, multi-accelerator scaling | Primarily finite volume; order implementations are hard-coded; JAX/XLA rather than PyTorch/TorchInductor |
| [HOPE](https://doi.org/10.5194/gmd-18-8175-2025) | Arbitrary-order differentiable PyTorch WENO, order 11, CPU/GPU execution and about 2x reported GPU speedup | Genuinely 2-D finite-volume shallow water; convolution/einsum; not Jiang--Shu FD flux reconstruction or characteristic Euler |
| [JAX-Shock](https://arxiv.org/abs/2601.04400) | Differentiable WENO5 compressible solver and inverse parameter inference | Fixed WENO5; source and exact FD/FV classification were not established during review |
| [HyPar GPU](https://doi.org/10.1016/j.compfluid.2022.105744) and [OpenCFD-SCU](https://arxiv.org/abs/2209.15333) | Mature, very large-scale GPU finite-difference compressible CFD with major speedups | Hand-engineered/native CUDA or HIP/MPI; fixed production schemes; no autograd |
| [PIF-WENO GPU](https://doi.org/10.1016/j.compfluid.2017.11.012) | Real-time finite-difference WENO shallow water on a consumer GPU | Specialized fixed third-order single-kernel application, not general generated WENO |

Finite volume and finite difference remain distinct. Their overlap is still
scientifically relevant: a paper cannot dismiss HOPE or JAX-Fluids merely
because their primary discretization differs, especially when it makes a
claim about PyTorch/JAX, automatic differentiation, GPU portability, or user
experience.

## Claim decisions

| ID | Decision | Consequence |
|---|---|---|
| C0 | `established_non_novel` | Fixed PyTorch WENO5 is a seed, never a novelty claim. |
| C1 | `narrowed_candidate_contribution` | Generation mathematics is prior art. Study the exact-generated direct-PyTorch integration, auditability, conditioning, differentiation, and compiler behavior. |
| C2 | `narrowed_candidate_contribution` | Characteristic systems and WENO15 are prior art. Use GradFlow's generated Euler 5--15 path as a controlled test vehicle, not a first. |
| C3 | `rejected_candidate_contribution` | Differentiable WENO and inverse use already exist. GradFlow still needs a gradient-checked inverse experiment as utility/correctness evidence, not as the headline novelty. |
| C4 | `narrowed_candidate_contribution` | GPU WENO and generated portable CFD are mature. A matched, endpoint-explicit PyTorch/TorchInductor study across order may still contribute. |
| C5 | `insufficient_evidence` | CPU/GPU automatic placement needs a separate review of autotuning and performance-portability literature and should stay outside the first paper. |

No candidate received `supported_candidate_contribution` yet. Phase C permits
the narrowed experiments; it does not establish that their eventual result
will be new or publishable.

## Required wording

GradFlow may say:

- its current source is one exact-generated, readable ordinary-PyTorch
  Jiang--Shu FD-WENO path through characteristic Euler orders 5--15;
- its committed tests establish the bounded correctness/compiler facts they
  actually measured;
- it will compare mathematically matched endpoints and report both wins and
  losses; and
- the review did not surface an identical maintained system combining every
  property, while absence is unproved.

GradFlow must not say:

- first or only PyTorch WENO;
- first arbitrary-order WENO generator or first WENO15 implementation;
- first differentiable WENO or WENO inverse solver;
- first high-level or portable GPU WENO/CFD system; or
- universal superiority over native, production, JAX, OpenSBLI, or CPU CFD.

## Effect on the academic roadmap

The first paper should be framed as a reproducible empirical characterization,
not a new WENO formula. The remaining academic sequence is:

1. compare GradFlow's exact construction and formulation boundary explicitly
   with OpenSBLI and PyWENO;
2. characterize conditioning, roundoff, epsilon sensitivity, critical points,
   and failure behavior as order rises;
3. run one independently gradient-checked sensitivity or inverse problem,
   treating differentiation as demonstrated utility rather than novelty;
4. freeze and run a formulation-matched arbitrary-order performance matrix,
   including cold/warm/AOT, transfer/resident, memory, compiler failures,
   optimized CPU, native GPU, and a feasible JAX/OpenSBLI close baseline; and
5. obtain an external numerical-CFD prior-art audit before freezing paper
   wording.

The broad commercial product question remains valid but is not evidence for
the first academic paper. General PDE selection, geometry, UI, and automatic
placement remain separately gated.

## Limitations and unresolved questions

- Scopus and Web of Science were unavailable.
- Crossref and Semantic Scholar rate-limited parts of the protocol.
- Search indexes have acronym noise and incomplete software coverage.
- OpenSBLI arbitrary-order generation is established, but an actual
  like-for-like order-15 execution comparison has not yet been run.
- JAX-Fluids' primary framing is finite volume; its flux-splitting overlap must
  be described precisely rather than relabeled wholesale as finite difference.
- JAX-Shock source, license, and precise spatial classification remain unknown.
- HOPE code should be reproduced from its archival deposit before relying on
  its performance number as a comparator.
- No external expert has audited the review, and the record cannot prove
  absence of unindexed work.
- License compatibility affects code reuse; this review records prior art and
  does not authorize copying code.

Phase C introduced no WENO implementation, optimization, or performance
campaign.
