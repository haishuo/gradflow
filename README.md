# GradFlow

GradFlow is restarting as a research project for general, differentiable,
high-performance finite-difference WENO in ordinary PyTorch.

Project decisions follow one strict precedence rule:

> **Correctness > performance > convenience.**

A convenience cannot justify a slower qualified path, and an optimization
cannot justify wrong or silently altered mathematics. The normative policy is
recorded in `docs/ENGINEERING_CHARTER.md`.

The current repository contains a validated exact-rational WENO-JS
constructor and a narrow 3-D Euler characteristic `Solver`, both qualified for
orders 5 through 15. Their numerical paths are written as readable shifts,
slicing/indexing, and elementwise tensor
operations. `conv1d` is an implementation hypothesis to be tested later, not
the project premise. No handwritten CUDA, Triton, C++, or custom operator is
part of the canonical package code.

The central research question is:

> Can a direct, maintainable PyTorch system construct, verify, differentiate,
> and efficiently execute arbitrary-order finite-difference WENO schemes—including
> a realistic WENO-15 case—without bespoke CUDA or Triton engineering?

This remains a research question, not a completed general capability or
novelty claim. Scalar and characteristic Euler periodic WENO-15 paths are
correctness-qualified. A bounded one-dimensional periodic/transmissive Euler
path has also passed frozen Sod and Shu--Osher gates at representative orders
5, 11, and 15, but general equations, multidimensional boundaries, geometry,
performance, and the target `Solver` surface remain open.

Phase C found that arbitrary-order symbolic FD-WENO generation, differentiable
WENO CFD, arbitrary-order PyTorch finite-volume WENO, characteristic GPU WENO,
and high-level heterogeneous CFD generation are all prior art. GradFlow's
academic candidate is consequently a narrower matched characterization of one
exact-generated direct-PyTorch FD-WENO path—not a claim to have invented those
ingredients. See `docs/LITERATURE_REVIEW_PHASE_C_RESULTS.md`.

FD/FV Phase 1 found substantial direct comparative literature and rejects any
universal “FD versus FV” winner. It freezes a future structured-grid study as
an accuracy-, capability-, memory-, and execution-conditioned phase diagram,
with matched-component and best-practical lanes kept separate. It contains no
new FV implementation or timing result. See `docs/FD_FV_PHASE_1_RESULTS.md`
and `docs/FD_FV_EXPERIMENTAL_CONSTITUTION.md`.

FD/FV Phase 2 has now frozen and independently derived the first scalar
finite-volume WENO-JS5 mathematical contract. Exact rational coefficient,
polynomial, smoothness, projection, upwinding, and conservation oracles are
committed. Phase 3 implemented that contract and passed nine gate areas, but
the first frozen qualification failed its negative-advection spatial-rate and
profiler-event criteria. The prospective Phase-3R study then demonstrated
approximately fifth-order noncritical behavior in both orientations and found
no observed CPU copy/data-movement event. The exact scalar CPU seed is now
qualified under the combined evidence, while the original failure remains
preserved. The formerly deferred scalar-FV CUDA agreement, compilation, and
movement gates, plus the one-dimensional Euler boundary device gates, have now
passed on Forge's RTX 5070 Ti. That admission preceded all Phase-4 timing. See
`docs/FD_FV_PHASE_2_RESULTS.md`, `docs/FD_FV_PHASE_3_RESULTS.md`, and
`docs/FD_FV_PHASE_3_RESOLUTION_RESULTS.md`. The linked prospective device
supplement is `docs/DEFERRED_CUDA_GATES_RESULTS.md`.

FD/FV Phase 4 admitted and measured the matched smooth scalar CPU matrix in
1-D, 2-D, and 3-D. Its surprising `N=27^3` compiled FV observation was then
replicated and causally characterized in 38 fresh CPU workers. Similar fast FV
modes recurred, but the frozen strong-replication criterion failed; the effect
was unique to `N=27`, highly multimodal, absent under stable one-thread timing,
and unsupported by a unique generated-kernel transition. It is therefore a
localized CPU multithread-runtime observation, not an FV crossover. CUDA
was initially hidden by the execution sandbox, then admitted and replicated
on Forge's RTX 5070 Ti. Compiled resident FD was unresolved at `N=18^3` and
approximately `9--12%` faster at `N=27^3--64^3`, with tightly replicated GPU
timings. See `docs/FD_FV_PHASE_4_RESULTS.md`,
`docs/FD_FV_PHASE_4_REPLICATION_RESULTS.md`, and
`docs/FD_FV_PHASE_4_CUDA_RESULTS.md`.

Phase D has now completed an exhaustive scalar binary32/binary64 search over
384 coarse and 768 refined order/policy pairs. On the frozen cases, binary32
smoothness indicators and unnormalized nonlinear-weight formation can be
combined only when weight normalization and the reconstructed face-flux path
remain binary64. The combined split delivered `3.267x`, `7.062x`, and `1.834x`
warm compiled scalar-RHS speedups at WENO-5, WENO-11, and WENO-15 on the local
RTX 5070 Ti. This is a bounded candidate result, not yet an Euler or universal
mixed-precision recommendation. See
`docs/MIXED_PRECISION_PHASE_D_PERFORMANCE_RESULTS.md`.

FD/FV nonlinear Phase 5A has frozen the next scalar boundary: smooth
pre-shock periodic inviscid Burgers with exact characteristic point values and
exact conservation-primitive cell averages. The standard-library oracle and
immutable records pass without adding a production Burgers solver or collecting
timing. The same phase also corrects future infrastructure records so
process-hidden CUDA is not conflated with absent host hardware. See
`docs/FD_FV_PHASE_5A_RESULTS.md` and
`docs/EXECUTION_INFRASTRUCTURE_ADMISSION.md`.

Phase 5B has now qualified the corresponding production FD and FV Burgers JS5
operators. Both pass exact-oracle, convergence, conservation, differentiation,
CPU/CUDA, full-graph compiler, and no-transfer gates. On the frozen smooth
case, FV retained approximately fifth-order whole-grid behavior; classical FD
retained approximately fifth order away from critical points while recording
the expected global JS critical-point degradation. No timing was collected.
See `docs/FD_FV_PHASE_5B_RESULTS.md`.

Phase 5C has completed the first nonlinear accuracy-to-time comparison. On the
frozen smooth pre-shock Burgers case, FV reached every error target with fewer
cells and was `1.97--3.38x` faster on CPU and `2.12--3.54x` faster on resident
CUDA. CPU still won every bounded complete-solve and cold device comparison;
CUDA became strongly beneficial only for large resident step states, reaching
`7.09x` FD and `12.04x` FV speedups at `N=524,288`. The initial accumulated
roundoff gate failure and its prospective timing-free resolution are both
preserved. See `docs/FD_FV_PHASE_5C_INITIAL_RESULTS.md` and
`docs/FD_FV_PHASE_5C_RESULTS.md`.

FD/FV Euler Phase 6A has frozen the matched one-dimensional system contract
before production FV implementation. It preserves point-valued FD and
cell-average FV semantics, reuses the exact Sod and independent Shu--Osher
authorities, and commits exact smooth/Sod projections plus conservative
Shu--Osher restrictions. All oracle gates pass; no Euler FD/FV timing has begun.
See `docs/FD_FV_PHASE_6A_RESULTS.md`.

## Current seed

`gradflow.weno5_rhs` operates on unique periodic point samples along the last
tensor dimension. It preserves the caller's tensor, device, and float64 dtype;
it does not perform hidden host/device conversion. A separate
`gradflow.weno5_rhs_gottlieb_periodic` adapter preserves the duplicated-endpoint
convention needed by the committed MATLAB oracle.

```python
import math
import torch

from gradflow import weno5_rhs

n = 128
x = torch.arange(n, dtype=torch.float64) / n
u = torch.sin(2.0 * math.pi * x)
rhs = weno5_rhs(u, 1.0 / n, lambda q: q, alpha=1.0)
```

The same function is intended to be passed to `torch.compile`; compilation is
an execution choice, not a different numerical implementation.

The generated scalar interface is:

```python
from gradflow import WENOJS

scheme = WENOJS(order=11)
rhs = scheme.rhs(u, 1.0 / n, lambda q: q, alpha=1.0)
```

One exact-rational constructor generates candidate polynomials, optimal
weights, and smoothness indicators for every odd order. Orders 5, 7, 9, 11,
13, and 15 have passed the bounded scalar periodic gate. See
`docs/ARBITRARY_ORDER_WENO_JS_RESULTS.md`. The same exact generated
reconstruction data now drives the separately qualified characteristic Euler
path.

## Current Solver slice

The first system API accepts arbitrary caller-provided states for the
formulation and orders that have passed the current gate:

```python
import torch
import gradflow

state, spacing = gradflow.periodic_vortex((32, 32, 32))
solver = gradflow.Solver(
    equations="euler",
    dimension=3,
    weno=("JS", 11),
    flux_split="global_lf",
    boundaries="periodic_duplicated",
    dtype=torch.float32,
    spacing=spacing,
)
result = solver.run(state, steps=1)
```

Direct eager PyTorch remains the zero-configuration path. A hash-qualified
DVEB portable ABI v1 artifact can accept a matching caller-owned CPU state and
execute the WENO-5 forward step on CPU SIMD/OpenMP or CUDA. Native use is
restricted to the exact compiled 3-D Euler WENO-5 formulation, positive fixed
step counts, cubic grids, spacing `10/N`, CFL 0.1, contiguous float32 state,
and no autograd. Higher-order and float64 requests remain on PyTorch.
Automatic DVEB dispatch additionally requires an explicitly
verified bounded placement model; otherwise `auto` falls back to PyTorch.
See `docs/SOLVER_VERTICAL_SLICE.md` and `docs/DVEB_ABI_V1.md`.

DVEB portable device ABI v2 is a separate explicit CUDA-resident interface.
`gradflow.DvebDeviceContext` binds one cubic grid to one CUDA device, owns a
reusable native workspace, and accepts caller-owned contiguous CUDA float32
input/output tensors on the current PyTorch stream. It performs no implicit
H2D/D2H conversion and is not an autograd backend. The first frozen E4
requalification found it 2.53--7.36x faster than packaged AOTInductor across
all ten tested points for this one fixed Shu Euler 3-D WENO-5 artifact. See
`docs/DVEB_DEVICE_ABI_V2.md`; automatic selection remains future work.

## Repository map

- `src/gradflow/weno5.py` — canonical direct PyTorch WENO-5 seed
- `src/gradflow/burgers.py` — matched smooth scalar Burgers FD/FV JS5 operators
- `src/gradflow/fv_weno5.py` — unqualified Phase-3 scalar periodic FV-WENO-JS5 candidate
- `src/gradflow/weno_js_coefficients.py` — exact-rational arbitrary-order construction
- `src/gradflow/weno_js.py` — generated axis-general scalar PyTorch WENO-JS
- `src/gradflow/euler3d.py` — generated characteristic Euler WENO-JS orders 5--15
- `src/gradflow/euler1d.py` — physical-state periodic/transmissive Euler path
- `src/gradflow/solver.py` — narrow validated problem and backend surface
- `src/gradflow/dveb_abi.py` — hash-checked DVEB CPU-state v1 and CUDA-state v2 adapters
- `tests/` — bounded oracle, convergence, conservation, device, and compiler gate
- `references/` — byte-preserved Gottlieb MATLAB and Jiang--Shu Fortran sources
- `baselines/` — exact DVEB screened comparator and its evidence
- `legacy/` — noncanonical historical representation experiments
- `experiments/fortran_scaling/` — frozen original and dynamically sized,
  fixed-form-repaired Jiang--Shu Fortran scaling descendant
- `experiments/shu_torch_ablation/` — matched 2-D/3-D Euler WENO Fortran versus
  direct-PyTorch CPU/GPU crossover experiment and the 30-run automatic-DVEB
  deployment bakeoff
- `experiments/weno_js_arbitrary_order/` — scalar orders 5--15 qualification record
- `experiments/characteristic_arbitrary_order/` — Euler orders 5--15 qualification
- `experiments/literature_review/` — frozen Phase-C search, study, and claim records
- `experiments/fd_fv_review/` — FD/FV Phase-1 search, study, claim, and
  experimental-constitution evidence
- `experiments/fd_fv_contract/` — independent exact FV-JS5 contract and oracle
  evidence for FD/FV Phase 2
- `experiments/fd_fv_qualification/` — frozen Phase-3 scalar FV implementation
  qualification and immutable failed first run
- `experiments/fd_fv_bakeoff/` — admitted Phase-4 scalar CPU accuracy/time/memory matrix
- `experiments/fd_fv_nonlinear/` — exact pre-shock Burgers oracle and Phase-5A freeze
- `experiments/mixed_precision/` — exhaustive scalar precision assignments and
  isolated CUDA performance records
- `docs/RESEARCH_DIRECTION.md` — research charter and claim boundaries
- `docs/LITERATURE_REVIEW_PHASE_C_PROTOCOL.md` — frozen systematic-review method
- `docs/LITERATURE_REVIEW_PHASE_C_RESULTS.md` — prior-art boundary and claim matrix
- `docs/FD_FV_PHASE_1_PROTOCOL.md` — pre-search FD/FV review protocol
- `docs/FD_FV_PHASE_1_RESULTS.md` — controlled-comparison literature result
- `docs/FD_FV_EXPERIMENTAL_CONSTITUTION.md` — binding rules for later bakeoffs
- `docs/FD_FV_PHASE_2_PROTOCOL.md` — scalar FV-JS5 contract/oracle freeze
- `docs/FD_FV_PHASE_2_RESULTS.md` — exact derivation and invariant results
- `docs/FD_FV_PHASE_3_PROTOCOL.md` — frozen scalar FV qualification rules
- `docs/FD_FV_PHASE_3_RESULTS.md` — passed evidence, two frozen failures, and next boundary
- `docs/FD_FV_PHASE_3_RESOLUTION_PROTOCOL.md` — prospective failure-resolution rules
- `docs/FD_FV_PHASE_3_RESOLUTION_RESULTS.md` — qualified CPU seed and remaining limits
- `docs/FD_FV_PHASE_4_PROTOCOL.md` — frozen scalar multidimensional bakeoff rules
- `docs/FD_FV_PHASE_4A_RESULTS.md` — timing-free 1-D/2-D/3-D admission
- `docs/FD_FV_PHASE_4_RESULTS.md` — matched CPU performance result and limits
- `docs/FD_FV_PHASE_5A_PROTOCOL.md` — frozen nonlinear Burgers contract and next gate
- `docs/FD_FV_PHASE_5A_RESULTS.md` — independent point/cell-average oracle result
- `docs/FD_FV_PHASE_5B_PROTOCOL.md` — frozen nonlinear correctness gate
- `docs/FD_FV_PHASE_5B_RESULTS.md` — qualified nonlinear FD/FV seed result
- `docs/EXECUTION_INFRASTRUCTURE_ADMISSION.md` — host/device visibility taxonomy
- `docs/MIXED_PRECISION_PHASE_D_PROTOCOL.md` — frozen Tier-1a scalar precision search
- `docs/MIXED_PRECISION_PHASE_D_TIER1B_RESULTS.md` — refined numerical seam
- `docs/MIXED_PRECISION_PHASE_D_PERFORMANCE_RESULTS.md` — verified RTX timing result
- `docs/ENGINEERING_CHARTER.md` — governing priority law, promotion gates,
  architecture, and technical-debt policy
- `docs/ACADEMIC_SCOPE.md` — bounded paper target, required evidence, and
  explicit commercial deferrals
- `docs/DVEB_RELATIONSHIP.md` — project independence and DVEB change-admission rule
- `docs/EULER_BOUNDARY_SHOCK_PROTOCOL.md` — next correctness-first trunk boundary
- `docs/EULER_BOUNDARY_SHOCK_PHASE_A_RESULTS.md` — frozen exact Sod and
  independent high-resolution Shu--Osher oracle record
- `docs/EULER_BOUNDARY_SHOCK_PHASE_B_PROTOCOL.md` — frozen nonperiodic
  implementation and qualification details
- `docs/EULER_BOUNDARY_SHOCK_PHASE_B_RESULTS.md` — exact Sod and independent
  Shu--Osher qualification result
- `docs/PRODUCT_VISION.md` — one-engine product target and user levels
- `docs/PROBLEM_MODEL.md` — backend-neutral scientific request model
- `docs/EQUATION_EXTENSION_CONTRACT.md` — requirements for equation families
- `docs/RESULT_AND_PROVENANCE_MODEL.md` — result, diagnostics, and audit target
- `docs/UI_WORKFLOW_CONCEPT.md` — guided, advanced, and show-code workflow
- `docs/ARBITRARY_ORDER_WENO_JS_RESULTS.md` — generated scalar qualification
- `docs/CHARACTERISTIC_ARBITRARY_ORDER_RESULTS.md` — characteristic Euler qualification
- `docs/BACKEND_SELECTION_CONTRACT.md` — evidence-bound automatic placement
  and explicit-backend rules
- `docs/SOLVER_VERTICAL_SLICE.md` — implemented API, gates, and limitations
- `docs/FORMULATION_LINEAGE.md` — mathematical and implementation lineage
- `docs/ARCHIVE_MANIFEST.md` — preservation artifacts and restoration steps

## Development gate

Install the project with its test dependencies and run:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

CUDA checks skip with an explicit reason when CUDA is unavailable. MPS is
recorded as untested in the gate documentation; it is not simulated.

## Reference redistribution

The local research references have strong recorded provenance, but no public
redistribution permission was found in the supplied material. See
`references/README.md` before preparing any public release.
