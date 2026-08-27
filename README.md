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
novelty claim. Scalar and one characteristic Euler periodic WENO-15 path are
now correctness-qualified, but general equations, boundaries, geometry,
performance, and the target `Solver` surface remain open.

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
- `src/gradflow/weno_js_coefficients.py` — exact-rational arbitrary-order construction
- `src/gradflow/weno_js.py` — generated axis-general scalar PyTorch WENO-JS
- `src/gradflow/euler3d.py` — generated characteristic Euler WENO-JS orders 5--15
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
- `docs/RESEARCH_DIRECTION.md` — research charter and claim boundaries
- `docs/ENGINEERING_CHARTER.md` — governing priority law, promotion gates,
  architecture, and technical-debt policy
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
