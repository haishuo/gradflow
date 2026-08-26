# GradFlow

GradFlow is restarting as a research project for general, differentiable,
high-performance finite-difference WENO in ordinary PyTorch.

The current repository contains a validated scalar WENO-5 seed and one narrow
3-D Euler characteristic JS-WENO-5 `Solver` vertical slice. Their numerical
paths are written as readable shifts, slicing/indexing, and elementwise tensor
operations. `conv1d` is an implementation hypothesis to be tested later, not
the project premise. No handwritten CUDA, Triton, C++, or custom operator is
part of the canonical package code.

The central research question is:

> Can a direct, maintainable PyTorch system construct, verify, differentiate,
> and efficiently execute arbitrary-order finite-difference WENO schemes—including
> a realistic WENO-15 case—without bespoke CUDA or Triton engineering?

This is a question, not a completed capability or novelty claim. WENO-15 has
not been implemented here.

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

## Current Solver slice

The first system API accepts arbitrary caller-provided states, but only for
the formulation that has actually passed the current gate:

```python
import torch
import gradflow

state, spacing = gradflow.periodic_vortex((32, 32, 32))
solver = gradflow.Solver(
    equations="euler",
    dimension=3,
    weno=("JS", 5),
    flux_split="global_lf",
    boundaries="periodic_duplicated",
    dtype=torch.float32,
    spacing=spacing,
)
result = solver.run(state, steps=1)
```

Automatic placement currently selects direct eager PyTorch on the existing
state device. DVEB is not yet wired into this API because its qualified
executable always constructs the benchmark vortex and cannot accept the
caller's state. See `docs/SOLVER_VERTICAL_SLICE.md` for the exact supported and
rejected surface.

## Repository map

- `src/gradflow/weno5.py` — canonical direct PyTorch WENO-5 seed
- `src/gradflow/euler3d.py` — differentiable characteristic Euler JS-WENO-5
- `src/gradflow/solver.py` — narrow validated problem and backend surface
- `tests/` — bounded oracle, convergence, conservation, device, and compiler gate
- `references/` — byte-preserved Gottlieb MATLAB and Jiang--Shu Fortran sources
- `baselines/` — exact DVEB screened comparator and its evidence
- `legacy/` — noncanonical historical representation experiments
- `experiments/fortran_scaling/` — frozen original and dynamically sized,
  fixed-form-repaired Jiang--Shu Fortran scaling descendant
- `experiments/shu_torch_ablation/` — matched 2-D/3-D Euler WENO Fortran versus
  direct-PyTorch CPU/GPU crossover experiment and the 30-run automatic-DVEB
  deployment bakeoff
- `docs/RESEARCH_DIRECTION.md` — research charter and claim boundaries
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
