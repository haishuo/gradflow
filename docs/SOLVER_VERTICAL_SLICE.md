# Solver vertical slice

## Implemented surface

GradFlow now has a deliberately narrow `Solver` that accepts caller-provided
initial states for one fully specified system formulation:

```python
import torch
import gradflow

state, spacing = gradflow.periodic_vortex(
    (64, 64, 64), device="cuda", dtype=torch.float32
)
solver = gradflow.Solver(
    equations="euler",
    dimension=3,
    weno=("JS", 5),
    flux_split="global_lf",
    boundaries="periodic_duplicated",
    dtype=torch.float32,
    spacing=spacing,
)
result = solver.run(state, steps=10)
```

The state layout is `(5, nz+1, ny+1, nx+1)`: density, three momenta, and total
energy. Every axis stores both periodic endpoints. Spacing is supplied in
coordinate order `(dx, dy, dz)`; tensor axes are conventionally reversed.

The solver validates shape, dtype, layout, finite values, positive density,
and positive pressure. It never changes the caller's dtype or device. A CUDA
physical-state validation performs one declared scalar synchronization before
integration; `solver.last_run` records it. Fixed-step numerical loops perform
no host/device transfer or scalar extraction.

## Mathematics

The slice is exactly the preserved Shu Euler specialization:

- 3-D compressible Euler with gamma 1.4;
- dimension-by-dimension Roe characteristic JS-WENO-5;
- Shu's central-flux-plus-nonlinear-correction form;
- `epsilon=1e-6` with the preserved 12-times-scaled indicators;
- per-line, per-family global LF speeds enlarged by 10 percent;
- duplicated periodic endpoints; and
- adaptive-CFL SSP-RK3.

For differentiability, the package evaluates the nonlinear weights as
normalized inverse squared indicators. This is algebraically identical to the
historical product form, but avoids float32 autograd overflow in perfectly
smooth regions. The frozen benchmark source remains unchanged.

Forward gates observed:

| Point | Package versus frozen PyTorch | Package CPU versus CUDA |
|---|---:|---:|
| N=6, one step | `5.960e-8` | `4.768e-7` |
| N=6, ten steps | `7.451e-9` | `1.192e-6` |
| N=32, one step | `3.638e-11` | `4.768e-7` |

One-step autograd on the smooth vortex produces finite, nonzero gradients.
TorchDynamo records the scientific Euler step as one graph with zero graph
breaks. These are seed gates, not a general differentiable-CFD claim.

## Time control

`steps=N` is available on CPU and CUDA. CFL values and accumulated simulated
time remain tensors on the input device.

`final_time=T` is currently available only on CPU. Adaptive final-time control
needs a data-dependent stopping decision. On CUDA that would require a hidden
device-to-host scalar synchronization each timestep, so this slice refuses it
and asks for fixed steps. A future compiled/runtime loop may lift that
restriction without weakening the transfer contract.

## Backend behavior

`backend="pytorch"` and `"pytorch-eager"` select direct eager PyTorch on the
state device. With no installed native artifact, `backend="auto"` does the
same. The decision and reason are inspectable via
`solver.explain_backend(state)` and `solver.last_run`.

DVEB portable ABI v1 now makes three forward-only native requests legal after
the caller or installation supplies a `DvebArtifact` verified from its v2
manifest:

- `backend="cpu-simd"` uses the declared CPU worker count;
- `backend="cuda-native"` performs the ABI's CPU-to-GPU and GPU-to-CPU path;
- `backend="dveb"` asks a supplied, verified bounded model to choose; and
- `backend="auto"` may use that bounded model, but falls back to PyTorch when
  the point is outside its envelope.

The ordinary installation can configure the artifact through
`GRADFLOW_DVEB_ARTIFACT`, with optional `GRADFLOW_DVEB_MODEL` and
`GRADFLOW_DVEB_MODEL_SHA256`. Expert code may instead pass a
`DvebArtifact.from_manifest(...)` object to the constructor. The solver checks
the library, header, program, module, and model identities before execution.

ABI v1 accepts caller-owned CPU memory only. Its CUDA target includes required
H2D/D2H copies and reports those transfers. It cannot satisfy autograd,
GPU-resident input, adaptive `final_time`, noncubic grids, alternate spacing or
CFL, or a different mathematical formulation; `auto` uses PyTorch in those
cases and explicit native requests refuse.

The arbitrary-state gate passed at N=6 for one and ten steps and N=32 for one
step. CPU ABI, CUDA ABI, the unchanged direct portable runner, and independent
PyTorch all agreed within `2e-5`; the largest observed difference was
`8.345e-7`. See `DVEB_ABI_V1.md`.

Device-resident ABI v2 is exposed explicitly through `DvebDeviceContext`, not
through `Solver.run` automatic placement. It accepts and returns CUDA tensors
without host transfers and passed the separate E4 gate. See
`DVEB_DEVICE_ABI_V2.md`. Keeping this boundary explicit prevents the new
performance result from silently changing the older CPU-state `Solver`
contract before resident-state dispatch has its own design and calibration.

## Explicitly unsupported

The constructor rejects rather than approximates:

- Navier--Stokes viscosity;
- dimensions other than 3;
- WENO orders other than JS-5;
- local LF or other flux splitting;
- componentwise reconstruction;
- unique-node periodic grids or nonperiodic boundaries;
- dtypes other than float32;
- CUDA adaptive `final_time` control.

WENO-11 and WENO-15 have not begun.
