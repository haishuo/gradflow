# Deferred CUDA correctness-gates protocol

Status: frozen before supplemental execution.

Freeze date: 2026-08-28 UTC.

## Purpose

Several immutable GradFlow qualification records correctly reported CUDA as
unavailable to their original processes. Forge's RTX 5070 Ti was subsequently
found to be hidden by the default Codex device sandbox rather than absent from
the host. This supplement executes the correctness gates that genuinely
remain missing without rewriting the historical records or rerunning unrelated
CPU convergence and shock studies.

This is a correctness supplement. It collects no performance measurement and
changes no numerical formulation.

## Required immutable predecessors

Before execution, the committed verifiers must pass for:

- FD/FV Phase 3;
- FD/FV Phase 3R; and
- Euler boundary/shock Phase B.

Their record hashes and verifier output are retained. The supplement must run
from a clean source commit and preserve hashes of the canonical numerical
modules and this protocol.

## Scalar finite-volume WENO-JS5 gates

For a 37-cell deterministic state, compare CPU and CUDA results for float32
and float64:

- left and right reconstructed face states; and
- the global-LF scalar RHS.

The frozen maximum-absolute-difference tolerances are `2e-4` for float32 and
`2e-11` for float64. Outputs must be finite and remain on CUDA.

For both dtypes, compile the RHS and one SSP-RK3 step with
`torch.compile(fullgraph=True, dynamic=False)`. Each callable must produce one
graph, zero graph breaks, resident finite output, and compiled/eager agreement
within `5e-5` for float32 or `2e-11` for float64.

Profile one native-float64 CUDA RHS with CPU and CUDA profiler activities,
memory recording, and explicit synchronization. Record all `aten::to` events
and reject `aten::_to_copy`, `aten::copy_`, memcpy, H2D, D2H, host-to-device,
or device-to-host events. The output must remain on the input device and dtype.
Profiler synchronization is outside any numerical loop and no timing is
collected.

## One-dimensional Euler boundary gates

For generated WENO-JS orders `(5, 7, 9, 11, 13, 15)`, dtypes float32 and
float64, and periodic and transmissive boundaries, compare the same smooth
37-point conservative state on CPU and CUDA. The RHS tolerance is `3e-4` for
float32 and `5e-11` for float64. Every output must be finite and CUDA-resident.

For representative orders `(5, 11, 15)` and both boundaries, compile the
float64 CUDA RHS full-graph and static-shape. Require one graph, zero breaks,
finite resident output, and compiled/eager agreement at `5e-11` absolute.

On a CUDA state, the Euler CFL function must return a finite, positive,
zero-dimensional CUDA tensor. No host scalar conversion is performed.

This supplement does not rerun Sod or Shu--Osher time integrations: those
problems were already independently qualified on CPU, while the frozen missing
device gate was RHS agreement. It makes no CUDA shock-performance claim.

## Environment and decisions

Record GPU name, UUID, memory, compute capability, multiprocessor count,
driver, CUDA runtime, PyTorch version, Python, and platform. Apple MPS remains
separate and is recorded as untested unavailable on Forge.

The supplement passes only if every predecessor verifies, every scalar-FV
gate passes, every Euler gate passes, the source tree was clean at execution,
and no performance samples were collected. Failures remain in the record.
