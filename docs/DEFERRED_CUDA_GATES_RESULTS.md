# Deferred Forge CUDA correctness-gates result

Status: **passed**.

Execution date: 2026-08-28 UTC.

Source commit: `a73c777e0cd956d6bac6a9f1e6a307b7cf0e51bf`.

The immutable record is
`experiments/deferred_cuda_gates/results/qualification_20260828/qualification.json`,
SHA-256
`d0f0b1efcb76a5af5d3283dd0fdd51e9d503737c405d7332a302cf05446ea31b`.

## Purpose and decision

Forge's RTX 5070 Ti was hidden from several earlier qualification processes by
the default Codex device sandbox. This supplement executed the genuinely
missing correctness gates without rewriting those records or repeating their
unrelated CPU convergence and shock studies.

Every gate passed. No performance timing was collected.

## Scalar finite-volume WENO-JS5

For the frozen 37-cell state:

- float32 CPU/CUDA RHS maximum absolute difference was `1.1921e-7`;
- float64 CPU/CUDA RHS difference was `2.2204e-16`;
- reconstructed left and right face states agreed exactly in both dtypes; and
- every CUDA output was finite and resident.

The RHS and one SSP-RK3 step compiled for float32 and float64 as one graph with
zero graph breaks. The largest compiled/eager differences were `4.4107e-6`
for the float32 RHS and `8.4377e-15` for the float64 RHS, both inside their
frozen tolerances.

The CUDA profiler reproduced the 18 `aten::to` dispatches seen on CPU. They
allocated zero CPU and device memory and emitted no `_to_copy`, `copy_`,
memcpy, H2D, or D2H event. Input and output remained float64 on `cuda:0`.
This closes the Phase-3R device-movement question for the native CUDA path.

## One-dimensional Euler boundaries

All 24 CPU/CUDA RHS comparisons passed across:

- generated WENO-JS orders 5, 7, 9, 11, 13, and 15;
- float32 and float64; and
- periodic and transmissive boundaries.

The largest float32 difference was `5.2929e-5`, below `3e-4`. The largest
float64 difference was `7.8049e-14`, below `5e-11`. All CUDA outputs were
finite and resident.

For representative orders 5, 11, and 15, both boundary paths compiled on
CUDA as one graph with zero breaks. Compiled/eager float64 differences were at
most `6.1617e-14`. The CFL function returned a finite, positive,
zero-dimensional float64 tensor on `cuda:0`.

This closes Phase B's explicitly deferred device-agreement gate. It does not
claim that full Sod or Shu--Osher time integrations were executed or timed on
CUDA; the frozen missing gate was the numerical RHS/device surface.

## Environment and remaining unavailable device

The supplement used the same Forge RTX 5070 Ti, driver 580.173.02, PyTorch
2.13.0+cu130, and CUDA 13.0 environment as the Phase-4R CUDA replication.

Apple MPS remains genuinely untested because Forge is not Apple Silicon. No
MPS behavior is simulated or inferred.

## Optional DVEB backend regressions

With Forge CUDA exposed, the previously environment-skipped GradFlow/DVEB
integration tests were also run against their explicit artifacts:

- all five device-resident ABI-v2 tests passed using the preserved E4 nested
  artifact manifest; and
- all seven portable ABI-v1 tests passed using the current official artifact,
  the committed hash-verified placement model, and the unchanged direct
  executable.

This follow-up exposed one dormant test-fixture error. The intended
out-of-calibration fallback case used `N=4`, which violates WENO-5's
independent minimum of five unique cells before placement is consulted. The
fixture now uses `N=5`: it is mathematically legal but remains below the
model's minimum training size `N=7`. The corrected fallback test passes.

These are integration regressions, not new DVEB performance measurements or
an expansion of DVEB's role in the GradFlow research claim.
