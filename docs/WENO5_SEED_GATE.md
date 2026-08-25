# WENO-5 seed gate

This gate is intentionally bounded. It validates the refounded WENO-5 seed; it
is not a performance campaign and does not authorize arbitrary-order work.

## Declared policy

- IEEE float64 tensors only for the validated seed.
- The canonical periodic grid contains unique nodes `x_j=j/n` on `[0,1)`.
- The Gottlieb oracle adapter separately accepts both endpoints on `[-1,1]`.
- Global Lax--Friedrichs alpha is either an explicit scalar/tensor or is
  computed on-device from `max(abs(flux_derivative(u)))`.
- The numerical loop performs no dtype/device conversion or scalar extraction.
- CUDA and MPS results may only be claimed when the hardware is actually
  available.

## Reproducible command

```bash
python -m pytest -ra
```

The test files separate the oracle, convergence/conservation, and device or
compiler checks. Exact environment and observed results are recorded below
after execution.

## Acceptance checks

1. The 75-step SSP-RK3 result agrees with the committed Gottlieb HDF5 oracle
   within the existing justified float64 tolerance `Linf <= 1e-12`.
2. Smooth unique-periodic linear advection demonstrates fifth-order spatial
   convergence on `n={40,80,160,320}`.
3. The periodic RHS sums to zero to a roundoff-scaled conservation bound.
4. CPU and CUDA agree in float64 under a declared scaled tolerance.
5. Eager execution and a fullgraph `torch.compile` RK step agree.
6. `torch._dynamo.explain` records one graph and zero graph breaks for that
   step.
7. The canonical numerical functions contain no `.item()`, `.cpu()`,
   `.cuda()`, `.to()`, or NumPy call, and outputs remain on the input device.
8. Autograd produces finite float64 gradients through the RHS.

## Recorded result

PASS on 2026-08-25:

```text
13 passed, 14 warnings
```

The warnings are PyTorch-internal deprecations of `torch.jit.script_method`;
they are not graph breaks or GradFlow warnings.

Environment:

- x86_64 Linux, Python 3.12.3;
- PyTorch `2.13.0+cu130`, CUDA runtime 13.0;
- NVIDIA GeForce RTX 5070 Ti, driver 580.173.02;
- pytest 9.1.1, h5py 3.16.0, NumPy 2.5.2; and
- the preserved DVEB `.venv-torch` supplied PyTorch, while test-only packages
  were installed under `/tmp`; neither DVEB nor `weno-reference` was modified.

Observed results:

- Gottlieb driver spacing was
  `max(diff(linspace(-1,1,101))) = 0.020000000000000018`. The 75-step
  SSP-RK3 result had oracle `Linf = 0.0` (`1e-12` allowed). Using decimal
  `0.02` instead produces about `1.33e-9`, so the test preserves the source
  grid convention rather than relaxing the tolerance.
- Right-moving linear-advection L2 errors for `n={40,80,160,320}` were
  `{5.8121081e-5, 1.6997017e-6, 5.0949299e-8, 1.5542512e-9}`, with observed
  orders `{5.0957, 5.0601, 5.0348}`.
- Left-moving errors were
  `{5.8121081e-5, 1.6997017e-6, 5.0949298e-8, 1.5542510e-9}`, with orders
  `{5.0957, 5.0601, 5.0348}`. This exercises the split family that the DVEB
  positive-advection screen could not observe.
- The unique-periodic implementation and the separately indexed Gottlieb
  duplicated-endpoint adapter agreed exactly (`Linf = 0.0`) for both
  left-moving linear advection and nonlinear Burgers flux. These regression
  cases keep the formerly unobservable negative split active.
- Three batched nonlinear periodic RHS sums had absolute residuals
  `{3.638e-12, 0, 0}`, below roundoff-scaled bounds
  `{2.820e-10, 2.627e-10, 2.647e-10}`.
- CPU/CUDA float64 RK-step agreement had `Linf = 2.220446049250313e-16`
  under the declared `1e-11` tolerance.
- `torch._dynamo.explain` recorded one graph, zero breaks, zero break reasons,
  and 408 captured operations for the CPU RK step.
- Fullgraph TorchInductor execution succeeded on CPU and CUDA. Compiled/eager
  `Linf` was `0.0` on CPU and `4.440892098500626e-16` on CUDA.
- The source-level AST check found no host transfer, device conversion, NumPy,
  or tensor scalar-extraction call in the numerical path. Device/dtype
  preservation and finite autograd gradients passed independently.
- SHA-256 guard tests passed for the selected MATLAB, Fortran, include, and
  DVEB baseline files.

Apple MPS is untested in this x86_64/NVIDIA environment; it is not simulated.
