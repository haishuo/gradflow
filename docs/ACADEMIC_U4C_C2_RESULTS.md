# Academic U4-C C2 external performance results

Status: **complete with three prospectively excluded sizes**.

Date: 2026-08-30 (UTC)

## Correctness admission

All four implementations were finite and conservative at all four frozen
sizes. Only `N=8192` passed both cross-implementation float64 bounds, so it is
the only size with admitted timing.

| N | largest OpenSBLI max-normalized difference | largest OpenSBLI RMS-normalized difference | decision |
|---:|---:|---:|---|
| 8,192 | `1.13046e-12` | `2.45675e-14` | admitted |
| 131,072 | `1.45803e-10` | `1.20272e-12` | excluded: max bound |
| 1,048,576 | `1.34320e-8` | `4.71028e-11` | excluded: both bounds |
| 8,388,608 | `5.14482e-10` | `6.95967e-12` | excluded: both bounds |

The bound failures occur despite byte-identical input and conservation. The
two implementations use algebraically equivalent but differently ordered
floating-point reconstruction expressions; the final flux difference is
multiplied by `1/dx`. U4-C did not precommit a grid-scaled roundoff model, so it
does not retroactively relax the fixed bounds. These exclusions are not proof
that OpenSBLI is mathematically incorrect, but they prohibit performance
claims at those sizes in this campaign.

## Admitted resident result at N=8,192

Times are milliseconds per already-resident WENO-JS5 reconstruction plus
divergence. Each lane has 120 retained observations from six independent
workers. Ratios are paired worker medians, `OpenSBLI / PyTorch`; values below
one favor OpenSBLI.

| device | OpenSBLI median | PyTorch/TorchInductor median | ratio (95% bootstrap CI) | decision |
|---|---:|---:|---:|---|
| one-thread CPU | `0.0809495` | `0.0985150` | `0.825789` (`0.810998`, `0.837994`) | OpenSBLI win |
| RTX 5070 Ti CUDA | `0.0102400` | `0.0331840` | `0.309704` (`0.299913`, `0.312734`) | OpenSBLI win |

Equivalently, OpenSBLI was about `1.21x` faster on the one-thread CPU lane and
`3.23x` faster on the CUDA lane for this admitted cell. This directly answers
the external-baseline question for one scalar order-5 grid: ordinary compiled
PyTorch is competitive in scale but does impose measurable overhead relative
to generated OPS code, especially on CUDA.

The secondary cross-device ratios are about `7.91x` for OpenSBLI and `2.97x`
for PyTorch/TorchInductor in favor of CUDA. They characterize this operator and consumer
GPU only; they do not establish a general CPU/GPU crossover.

## Preparation observations

At `N=8192`, OpenSBLI symbolic generation, OPS translation/sequential build,
and CUDA build took approximately `0.650`, `0.734`, and `1.635` seconds. The
PyTorch/TorchInductor qualification workers observed first compiled calls of `5.109`
seconds on CPU and `1.458` seconds on CUDA; their complete process endpoints
were `7.938` and `4.363` seconds. These are one-off preparation observations,
not stable distributions and not resident timings.

## Scope

U4-C C2 compares a matched one-dimensional scalar FD-WENO-JS5 RHS. It does not
compare full solvers, Euler systems, three-dimensional applications, arbitrary
order, or stock unadapted OpenSBLI examples. C3 keeps transfer and fresh-launch
costs separate from these resident results.

## Evidence

The full frozen record is in
`experiments/academic_u4c/evidence/u4c_c2_20260830/`, including every raw
sample, qualification array, command, build log, generated-source hash, and
SHA-256 manifest. Run `python experiments/academic_u4c/verify_performance.py`
for offline verification.
