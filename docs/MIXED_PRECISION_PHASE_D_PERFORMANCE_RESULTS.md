# Phase-D scalar mixed-precision performance results

Execution date: 2026-08-27 UTC.

Status: **all 15 frozen CUDA endpoints completed and verified, including
pointwise compiled-versus-eager parity**.

## Main result

On the local RTX 5070 Ti, the fastest numerically eligible scalar policy was
binary32 smoothness indicators plus binary32 unnormalized nonlinear-weight
formation, with binary64 state, weight normalization, flux split, candidate
polynomials, face combination, and divergence.

Warm, compiled, device-resident median times for one `N=2^20` Burgers RHS were:

| Order | All FP64 | Combined mixed policy | Speedup | Numerical class |
|---:|---:|---:|---:|---|
| 5 | `1.5473 ms` | `0.4736 ms` | `3.267x` | `engineering` |
| 11 | `8.7168 ms` | `1.2343 ms` | `7.062x` | `tight` |
| 15 | `13.1154 ms` | `7.1518 ms` | `1.834x` | `tight` |

All three exceed the frozen 5% practical-resolution threshold. The WENO-5
policy is not `tight`; its use would require an application's tolerance to
admit the previously recorded `1.99e-4` worst normalized discrepancy.

## Ablation

Warm compiled speedups over the same-order all-FP64 control were:

| Policy | WENO-5 | WENO-11 | WENO-15 |
|---|---:|---:|---:|
| Indicators FP32 only | `1.029x` | `1.491x` | `1.556x` |
| Weight formation FP32 only | `1.923x` | `1.919x` | `1.087x` |
| Both eligible blocks FP32 | `3.267x` | `7.062x` | `1.834x` |
| All internal blocks FP32 | `5.992x` | `7.035x` | `4.546x` |

Indicator-only WENO-5 is unresolved under the 5% rule. Every other passing
candidate in this table is performance-positive. The all-internal-FP32 row is
an inaccurate hardware floor and cannot be recommended regardless of speed.

The ablation is not simply proportional to the number of demoted operations.
At WENO-11, the accurate combined policy was marginally faster than the failed
all-internal-FP32 endpoint and used less measured peak memory. At WENO-15,
indicator evaluation dominated the useful gain while numerator-only demotion
was modest. This is evidence that compiler fusion, expression structure, and
memory traffic matter alongside nominal FP32 throughput.

## Eager versus compiled

The combined policy also improved eager median execution by `1.310x`,
`1.525x`, and `1.513x` at orders 5, 11, and 15. Compilation materially enlarged
the benefit, particularly at WENO-11, which supports treating the PyTorch
expression and TorchInductor lowering as one measured execution endpoint.

Fresh-cache first calls took approximately:

- `1.69--1.81 s` at WENO-5;
- `6.59--7.16 s` at WENO-11; and
- `14.33--15.20 s` at WENO-15.

Those costs are excluded from warm medians and remain important for one-shot
deployment. This result supports warm/AOT use; it does not make cold JIT free.

## Memory

Measured peak allocated CUDA memory during compiled recorded calls was:

| Order | All FP64 | Combined mixed policy | Reduction |
|---:|---:|---:|---:|
| 5 | `208 MiB` | `148 MiB` | `28.8%` |
| 11 | `440 MiB` | `236 MiB` | `46.4%` |
| 15 | `960 MiB` | `424 MiB` | `55.8%` |

These are allocator observations for the frozen scalar workload, not a general
closed-form memory model.

## Correctness of the compiled endpoint

Every numerically eligible mixed policy passed the separately frozen
compiled-versus-eager gate. The worst eligible maximum normalized difference
was `4.366e-10`, over five orders of magnitude below the `5e-5` threshold.

The failed all-internal-FP32 endpoint showed compiled/eager normalized maximum
differences around `0.185--0.191`. This does not weaken the passing result; it
reinforces why low-precision performance floors cannot be treated as valid
solutions without a numerical contract.

## What can and cannot be said

The experiment establishes that, for this generated scalar WENO-JS
implementation and frozen problem on an RTX 5070 Ti, a carefully placed mixed
binary32/binary64 split can be both much faster and numerically compliant. It
also identifies high-precision normalization and the face-flux path as
important under small-signal tests.

It does not yet establish:

- safety or speedup for characteristic Euler;
- safety around strong shocks, positivity limits, or general boundaries;
- persistent FP32 state or RK accumulation;
- gradient agreement;
- an A100/H100, CPU, MPS, or multi-GPU result;
- end-to-end solver speedup including timestepping and transfers; or
- a universal optimal split for all WENO formulations.

The next required boundary is Tier 2: carry only the passing scalar candidates
into the existing independently qualified Euler smooth/shock suite, add
gradient and long-time checks, and then measure a full Euler RHS/step. No new
precision assignment should enter Tier 2 without a separately frozen reason.

## Reproducibility

Machine-readable result:
`experiments/mixed_precision/results/phase_d_performance_20260827/benchmark.json`.

- Performance protocol commit: `6aac149`
- Compiled-parity addendum commit: `acb9bdb`
- Source/runner commit: `5b0133bdc5804cccef2630fc10681598189e7cd3`
- Completed records: 15 of 15
- Samples: 30 per eager and compiled endpoint after 5 warmups
- Result SHA-256:
  `c7026c3263cf3cbd6c6e17eef245a1f88367a7e3ff43c3c3f72b345334922999`
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB, driver `580.173.02`
- PyTorch: `2.9.0.dev20250705+cu128`

The result directory contains and hashes the aggregate record plus every raw
isolated-worker record. The verifier checks the complete matrix, sample counts,
positive finite timings, file hashes, and eligible-policy compiled parity.

After the record was verified, the complete GradFlow suite passed on the same
CUDA host: 246 tests passed and 12 optional DVEB-artifact tests skipped with
declared environment-variable reasons. Lint passed on the active source,
mixed-precision experiments, and changed tests. Pre-existing lint findings in
byte-preserved baselines and explicitly noncanonical legacy experiments were
left untouched.
