# FD/FV Euler Phase-6C performance protocol

Status: frozen before any Euler FD/FV performance measurement.

Freeze date: 2026-08-29 UTC.

## Purpose

Phase 6C measures only the two one-dimensional ideal-gas Euler WENO-JS5
formulations qualified by Phase 6B. Its primary question is:

> At a declared achieved smooth-solution error and execution boundary, which
> qualified FD or FV formulation and ordinary-PyTorch execution mode minimizes
> complete-solve time and observed memory on the local CPU and RTX GPU?

It separately characterizes large-state resident-step saturation and preserves
a bounded shock-workload pilot. Equal-grid ratios are diagnostics. They are not
the headline because FD stores point values while FV stores physical cell
averages and must be judged against its own oracle.

Correctness > performance > convenience remains binding. A timing is
ineligible unless its output retains the qualified formulation, float64 dtype,
device, finiteness, conservation, oracle accuracy, and eager/compiled parity.

## Immutable mathematics

Phase 6B governs equations, gamma, WENO-JS5 coefficients and epsilon, Roe
projection, global characteristic-family matrix-LF policy, boundary semantics,
CFL `0.1`, SSP-RK3, state meaning, exact projections, shock references, and
correctness thresholds. The registered identifiers remain:

```text
fd_classical_characteristic_js5_global_lf_euler1d_v1
fv_dimensional_characteristic_js5_global_matrix_lf_euler1d_v1
```

No coefficient, epsilon, flux, alpha, boundary, time-step, stabilization,
positivity, or projection policy may change after observing timing.

This phase does not add mixed precision, WENO-Z, HLLC, another order,
multidimensional Euler, Navier--Stokes, AOT packaging, DVEB, native code,
custom CUDA/Triton/C++, representation optimization, or data-center hardware.

## Required admission

Before the first timing call:

1. the Phase-6B independent verifier must pass;
2. the source tree must be clean at one committed revision;
3. Forge CUDA must be visible and freshly pass float64 CPU/CUDA and
   eager/full-graph-compiled FD/FV RHS parity;
4. the accelerator must be recorded as admitted under the infrastructure
   vocabulary; and
5. the canonical output directory must not exist.

An admission failure stops the campaign without substituting historical data.

## Machine and process controls

The CPU is Forge's AMD Ryzen 5 7600X. Workers use six PyTorch intra-operation
threads and one inter-operation thread and record logical CPUs and process
affinity. The GPU is Forge's NVIDIA GeForce RTX 5070 Ti. Workers record model,
UUID, driver, runtime, capability, multiprocessor count, and memory.

Each matrix cell runs in an isolated process with a fresh temporary
TorchInductor cache. CPU durations use `perf_counter_ns`. CUDA resident
durations use CUDA events with explicit synchronization. Transfer-inclusive
and launch-to-exit durations use synchronized wall time. Raw samples, failures,
and memory observations are retained.

These are consumer-RTX float64 observations. The GPU's deliberately restricted
FP64 throughput prevents treating the result as an algorithmic GPU ceiling.

## Lane A: smooth complete-solve accuracy matrix

Use the Phase-6A entropy wave at

```text
N = (24, 36, 54, 81, 162), T = 0.1.
```

For each method, size, and device, run three isolated workers. Every worker
measures eager and fixed-shape `torch.compile(fullgraph=True, dynamic=False)`
SSP-RK3 execution from the same method-appropriate initial projection:

- one untimed complete-solve warmup;
- three state-resident complete-solve samples; and
- on CUDA, one warmup plus three prepared transfer-inclusive samples from CPU
  input through synchronized CPU output.

The Python time loop and adaptive CFL decision are not compiled into one giant
graph. Every time step uses the qualified on-device global CFL reduction,
copies its scalar decision to host control, and applies exact final-step
shortening. That synchronization is part of every complete-solve timing. The
state remains on its declared device, but this endpoint is named
`state_resident_host_controlled`, not device-autonomous.

For compiled mode, the first complete solve—including first-call compilation—
is recorded separately and excluded from warm samples. It is not called AOT or
full cold latency.

Each worker records raw samples, statistics, steps, L1/L2/Linf error,
componentwise conservation drift and accumulated-roundoff bound, terminal
array hash, persistent bytes, peak RSS, compiler-cache bytes, and CUDA peak
allocated/reserved memory. Eager and compiled terminal arrays must agree within
`5e-11`; repeated terminal hashes must agree exactly.

The primary float64 L2 targets are:

```text
(5e-6, 1e-6, 1e-7, 1e-8, 1e-9).
```

For each target, method, and boundary, select the eligible measured
configuration with the smallest aggregate median. Aggregate timing is the
median of the three worker medians. No interpolation or fitted crossover is
used. Primary boundaries are warm CPU, warm CUDA state-resident/host-controlled,
and prepared transfer-inclusive CUDA. Eager or compiled may win independently.

## Lane B: cold launch-to-host pilot

At `N=(24,81,162)`, run one fresh process for each method, device, and
eager/compiled mode. The orchestrator measures launch through imports, state
construction, transfers, any compilation, the adaptive complete solve,
synchronization, final CPU materialization, validation, serialization, and
process exit.

These 24 single observations are a cold pilot, not replicated estimates. A
ratio within 10% is classified `unresolved_cold_pilot`; larger differences are
descriptive only. Packaged AOT is `not_implemented` and is never approximated
by a warm JIT cache.

## Lane C: large-state resident-step matrix

Measure one fixed-step SSP-RK3 update, excluding adaptive-CFL scalar control,
at

```text
N = (32, 128, 512, 2048, 8192, 32768, 131072, 524288).
```

The step uses `dt=0.05*dx`; it is an operator-through-step throughput probe,
not a physical-time solve. For each method and device, measure eager and
compiled modes with ten warmups and thirty resident samples. CUDA also records
five warmups and twenty wall-clock transfer-inclusive samples. First compiled
execution is recorded separately.

Run one baseline worker per cell. For each method, find the smallest size where
compiled resident CUDA is at least 5% faster than compiled CPU, then run two
additional independent workers on both devices at that size and the preceding
size. If no point qualifies, replicate the two largest sizes. A confirmed
crossover requires all three CUDA/CPU worker-median ratios below `1/1.05`.

Equal-grid FD/FV comparisons use a 5% indifference band and remain diagnostic.

## Lane D: shock complete-solve pilot

Preserve one fresh-process observation for each method, device, and
eager/compiled mode on:

```text
Sod:       N=(200,800), T=0.2
Shu--Osher N=(200,800), T=1.8
```

Each observation includes process launch, imports, input transfer, compilation
when requested, the adaptive-CFL solve, required per-stage density/pressure
checks, output transfer, oracle validation, and process exit. It records exact
failure stage, step count, minimum density/pressure, errors, structure/wave
metrics, RSS, CUDA memory, and host-visible result.

This is a 32-point application pilot with no close-ratio claim. FD and FV
errors retain their distinct point/cell-average meanings. A shock timing is
eligible only if all inherited Phase-6B thresholds for that method/problem/size
remain satisfied where the threshold is defined; otherwise its failure is
preserved and no timing winner is declared.

## Statistics and eligibility

All samples must be finite and strictly positive. Quartiles use linear
interpolation. A cell is eligible only when its worker succeeds and every
declared correctness, determinism, dtype, device, conservation, compiler, and
timing check passes. Missing, failed, out-of-memory, or compile-failed points
remain explicit.

Peak RSS includes interpreter and compiler state. CUDA peak memory includes
allocator behavior. Neither is presented as pure numerical working-set size;
persistent state bytes are reported separately.

## Records and independent verification

The canonical result directory is:

```text
experiments/fd_fv_euler/results/phase_6c_20260829/
```

The orchestrator preserves every worker command, stdout/stderr, return code,
fresh-cache identity, environment, raw sample, statistic, correctness metric,
memory observation, target selection, crossover replication, pilot, and
explicit failure. `SHA256SUMS` covers the aggregate and raw records.

An independent verifier recomputes hashes, sample statistics, eligibility,
accuracy-target selection, ratios, classifications, and crossover decisions
without rerunning timed work.

## Stop and claim boundary

Stop after all four lanes, conditional replication, immutable results,
independent verification, complete configured tests, bounded interpretation,
coherent local commits, and a clean tree.

Do not optimize after seeing results, begin Phase 6D, add AOT, change DVEB,
implement WENO-15, rent hardware, claim real-time CFD, infer H100 behavior,
claim universal FD/FV superiority, or push without explicit authorization.
