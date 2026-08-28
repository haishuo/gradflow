# FD/FV nonlinear Phase-5C performance protocol

Status: frozen before any nonlinear performance measurement.

Freeze date: 2026-08-28 UTC.

## Purpose

Phase 5C measures the qualified smooth pre-shock Burgers implementations from
Phase 5B. Its primary question is:

> At a declared achieved error and execution boundary, which qualified FD or FV
> formulation and ordinary-PyTorch execution mode minimizes time or memory on
> the local CPU and RTX GPU?

It separately asks where a large one-dimensional resident SSP-RK3 step begins
to benefit from CUDA. Equal-grid comparisons are secondary diagnostics because
Phase 5B found materially different FD and FV errors at the same `N`.

Correctness > performance > convenience remains binding. No timed result is
eligible unless the immutable Phase-5B record verifies and every worker repeats
its accuracy, conservation, eager/compiled parity, finiteness, dtype, and device
checks.

## Frozen mathematics and exclusions

The formulations, continuous problem, exact projections, `alpha=0.7`,
WENO-JS5 policy, SSP-RK3 time step, final time `T=0.1`, float64 dtype, and unique
periodic state semantics are unchanged from Phases 5A/5B.

This phase does not change epsilon, estimate alpha dynamically, introduce mixed
precision, optimize representations, add AOT packaging, use DVEB, add custom
CUDA/Triton/C++, extend WENO order, solve a shock, add dimensions, or claim a
universal FD/FV winner. It does not use a data-center GPU.

## Required admission

Before the first timing call:

1. the Phase-5A and Phase-5B verifiers must pass;
2. the source tree must be clean and identify one committed revision;
3. Forge CUDA must be visible to the measurement process and freshly pass one
   float64 eager/compiled FD/FV RHS and step parity probe;
4. CUDA status must be `admitted` under the infrastructure contract; and
5. no prior Phase-5C output directory may exist.

If admission fails, Phase 5C stops without timing. Historical timing is not
substituted.

## Hardware and process controls

The CPU is the local AMD Ryzen 5 7600X with six physical cores. Every worker
uses six PyTorch intra-operation threads and one inter-operation thread and
records logical CPUs and process affinity. No unverified affinity, clock, or
power control is claimed.

The GPU is Forge's NVIDIA GeForce RTX 5070 Ti. Workers record model, UUID,
driver, runtime, compute capability, multiprocessor count, and memory. Float64
results are explicitly consumer-RTX observations; the device's restricted FP64
rate prevents interpreting them as an algorithmic GPU ceiling.

Every matrix cell executes in an isolated process with a fresh temporary
TorchInductor cache. Synchronization is explicit. A worker failure, allocation
failure, compile failure, or nonstationary distribution remains in the record.

## Complete-solve accuracy matrix

Use

```text
N = (24, 36, 54, 81, 162).
```

For each method, `N`, and device, run three independent workers. Each worker
measures eager and fixed-shape full-graph compiled-step execution from the same
mathematical initial projection. The Python time loop calls either the eager or
compiled SSP-RK3 step; it is not unrolled into one giant solve graph.

For each mode:

- one complete-solve warmup;
- three warm device-resident complete-solve samples;
- all samples, median, mean, min, max, Q1, and Q3 retained.

CUDA workers additionally record three **prepared transfer-inclusive** samples:
each starts from the method-appropriate CPU tensor, transfers it to CUDA,
executes with already-prepared eager or compiled code, synchronizes, transfers
the final state to CPU, and exposes the host answer. One identical warmup
precedes them. CPU resident results are already host-visible and are not
duplicated under a transfer label.

For compiled mode, the first complete solve before warm sampling is recorded
separately and includes first-call TorchInductor compilation, but excludes
process launch/import. It is not labeled cold or AOT.

Each worker records method-appropriate L1/L2 error, mass change/bound, steps,
persistent bytes, process peak RSS, compiler-cache bytes, and CUDA peak
allocated/reserved bytes. Eager and compiled terminal arrays must agree within
`2e-11`; errors must match Phase-5B values at shared sizes within `2e-13`
absolute and otherwise pass the same exact oracle and conservation checks.

## Frozen achieved-error targets

The primary complete-solve targets are binary64 L2 error no larger than

```text
(2e-5, 3e-6, 5e-7, 1e-7, 5e-8).
```

For each target, method, and execution boundary, select the measured eligible
configuration with the smallest aggregate median. Aggregate timing is the
median of the three independent worker medians. No interpolation or fitted
crossover is used. If no measured `N` qualifies, record `not_reached`.

Primary views are error target versus:

- warm CPU complete-solve time;
- warm CUDA-resident complete-solve time;
- prepared transfer-inclusive CUDA complete-solve time; and
- corresponding peak process/device memory.

The bounded best-practical execution view may choose eager or compiled
independently for each method and target. It is an execution-policy result for
these two sources, not the best published implementation of either numerical
class.

## Cold launch-to-host pilot

At `N=(24,81,162)`, run one fresh process for every method, CPU/CUDA device,
and eager/compiled mode. The orchestrator measures process launch through
Python/PyTorch import, state construction, any compilation, input transfer,
complete solve, synchronization, final CPU materialization, validation,
serialization, and process exit.

These 24 single observations are explicitly a cold pilot, not replicated
distribution estimates. They may answer whether compilation/transfer dominates
a point, but cannot support a close ratio claim. A ratio within 10% is
`unresolved_cold_pilot`; larger observations remain descriptive until a later
deployment/AOT phase replicates them.

No packaged AOT artifact exists for the Burgers pair. The AOT endpoint is
`not_implemented`, never approximated by a warm JIT cache.

## Large-state resident-step matrix

To isolate device saturation and crossover without making an impractically
long physical-time solve, measure one SSP-RK3 step on

```text
N = (32, 128, 512, 2048, 8192, 32768, 131072, 524288).
```

For every method/device worker, measure eager and compiled modes with ten
warmups and thirty resident samples. CPU uses `perf_counter_ns`; CUDA uses CUDA
events with synchronization outside sample groups. CUDA additionally records
five warmups and twenty wall-clock transfer-inclusive step samples from CPU
input through CPU output. Compilation is excluded from warm samples and its
first call is separately recorded.

One baseline worker runs for every cell. For each method, the orchestrator finds
the smallest size where compiled resident CUDA is at least 5% faster than
compiled CPU. It then runs two additional independent workers for both devices
at that size and the immediately preceding size. If no such size exists, it
replicates the two largest sizes. Thus the declared crossover bracket has three
independent worker distributions without choosing replication based on which
method appears favorable.

A confirmed compiled device crossover requires all three CUDA/CPU
worker-median ratios at the claimed winning size to be below `1/1.05`. If that
condition fails, the crossover is `unresolved`. Eager and transfer-inclusive
results are reported separately and never substituted for resident compiled
evidence.

Equal-grid FV/FD ratios use a 5% indifference band. They are causal diagnostics,
not the primary accuracy-matched conclusion.

## Eligibility and statistics

All raw samples are retained. Derived statistics are recomputed by an
independent verifier. A timing cell is eligible only when:

- its worker exits successfully;
- outputs are finite float64 with correct shape and device;
- exact-oracle error and conservation checks pass;
- repeated outputs are deterministic under the declared tolerance;
- eager/compiled parity is at most `2e-11`;
- device/compiler admission applies; and
- every recorded duration is finite and positive.

Quartiles use linear interpolation between adjacent ordered samples. Ratios
are formed only from eligible medians. Peak RSS includes interpreter/compiler
state and is reported alongside persistent tensor bytes rather than presented
as pure numerical memory.

## Records and verification

The orchestrator preserves every isolated worker JSON, commands, environment,
fresh-cache identity, stdout/stderr, return code, source/protocol/predecessor
hashes, raw samples, statistics, errors, memory, derived target selections,
crossover replication, cold pilots, and explicit failures.

The canonical directory is
`experiments/fd_fv_nonlinear/results/phase_5c_20260828/`. `SHA256SUMS` covers
the aggregate and every raw record. The independent verifier recomputes hashes,
statistics, eligibility, target selections, classifications, and crossover
decisions.

## Stop and claim boundary

Stop after the frozen matrix, conditional replication, immutable record,
independent verification, full configured test suite, interpretation, coherent
local commits, and clean working tree. Do not optimize after seeing timings,
start Phase 5D, add AOT, run WENO-15, rent hardware, modify DVEB, claim
real-time CFD, claim a general FD/FV winner, or push without explicit
authorization.
