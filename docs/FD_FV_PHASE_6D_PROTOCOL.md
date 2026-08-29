# FD/FV Euler Phase-6D replication and causal protocol

Status: frozen before any Phase-6D timing measurement.

Freeze date: 2026-08-29 UTC.

## Purpose

Phase 6D addresses two observations selected by the immutable Phase-6C record:

1. replicate the `N=800` launch-to-host shock result against the fastest CPU
   endpoint observed for each problem; and
2. causally characterize the compiled CPU FV performance regime that appeared
   near `N=8,192`.

This is replication and diagnosis, not optimization. The production FD/FV
sources, mathematics, compiler configuration, and Phase-6C worker semantics
remain unchanged. No implementation may be altered after observing Phase-6D
timing.

## Immutable dependencies and exclusions

Phases 6A--6C govern formulation identities, state semantics, oracles,
correctness thresholds, float64 policy, hardware vocabulary, clocks, and
eligibility. The Phase-6C verifier must pass before timing.

Phase 6D does not implement AOT, CUDA graphs, a device-side time loop, another
CFL policy, mixed precision, WENO-Z, HLLC, another order, multiple dimensions,
Navier--Stokes, DVEB, custom CUDA/Triton/C++, or a data-center experiment. It
does not tune the numerical or compiler source.

The device-autonomous/AOT question is deferred to a prospectively qualified
Phase 6E. Phase 6D may identify requirements for that phase but may not build
or time the new endpoint.

## Required admission

Before timing:

1. the Phase-6C independent verifier passes;
2. the source tree is clean at one committed revision;
3. Forge CUDA is visible and freshly passes the Phase-6C float64 compiled
   stage parity probe;
4. the CPU reports the expected Ryzen 7600X environment and supports thread
   counts `(1,2,4,6)`; and
5. the canonical output directory does not exist.

Every worker uses a fresh TorchInductor cache. Raw failures, compilation
failures, and ineligible results remain in the record.

## Lane A: shock replication

Use the unchanged Phase-6C launch-to-host shock worker at `N=800`. Run three
new isolated replicates for both FD and FV on:

- Sod: CPU eager versus CUDA compiled;
- Shu--Osher: CPU compiled versus CUDA compiled.

These CPU endpoints are the fastest CPU choices in the Phase-6C pilot for the
corresponding problem. Selection occurred before Phase 6D and is recorded, not
re-decided from Phase-6D samples.

Each of the 24 workers includes process startup, imports, compilation when
declared, transfers, adaptive global-CFL host control, per-stage positivity
checks, complete solve, oracle evaluation, final host materialization,
serialization, and process exit. The orchestrator records wall time.

For each problem and method, pair CPU and CUDA by replicate index. A CUDA win
is confirmed only if all three CUDA/CPU launch-to-exit ratios are below
`1/1.05`. Ratios within the 5% band are unresolved. The result is
formulation-specific; FD point and FV cell-average errors are not directly
compared.

Every replicate must reproduce the Phase-6C terminal hash for the same
problem/method/device/mode exactly and pass all inherited shock gates.

## Lane B: compiled CPU FV regime characterization

Measure the unchanged fixed-step periodic entropy-state SSP-RK3 worker on CPU.
The primary matrix is:

```text
N       = (2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768)
threads = (1, 6)
methods = (FD, FV)
```

At `N=(4096,8192,32768)`, also measure threads `(2,4)`. Every baseline worker
measures eager and fixed-shape full-graph compiled execution with ten warmups
and thirty samples. At `N=(4096,8192,32768)` and threads `(1,6)`, run two
additional isolated replicates, producing three distributions for the main
interaction points.

The worker records:

- raw samples and recomputable statistics;
- eager/compiled terminal hashes and parity;
- peak RSS and compiler-cache bytes;
- TorchInductor generated-kernel count, generated vector-kernel count,
  pre-fusion IR nodes, and estimated bytes accessed; and
- generated C++ cache-file count, total bytes, hashes, line counts, OpenMP
  pragma count, `parallel_for` marker count, and vectorization-marker count.

Generated source text itself is not copied into the repository; its hashes and
structural counts are preserved. Compiler metrics are diagnostic and never
substituted for measured time.

## Frozen causal summaries

For each size/thread point compute:

- compiled `FV/FD` time ratio;
- eager `FV/FD` time ratio;
- per-method `T1/T6` compiled thread speedup where available;
- compiled/eager ratio;
- FV/FD generated-kernel, IR-node, and estimated-byte ratios; and
- structural code-signature changes between adjacent sizes.

The following classifications are frozen:

- `thread_interaction_supported`: at two consecutive primary sizes at or above
  `N=8,192`, the six-thread compiled FV/FD ratio is at least `1.5x` its
  one-thread value, with all replicated interaction-point ratios on the same
  side of that threshold;
- `traffic_expansion_supported`: at two consecutive primary sizes at or above
  `N=8,192`, FV estimated bytes accessed or pre-fusion IR nodes are at least
  `1.5x` FD while CUDA Phase-6C FV/FD time remained below `1.25` at the same
  recorded sizes where available;
- `codegen_transition_supported`: an FV generated-kernel or C++ parallelism
  signature changes at the first primary size where six-thread FV/FD compiled
  time exceeds `2.0`, without the corresponding FD signature change; and
- `unresolved_mixture`: none or more than one causal classification is
  supported without a unique mechanism.

These rules characterize evidence; they do not prove compiler internals from
correlation. If multiple mechanisms are supported, the correct result is a
mixture requiring lower-level follow-up.

## Statistics, records, and verification

Every duration must be finite and positive. Aggregate timing is the median of
worker medians where three replicates exist. Quartiles use linear
interpolation. A worker is eligible only if output is finite float64,
deterministic, shape/device-correct, and eager/compiled parity is within
`5e-11`.

The canonical directory is:

```text
experiments/fd_fv_euler/results/phase_6d_20260829/
```

`SHA256SUMS` covers the aggregate and every raw record. An independent verifier
recomputes hashes, worker statistics, eligibility, paired shock ratios,
aggregates, structural ratios, and causal classifications without rerunning
timed work.

## Stop and claim boundary

Stop after both lanes, immutable records, independent verification, complete
configured tests, bounded interpretation, coherent local commits, and a clean
tree.

Do not optimize the CPU FV path, implement Phase 6E, add AOT/device-autonomous
execution, rerun favorable points outside the matrix, modify DVEB, rent
hardware, claim a universal compiler mechanism, claim publication readiness,
or push without explicit authorization.
