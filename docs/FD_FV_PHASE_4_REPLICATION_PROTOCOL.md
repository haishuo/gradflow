# FD/FV Phase-4R replication and causal-characterization protocol

Status: frozen before new Phase-4R compilation, profiling, or timing.

Freeze date: 2026-08-27 UTC.

## Purpose

Phase 4R investigates the unexpected Phase-4B observation that, at `N=27^3`,
the compiled dimension-by-dimension FV step had a `0.575 ms` median versus
`2.887 ms` for classical FD, while smaller 3-D cells were mostly unresolved.
The Phase-4B record remains immutable.

The study asks:

1. does the `N=27` separation reproduce in fresh isolated processes;
2. at what nearby size does the separation appear or disappear;
3. is it associated with generated-kernel, fusion, vectorization, code-size,
   or runtime-thread behavior; and
4. does the same qualified comparison reproduce on CUDA when CUDA is visible?

No formulation is changed or optimized. No result may be described as a
mathematical FD/FV advantage merely because one compiler lowers the two
ordinary-PyTorch expressions differently.

## Frozen identities and admission

The numerical problem, float64 policy, methods, source projections, periodic
boundary, and SSP-RK3 step are exactly those in
`FD_FV_PHASE_4_PROTOCOL.md`. The Phase-4A and Phase-4B records must verify, and
the hashes of `problem.py`, `weno_js.py`, and `fv_weno5.py` must remain
unchanged.

Every worker checks eager/compiled parity at `rtol=0, atol=2e-11`, finite
output, device/dtype/shape preservation, and repeated deterministic output.
An ineligible cell is retained but excluded from timing conclusions.

## CPU size map and replication

The fixed 3-D isotropic sizes are:

```text
N = (18, 21, 24, 27, 30, 33, 36, 40, 48).
```

For each method and size, run two isolated worker processes. At `N=27`, run
three isolated workers. Every worker has a fresh TorchInductor cache, six
intra-operation threads, one inter-operation thread, and the inherited process
affinity recorded.

For one eager SSP-RK3 step use ten warmups and thirty measured repetitions.
For one compiled full-graph static SSP-RK3 step use ten warmups and fifty
measured repetitions. CPU timing uses `time.perf_counter_ns`; all raw samples,
median, mean, extrema, quartiles, median absolute deviation, and coefficient of
variation are retained.

The process-level value for a method/size is the median of the independent
worker medians. `FV/FD > 1` favors FD. Within 5% is unresolved.

The original `N=27` observation is replicated when at least two of three new
compiled workers are individually below `FV/FD=0.5` after pairing workers by
replicate index, and the process-level ratio is below `0.5`. A weaker ratio is
reported numerically but does not satisfy this frozen strong-replication rule.

The transition map reports the first and last sampled sizes whose process-level
compiled ratio is below `0.8`; it does not interpolate a crossover between
sampled sizes.

## Compiler evidence

Before compilation, record `torch._dynamo.explain` graph count, graph-break
count, operation count, and FX node count. Reset public Inductor metrics, make
the first compiled call, and record at least:

- generated kernel count;
- generated C++ vector-kernel count;
- IR nodes before fusion;
- compiler-estimated bytes accessed;
- outer-loop fused-inner counts;
- loop-reordering and auto-chunking counts; and
- first-call compilation-plus-execution latency.

Scan the fresh cache after compilation and record, without modifying it:

- generated `.cpp` file count, total bytes, lines, and SHA-256 identities;
- textual counts of `parallel_for`, OpenMP pragmas, vectorized types,
  `loadu`, `store`, and generated fused-kernel declarations; and
- total cache bytes and process peak RSS.

These are compiler diagnostics, not portable API guarantees. Missing metrics
are recorded as unavailable rather than invented.

After all timing, profile one compiled call and retain event names, counts, and
CPU-time totals. Profiling data never enters latency samples.

A causal compiler explanation requires a reproducible timing transition and a
corresponding method/size structural change in generated evidence. A mere
kernel-count or code-size difference supports “associated with,” not “caused
by.” Hardware-counter claims are prohibited because no frozen counter tool is
available in this environment.

## Runtime-thread characterization

In each `N=27` worker, after the six-thread primary measurements, time eager and
the already-compiled function with runtime intra-operation thread counts
`(1,2,3,6,12)`, keeping inter-operation threads at one. Each setting uses five
warmups and thirty repetitions. The original six-thread compiled samples remain
the primary replication; the thread sweep is diagnostic.

Changing runtime threads after compilation may or may not affect a schedule
baked into generated code. Therefore this sweep can show sensitivity or
insensitivity, but cannot by itself identify the compiler's scheduling cause.

## CUDA replication stratum

CUDA measurement is conditional on a fresh Phase-4A device admission on the
visible machine. If CUDA is not visible, Phase 4R records
`untested_unavailable` and performs no substitute GPU timing.

When admitted, CUDA uses device-resident float64 state at
`N=(18,27,40,64)`, three isolated workers per method/size, ten warmups and fifty
CUDA-event repetitions, explicit synchronization, fresh Inductor caches, and
peak allocated/reserved memory. Eager/compiled and CPU/CUDA parity are checked
before eligibility. Generated Triton/kernel metrics and device identity,
capability, driver/runtime, total memory, and FP64 hardware context are
recorded. Transfer-inclusive latency is outside this replication slice because
Phase 4R is testing the resident compiler observation.

## Records and decisions

The immutable aggregate and raw worker files contain complete samples,
eligibility, environment, hashes, compiler diagnostics, profiles, thread
sweeps, process-level ratios, transition brackets, and the strong-replication
decision. The verifier recomputes statistics, ratios, eligibility, raw hashes,
and source/admission identity.

Permitted conclusions distinguish:

- reproduced or not reproduced at `N=27`;
- stable, multimodal, or process-variable timing;
- compiler evidence associated or not associated with the transition;
- thread sensitivity observed or not observed; and
- CUDA reproduced, failed, or untested unavailable.

No universal FD/FV, arbitrary-grid, nonlinear, Euler, GPU, or publication
claim follows.

## Stop condition

Stop after CPU replication/characterization, conditional CUDA execution,
immutable records, verifier, interpretation, tests, coherent local commits,
and a clean working tree. Do not optimize either path, change DVEB, add a
nonlinear problem, begin Euler, or push without new explicit authorization.
