# FD/FV Euler Phase-6D replication and causal result

Status: **verified bounded result; shock confirmation unresolved**.

Measurement date: 2026-08-29 UTC.

Timing source: `7952c9fabbca5114994d457b563cc907c477db4e`.

Aggregation correction: `889fbc321f925c66fc12f2c863f8f8cb08a56c77`.

## Decision

Phase 6D completed all 92 prospectively frozen isolated workers:

- 24 complete `N=800` Sod/Shu--Osher CPU/CUDA replications; and
- 68 CPU thread/size/code-generation characterization workers.

All CPU characterization records passed their float64, finite, deterministic,
shape, and eager/compiled parity gates. Every shock solve completed, retained
positive states, and passed the inherited physical/oracle gates. Seven CUDA
shock workers nevertheless failed the additional Phase-6D requirement that
the terminal tensor hash exactly reproduce Phase 6C. The shock lane therefore
does not confirm a CUDA win under the frozen correctness-first rule.

The CPU result reproduces a substantial compiled FV penalty but rejects the
pilot interpretation of one clean size-triggered compiler transition. None of
the three frozen causal classifications is uniquely supported. The registered
decision is `unresolved_mixture`.

## Shock replication

The observed launch-to-host medians and descriptive ratios were:

| Problem | Method | CPU endpoint | CPU median | CUDA compiled median | CUDA/CPU |
| --- | --- | --- | ---: | ---: | ---: |
| Sod | FD | eager | 17.175 s | 13.984 s | 0.814 |
| Sod | FV | eager | 17.192 s | 14.143 s | 0.823 |
| Shu--Osher | FD | compiled | 21.290 s | 16.105 s | 0.756 |
| Shu--Osher | FV | compiled | 22.585 s | 16.400 s | 0.726 |

Numerically, these medians would correspond to CUDA speedups of `1.21--1.38x`
over the selected CPU endpoints. They are not confirmed performance claims:
the protocol required all three paired workers to be eligible, and eligibility
included an exact Phase-6C terminal hash match.

CPU terminal hashes reproduced Phase 6C exactly in all 12 workers. CUDA exact
hash reproduction was:

| Problem | Method | Exact matches |
| --- | --- | ---: |
| Sod | FD | 1 / 3 |
| Sod | FV | 2 / 3 |
| Shu--Osher | FD | 0 / 3 |
| Shu--Osher | FV | 2 / 3 |

The mismatched runs still passed every inherited physical/oracle gate. Their
reported minima and Shu--Osher feature metrics differed only in trailing
float64 digits where variants occurred. The preserved worker format does not
contain terminal arrays, so Phase 6D cannot retroactively compute a norm of
the cross-process difference. Exact-hash instability is therefore a new
correctness/reproducibility question, not permission to infer that the arrays
are either materially wrong or acceptably close.

This supersedes the Phase-6C single-observation shock wording: a process-entry
CUDA advantage remains promising in the observed timings, but is unconfirmed
until the terminal variation is explained and prospectively bounded.

## CPU compiled regime

Compiled FV/FD median-time ratios were:

| N | 1 thread | 6 threads | thread-interaction factor |
| ---: | ---: | ---: | ---: |
| 2,048 | 0.926 | 1.037 | 1.120 |
| 4,096 | 2.729 | 3.882 | 1.423 |
| 6,144 | 2.908 | 4.297 | 1.478 |
| 8,192 | 2.971 | 4.859 | 1.635 |
| 12,288 | 2.958 | 5.021 | 1.698 |
| 16,384 | 2.991 | 3.660 | 1.224 |
| 24,576 | 3.014 | 3.298 | 1.094 |
| 32,768 | 2.997 | 3.238 | 1.081 |

The primary observations are:

1. The FV penalty is reproducible as a broad compiled CPU regime. With one
   thread it is tightly near `3x` from `N=4,096` onward, rather than beginning
   uniquely at `N=8,192`.
2. Eager FV/FD ratios remain close to one. The separation is specific to these
   TorchInductor CPU graphs, not the mathematical operation count as observed
   by eager PyTorch alone.
3. Six-thread behavior adds a size-dependent penalty around `N=8,192--12,288`,
   but the replicated interaction rule fails. At `N=8,192`, replicate factors
   were `1.639`, `1.548`, and `1.466`; at `N=32,768`, they were `2.597`,
   `0.817`, and `1.098`.
4. More threads were not monotonically better for FV. For example, the
   `N=8,192` compiled FV medians were 5.596 ms at two threads, 2.208 ms at
   four, and 4.039 ms at six.

At `N=32,768`, one-thread FV worker medians were 17.733, 42.934, and
42.388 ms even though all three workers had identical compiler metrics,
generated-source structural totals, and generated-source hashes. This is
evidence that fresh-process runtime variability contributes materially; it
cannot be explained by a generated-code structure change in the recorded
signals.

## Frozen causal tests

The compiler structures were constant across every measured size within a
method/thread regime:

| Metric | FD | FV | FV/FD |
| --- | ---: | ---: | ---: |
| Generated kernels | 72 | 94 | 1.306 |
| Generated vector kernels | 43 | 52 | 1.209 |
| Pre-fusion IR nodes | 264 | 306 | 1.159 |

The generated six-thread C++ inventories contained 69 OpenMP pragmas for FD
and 90 for FV, but the recorded `parallel_for` marker count was zero. The
TorchInductor `num_bytes_accessed` counter was zero for these CPU graphs, so
the estimated-byte ratio is explicitly unavailable rather than interpreted
as zero traffic. This instrumentation edge case was documented in
`FD_FV_PHASE_6D_PROTOCOL_AMENDMENT.md`; no timed worker was rerun.

The prospective classifications resolve as follows:

- `thread_interaction_supported = false`: adjacent aggregate points met the
  threshold, but the required replicate consistency did not;
- `traffic_expansion_supported = false`: the available IR-node ratio was
  `1.159`, below the frozen `1.5` threshold, and byte estimates were
  unavailable;
- `codegen_transition_supported = false`: no FV-only structural signature
  changed where the slowdown appeared; and
- `classification = unresolved_mixture`: no unique registered mechanism was
  supported.

This does not mean that kernels, traffic, or threading are irrelevant. It
means the available aggregate counters and timing replications do not identify
one of them as the unique cause under the rules frozen before measurement.

## Reproducibility

Canonical aggregate:

`experiments/fd_fv_euler/results/phase_6d_20260829/benchmark.json`

Hashes:

- `benchmark.json`:
  `8b3010b7ad42a7830134624dab660f4ed3c48b0d05385f6ebd1565c8f21ad58b`
- `SHA256SUMS`:
  `8aed5a0dc335c49178fea531153151f35a21b42145e55efd3a0371995564bddc`

The aggregate records both the original timing-source hashes and the later
aggregation-source hashes. The independent verifier checks those git blobs,
all 92 raw records, checksums, timing statistics, eligibility, aggregates,
shock decisions, and causal classifications without rerunning timing:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6d.py
```

The final configured Forge suite, including CUDA and the declared read-only
DVEB fixtures, passed: `349 passed`, with 14 upstream PyTorch deprecation
warnings.

## Scientific interpretation and next boundary

Phase 6D strengthens two claims and blocks one:

- the current compiled CPU FV implementation has a real broad performance
  deficit relative to matched FD on this machine;
- that deficit is not explained by a unique size-local code-generation
  transition in the recorded evidence; and
- the apparent complete-shock CUDA latency advantage cannot yet pass the
  project's exact reproducibility gate.

Correctness remains above performance. Before a device-autonomous or AOT
Phase 6E can use these shock timings as evidence, a prospectively frozen gate
must retain terminal arrays (or a sufficiently strong independent numerical
comparison), characterize CUDA cross-process determinism, and decide an
accuracy tolerance from mathematical evidence rather than from the observed
timing result. A lower-level CPU profiling phase may separately examine the
stable `~3x` single-thread deficit and the nonmonotonic multithread behavior.

No production source, DVEB source, optimization, AOT path, device-autonomous
loop, or Phase 6E implementation was begun. This result is bounded to the
Ryzen 7600X, RTX 5070 Ti, PyTorch `2.13.0+cu130`, float64, one-dimensional
ideal-gas Euler WENO-JS5 experiment. It is not a universal FD/FV, CPU/GPU, or
compiler claim and is not a publication claim.
