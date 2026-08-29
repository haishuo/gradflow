# FD/FV Euler Phase-6C performance result

Status: **verified bounded result**.

Measurement date: 2026-08-29 UTC.

Qualified measurement source: `2ad7867367098ca7048360e6cea949ceb067e944`.

## Decision

All 164 isolated worker records passed their declared numerical and execution
eligibility gates:

- 60 replicated smooth complete-solve workers;
- 48 resident-step workers, including conditional crossover replication;
- 24 cold launch-to-host pilots; and
- 32 Sod/Shu--Osher launch-to-host pilots.

The independent verifier recomputed checksums, raw-sample statistics,
eligibility, achieved-error selections, ratios, classifications, conditional
replication sizes, and device crossover decisions.

The result does not identify one universally superior discretization or
device. It identifies distinct regimes on this Ryzen 7600X, RTX 5070 Ti,
float64, one-dimensional Euler WENO-JS5 experiment.

## Smooth achieved-accuracy result

FD and FV achieved nearly identical error at every frozen grid, so both
methods selected the same `N` for every accuracy target. Compiled execution
won every warm selection.

| L2 target | Selected N | CPU FV/FD time | CUDA-resident FV/FD | Prepared CUDA FV/FD |
| ---: | ---: | ---: | ---: | ---: |
| 5e-6 | 24 | 1.124 | 1.189 | 1.194 |
| 1e-6 | 36 | 1.098 | 1.199 | 1.191 |
| 1e-7 | 54 | 1.106 | 1.225 | 1.215 |
| 1e-8 | 81 | 1.105 | 1.189 | 1.185 |
| 1e-9 | 162 | 1.122 | 1.134 | 1.139 |

Thus FV required 9.8--12.4% more warm CPU time and 13.4--22.5% more time on
the two CUDA boundaries at matched achieved error. Equivalently, selecting FD
reduced time by 8.9--11.0% on CPU and 11.8--18.3% on CUDA. This is the opposite of the
Phase-5C smooth Burgers result, where FV's accuracy advantage let it use fewer
cells. Together, the two phases are evidence against a universal FD/FV
hierarchy: the balance depends on the equation, solution, formulation, and
accuracy target.

The smooth complete-solve L2 errors were:

| N | FD | FV |
| ---: | ---: | ---: |
| 24 | 4.20800e-6 | 4.21100e-6 |
| 36 | 5.26686e-7 | 5.30551e-7 |
| 54 | 6.60053e-8 | 6.71091e-8 |
| 81 | 8.32294e-9 | 8.55454e-9 |
| 162 | 2.48834e-10 | 2.58672e-10 |

At these small complete-solve sizes, compiled CPU remained faster than
compiled CUDA. CUDA/CPU time ratios ranged from `1.42` to `1.67` for FD and
`1.43` to `1.76` for FV. Every adaptive solve included its global-CFL scalar
synchronization with host control; none is mislabeled device-autonomous.

## Compilation and cold latency

Warm compiled solves took approximately 9--83 ms on CPU and 15--119 ms on
CUDA. First compiled complete solves instead had median durations of roughly
15.0--15.7 seconds on CPU and 7.2--7.7 seconds on CUDA because every worker
used a fresh TorchInductor cache.

The cold pilot includes process startup and imports. At `N=24--162`:

- CPU eager took `1.03--1.67 s`;
- CUDA eager took `1.42--2.63 s`;
- CPU compiled took `18.20--19.46 s`; and
- CUDA compiled took `11.12--11.88 s`.

Consequently, warm JIT speed is not the one-shot small-problem answer. CPU
eager is the appropriate observed cold endpoint in this bounded matrix.
Packaged AOT remains `not_implemented`; these results strengthen its value
proposition but do not measure it.

## Large-state resident-step result

The compiled resident SSP-RK3 step produced two independently verified
crossovers:

- FD: confirmed at `N=32,768`, with three CUDA/CPU ratios
  `0.7742`, `0.7759`, and `0.7720`;
- FV: confirmed at `N=8,192`, with ratios `0.3128`, `0.4304`, and `0.4227`.

Selected compiled medians and CUDA speedups were:

| N | FD CPU | FD CUDA | FD speedup | FV CPU | FV CUDA | FV speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.423 ms | 0.536 ms | 0.79× | 0.448 ms | 0.547 ms | 0.82× |
| 8,192 | 0.822 ms | 1.152 ms | 0.71× | 2.900 ms | 1.226 ms | 2.37× |
| 32,768 | 4.604 ms | 3.572 ms | 1.29× | 14.573 ms | 3.681 ms | 3.96× |
| 131,072 | 20.655 ms | 13.089 ms | 1.58× | 52.957 ms | 13.594 ms | 3.90× |
| 524,288 | 122.999 ms | 14.114 ms | 8.71× | 227.162 ms | 16.650 ms | 13.64× |

FV's earlier device crossover must not be read simply as “FV maps better to a
GPU.” Its compiled CPU path develops a replicated large-size penalty: FV/FD
CPU time is `3.53` at `N=8,192` and `3.17` at `N=32,768`, while the
corresponding CUDA ratios are only `1.06` and `1.03`. This points to a
representation/code-generation regime in the current CPU implementation. It
is a causal follow-up question, not a reason to alter Phase 6C after observing
the result.

At `N=524,288`, FV remained 18.0% slower than FD on CUDA, whereas it was 84.7%
slower on CPU. Both GPU paths nevertheless achieved large speedups despite the
RTX 5070 Ti's restricted FP64 hardware.

## Shock application pilot

Every shock pilot retained positive stages and passed its applicable inherited
oracle gates. These are single launch-to-host observations and cannot support
a close-ratio claim.

At `N=200`, CPU eager was fastest:

| Problem | FD | FV |
| --- | ---: | ---: |
| Sod | 3.115 s | 3.091 s |
| Shu--Osher | 5.141 s | 5.052 s |

At `N=800`, compiled CUDA was fastest:

| Problem | FD CUDA compiled | FV CUDA compiled | FD CPU eager | FV CPU eager |
| --- | ---: | ---: | ---: | ---: |
| Sod | 13.933 s | 13.928 s | 17.147 s | 17.318 s |
| Shu--Osher | 16.157 s | 16.093 s | 32.925 s | 33.099 s |

Relative to CPU eager, compiled CUDA was about 1.23× faster on Sod and 2.04×
faster on Shu--Osher for FD, and about 1.24× and 2.06× faster for FV. Because
each is one fresh-process observation, the near equality of FD and FV CUDA
times is unresolved. The CPU-to-GPU regime change is strong enough to justify
prospective replication.

FD used 6,892 Shu--Osher `N=800` steps while FV used 6,894 because each method
computed its own qualified adaptive CFL sequence. This is real complete-solve
behavior, not an equal-step substitution.

## Memory and hardware boundary

At the strictest smooth target, peak CUDA allocated memory was approximately
50.5 MB for both methods. Peak process RSS was approximately 1.35 GB, dominated
by interpreter/compiler state and therefore not a pure numerical working-set
measure.

The accelerator was Forge's NVIDIA GeForce RTX 5070 Ti under PyTorch
`2.13.0+cu130` and CUDA runtime 13.0. All results are float64. They must remain
bounded by the device's intentionally restricted consumer FP64 throughput;
they neither predict nor substitute for A100/H100 measurements.

## Reproducibility

Canonical aggregate:

`experiments/fd_fv_euler/results/phase_6c_20260829/benchmark.json`

Hashes:

- `benchmark.json`:
  `a44eec63c1a37b7f0aff92a2aedb4a56c46b61813c17e9c59527f02a7ad56e33`
- `SHA256SUMS`:
  `49784336f9cd736c27c1360c3f284faca2e5850f742a526e417781a05f0c5ddf`

Verify without rerunning timing:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6c.py
```

The final configured Forge suite, including CUDA and the declared read-only
DVEB fixtures, passed: `345 passed`, with 14 upstream PyTorch deprecation
warnings.

## Scientific interpretation and next boundary

Phase 6C provides three paper-relevant evidence components, but not yet a
publishable result by itself:

1. a matched Euler counterexample to the Burgers result, showing that FD/FV
   accuracy-to-time rank is problem-dependent;
2. independently replicated formulation-specific GPU crossover regions; and
3. a realistic shock pilot showing a process-entry-to-answer GPU advantage at
   `N=800`, even with compilation and consumer FP64 limitations.

The next prospectively frozen work should separate three questions:

- replicate the promising `N=800` shock launch-to-host result;
- causally characterize the compiled CPU FV regime beginning near `N=8,192`;
- evaluate a qualified device-autonomous/AOT execution boundary that removes
  repeated adaptive-CFL host synchronization and runtime compilation.

Those are not Phase-6C post-hoc optimizations. No Phase 6D, AOT implementation,
DVEB work, multidimensional extension, or data-center rental began here.
