# FD/FV Phase-4R replication and causal-characterization result

Status: **CPU replication complete; strong `N=27` replication failed; CUDA
untested unavailable**.

Measurement date: 2026-08-27 UTC.

Measurement source commit:
`75cb329f948736a2513dafdac143de1479b2ef83`.

The immutable aggregate record is
`experiments/fd_fv_bakeoff/results/phase_4r_20260827/replication.json`,
SHA-256
`17d52fd8bf851d0fad87497857c5c30e6e3e378426a7ddeb57c2414d59d20fff`.
All 38 isolated CPU workers passed their numerical eligibility checks. The
verifier recomputes 4,840 timing samples and verifies the aggregate plus 38
raw records against the 39-entry manifest.

## Bottom line

The earlier Phase-4B `N=27^3` compiled FV result is a real observation in the
sense that similarly low FV timings appeared again in fresh processes. It is
not a robust performance result:

- the frozen strong-replication criterion failed;
- `N=27` was the only sampled size with process-level `FV/FD < 0.8`;
- FD was faster again at every sampled size from `N=30` through `N=48`;
- no generated-kernel, pre-fusion IR, vector-kernel, or source-code structural
  transition uniquely occurred at `N=27`; and
- the apparent FV advantage was accompanied by extreme multithread timing
  variance and disappeared in stable one-thread measurements.

The evidence therefore does not support a compiler-generated FV kernel that
is stably faster at `N=27`, a crossover regime beginning there, or a
mathematical FV advantage. The narrowest defensible interpretation is a
localized CPU multithread-runtime/scheduling instability. That is an
association, not an identification of the exact OpenMP, operating-system, or
hardware cause.

## Size replication

Each method/size value below is the median of independent worker medians for
one compiled SSP-RK3 step. There were two workers per method except three at
`N=27`. Times are milliseconds and `FV/FD > 1` favors FD.

| N | Cells | FD ms | FV ms | FV/FD | Frozen decision |
|---:|---:|---:|---:|---:|---|
| 18 | 5,832 | 1.105 | 1.258 | 1.139 | FD faster |
| 21 | 9,261 | 1.753 | 1.822 | 1.040 | unresolved |
| 24 | 13,824 | 2.364 | 2.527 | 1.069 | FD faster |
| 27 | 19,683 | 3.043 | 1.770 | 0.582 | FV faster, but unstable |
| 30 | 27,000 | 4.461 | 4.779 | 1.071 | FD faster |
| 33 | 35,937 | 3.430 | 4.020 | 1.172 | FD faster, but process-variable |
| 36 | 46,656 | 7.268 | 8.639 | 1.189 | FD faster |
| 40 | 64,000 | 10.095 | 11.768 | 1.166 | FD faster |
| 48 | 110,592 | 16.682 | 20.142 | 1.207 | FD faster |

The paired `N=27` ratios were `0.582`, `0.251`, and `0.475`. Two of three were
below `0.5`, but the process-level ratio was `0.582`, not below the frozen
`0.5` boundary. Phase 4R therefore records `n27_strong_replication=false`.
The weaker observation is localized: the only sampled transition point below
`0.8` is exactly `N=27`; no interpolation or persistent crossover is claimed.

`N=33` provides a second warning against reading process medians without their
distributions. Its two FV worker medians were `6.805 ms` and `1.235 ms` even
though both workers emitted the same compiler metrics and generated-code
size. That disagreement cannot represent a stable formulation cost.

## Compiler evidence

Across every size and independent worker, compiled FD emitted:

- 39 generated kernels;
- 36 generated C++ vector kernels;
- 228 pre-fusion IR nodes; and
- seven generated C++ files.

Compiled FV emitted 51 kernels, 47 C++ vector kernels, 240 pre-fusion IR nodes,
and seven C++ files everywhere. Recorded loop-reordering and auto-chunking
counts were zero. The compiler's bytes-accessed metric was zero and is treated
as non-informative rather than as a literal memory-traffic measurement.

Generated C++ size changes formed divisibility-related bands, but `N=27`
shared its approximately 398 kB FD / 396 kB FV band with `N=18`, `N=21`,
`N=30`, and `N=33`. At `N=27`, the main FD file contained 40 OpenMP pragmas
and 442 textual `loadu` occurrences; FV contained 52 and 472. Those counts are
formulation differences that persist within the band, not a unique `N=27`
transition. The frozen condition for a compiler-structural causal explanation
was therefore not met.

## Thread characterization

Primary six-thread `N=27` compiled FD workers were relatively narrow, with
coefficients of variation from `0.037` to `0.039`. FV's coefficients of
variation were `0.502`, `0.704`, and `0.614`; its latency samples were strongly
multimodal. A median from that distribution is not a stable operational
latency.

The diagnostic thread sweep timed the already-compiled function after the
primary samples:

- at one thread, FD medians were `9.14--9.52 ms` and FV medians were
  `9.88--10.13 ms`, with coefficients of variation at or below about 1%;
- at two and three threads, the distributions remained comparatively stable
  and FD was generally modestly faster;
- after returning to six threads, the process-level median was approximately
  `3.734 ms` for FD and `4.037 ms` for FV, so the primary FV advantage did not
  persist; and
- at twelve threads, two FV workers again produced sub-millisecond medians
  with coefficients of variation `0.683` and `0.719`, while the third produced
  `4.041 ms`. This is further evidence of a timing mode rather than stable
  computational throughput.

Changing the runtime thread count after compilation cannot, by itself,
separate a compiler schedule from OpenMP/runtime and operating-system state.
These observations falsify a stable single-thread kernel advantage and locate
the anomaly in the multithread execution path, but they do not prove its
lowest-level cause. Pinning workers to explicit physical-core and SMT affinity
sets would be the next bounded CPU diagnostic if that mechanism matters to a
future claim.

## Correctness and scope

Every worker retained one full graph with zero graph breaks, exact repeated
outputs, finite float64 CPU state, shape/device/dtype preservation, and
compiled/eager maximum absolute difference no greater than `2e-11`. No timing
cell failed admission. The result changes the performance interpretation, not
the mathematical qualification of either scalar method.

This is still one smooth, periodic, linear-advection problem using the
dimension-by-dimension formulations frozen in Phase 4. It does not establish
a general FD/FV result, nonlinear behavior, Euler behavior, or arbitrary-grid
performance.

## CUDA boundary

CUDA was not visible to the Phase-4A admission check in this environment.
Phase 4R therefore records `untested_unavailable` and collected no substitute
GPU measurement. The frozen CUDA stratum remains ready for a fresh eligible
machine; no RTX, A100, H100, or consumer-FP64 conclusion follows from this CPU
record.

## Revised Phase-4 interpretation

Phase 4B correctly preserved its original `2.69x` solve and `5.02x` step
observations. Phase 4R does not erase them; it resolves their evidentiary
weight. Similar fast FV modes can recur, but they are highly variable, fail
the preregistered strong-replication rule, disappear at stable low thread
counts, and do not extend to neighboring larger grids. They must not be used
as evidence that FV becomes faster in 3-D at `N=27`.

The robust result from the current CPU neighborhood is more modest: after the
localized unstable points, FD's process-level compiled median was
approximately `1.07x--1.21x` faster over `N=30--48`. Even that remains a
bounded result for this machine, compiler, formulation pair, and problem.
