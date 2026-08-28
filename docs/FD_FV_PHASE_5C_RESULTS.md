# FD/FV nonlinear Phase-5C performance result

Status: **passed after prospectively frozen conservation resolution**.

The immutable timing campaign ran from source commit
`7b0a989a8cad1b02ebd5a67446e2336c4a25675a`. Its aggregate is
`experiments/fd_fv_nonlinear/results/phase_5c_20260828/benchmark.json`,
SHA-256
`e5b80ab950e20b50023f08426b5c2dea1f550aa77969dc3f119cdc5ff474ef3b`.

The timing-free Phase 5CR resolution ran from
`aaa61040687b47b5b3d4bd690ba69d0b8f2f9220`. Its record is
`experiments/fd_fv_nonlinear/results/phase_5cr_20260828/resolution.json`,
SHA-256
`534ec63e6af0ec226ef9a5fb599a1d6000d1c05dff0aa9739ae5dc0f579f6ebb`.

## Eligibility resolution

The initial campaign and its failed gate remain immutable. Twelve complete
CUDA records and eight cold CUDA records at `N=(81,162)` exceeded a bound that
allocated roundoff once for a solve containing 531 or 1,685 SSP-RK3 steps.

Phase 5CR was frozen before resolution and collected no timing. Fresh FD/FV,
CPU/CUDA, eager/compiled diagnostics established that:

- semidiscrete mass residuals were at most `8.91e-18`;
- eager and compiled one-step conservation passed the original bound;
- host tensor and `math.fsum` reductions agreed within `2e-16`;
- full-solve CPU/CUDA differences remained below `7.18e-14`;
- full-solve errors remained finite and oracle-consistent; and
- the largest accumulated-bound utilization was `0.00392`.

Every affected record passed the prospective bound

```text
B_accumulated = steps*(B_single - 2e-15) + 2e-15
mass_change/steps <= B_single.
```

The approximately linear CUDA drift is therefore consistent with roundoff
accumulated by repeated state updates, not a nonconservative spatial operator
or a reduction artifact. Phase 5CR reclassifies copies of the preserved
records; it changes no duration, sample, implementation, or initial eligibility
field.

## Primary accuracy-matched result

Times below are aggregate medians in milliseconds. Each warm cell has three
independent workers with three complete-solve samples per worker. The selected
mode was compiled at every warm target.

| L2 target | CPU FD N / ms | CPU FV N / ms | FV/FD | CUDA FD N / ms | CUDA FV N / ms | FV/FD |
|---:|---:|---:|---:|---:|---:|---:|
| `2e-5` | 36 / 6.758 | 24 / 3.428 | 0.507 | 36 / 14.469 | 24 / 6.690 | 0.462 |
| `3e-6` | 54 / 13.709 | 36 / 6.674 | 0.487 | 54 / 28.125 | 36 / 13.260 | 0.471 |
| `5e-7` | 81 / 27.764 | 54 / 13.894 | 0.500 | 81 / 54.606 | 54 / 25.791 | 0.472 |
| `1e-7` | 162 / 94.081 | 81 / 27.812 | 0.296 | 162 / 176.824 | 81 / 49.911 | 0.282 |
| `5e-8` | 162 / 94.081 | 81 / 27.812 | 0.296 | 162 / 176.824 | 81 / 49.911 | 0.282 |

FV is faster at every frozen achieved-error target on both devices, by
`1.97--3.38x` on CPU and `2.12--3.54x` on resident CUDA. This is chiefly an
accuracy-to-work result: on this smooth pre-shock Burgers problem, FV reaches
each target on a smaller grid because the classical FD WENO-JS5 path exhibits
its known critical-point accuracy degradation.

It is not evidence that an FV update is universally cheaper. In equal-grid
compiled resident-step diagnostics, CUDA FV was `8--15%` faster than FD at all
measured sizes. CPU was near-tied at the two smallest sizes, favored FV through
`N=8,192`, then favored FD by approximately `1.49x` at the two largest sizes.
The isolated CPU FV `N=32,768` slowdown is a compiler/runtime regime effect,
not a mathematical ordering.

Prepared input/output transfer changed the selected CUDA complete times by
less than approximately one percent at these tiny one-dimensional states. It
does not alter any target decision. Process RSS was dominated by the Python and
compiler runtime (approximately 1.16--1.21 GB), while selected CUDA allocation
was about 14 KB or 50.3 MB depending on the compiled shape regime. These peaks
do not support a pure FD/FV storage claim.

## Does CUDA win?

Not for the bounded physical-time solves. Resident CUDA was
`1.79--2.14x` slower than CPU at every selected target. In the cold
launch-to-host pilot, every fastest selection was eager and CPU was faster than
CUDA; compiled cold execution took roughly 7.15--11.34 seconds because JIT
compilation dominated. These are single cold observations, not replicated
deployment estimates. Packaged AOT remains `not_implemented`.

CUDA does win once the isolated resident step is large enough:

| Formulation | N | CPU compiled step | CUDA compiled step | CUDA speedup |
|---|---:|---:|---:|---:|
| FD | 2,048 | 0.1359 ms | 0.1276 ms | 1.066x |
| FD | 131,072 | 4.7371 ms | 0.7690 ms | 6.160x |
| FD | 524,288 | 19.6620 ms | 2.7732 ms | 7.090x |
| FV | 32,768 | 1.4293 ms | 0.2404 ms | 5.946x |
| FV | 131,072 | 7.0524 ms | 0.6837 ms | 10.315x |
| FV | 524,288 | 29.2366 ms | 2.4280 ms | 12.042x |

The prospectively replicated first qualifying points were `N=2,048` for FD
and `N=32,768` for FV. The FD result is only a marginal local crossing: CUDA
lost at `N=8,192` and was within the 5% indifference band at `N=32,768` before
winning strongly at `N>=131,072`. It must not be described as a permanent
saturation threshold. The FV crossing was large and remained large at the two
larger sampled states, although those larger points each have only the frozen
baseline worker.

## Interpretation and claim boundary

For this one smooth scalar nonlinear problem, accuracy matching changes the
answer: FV is the faster method even where its equal-grid CPU step can cost
more, because it requires fewer cells. For these complete one-dimensional
solves CPU is still the best device. For sufficiently large resident states,
CUDA amortizes launch overhead and becomes much faster, even in float64 on a
consumer RTX 5070 Ti with restricted FP64 throughput.

This is a bounded phase-diagram point, not a universal FD/FV verdict. It does
not cover shocks, systems, multidimensional reconstruction, characteristic
projection, boundary closures, dynamic wave speeds, data-center GPUs, mixed
precision, generated C++, DVEB, AOT deployment, or end-to-end application
work. No optimization was performed after seeing the timings, and Phase 5D was
not started.
