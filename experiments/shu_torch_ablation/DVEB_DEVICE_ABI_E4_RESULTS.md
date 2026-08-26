# DVEB device-resident ABI v2 E4 results

Status: **complete; the frozen correctness and timing gates passed**.

## Result

DVEB v2 wins all ten counted resident-state points and all 60 randomized worker blocks. It is the only lane within 10% of the winner at every point. Across the frozen points it is 2.53--7.36 times faster than packaged AOTInductor. This qualifies DVEB for the fixed Shu Euler 3-D WENO-5 E4 region on this machine; it does not qualify automatic placement or other programs.

## Correctness

The full-array gate passed with worst absolute error `1.430511475e-06` against `2e-5`. The non-default-stream exact-alias check also passed.

## Counted E4 medians

| N | Steps | DVEB v2 ms | AOT ms | compile ms | eager ms | DVEB/AOT speedup | DVEB block wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 0.130 | 0.328 | 0.497 | 9.307 | 2.53x | 6/6 |
| 16 | 1 | 0.130 | 0.383 | 0.496 | 9.377 | 2.95x | 6/6 |
| 32 | 1 | 0.284 | 0.972 | 1.001 | 9.537 | 3.43x | 6/6 |
| 64 | 1 | 1.256 | 8.569 | 8.591 | 39.820 | 6.82x | 6/6 |
| 96 | 1 | 4.017 | 29.180 | 29.310 | 149.470 | 7.26x | 6/6 |
| 128 | 1 | 9.638 | 69.790 | 69.972 | 425.656 | 7.24x | 6/6 |
| 16 | 10 | 1.143 | 3.695 | 4.610 | 93.987 | 3.23x | 6/6 |
| 32 | 10 | 2.665 | 9.621 | 9.620 | 95.796 | 3.61x | 6/6 |
| 64 | 10 | 12.312 | 85.393 | 85.432 | 397.687 | 6.94x | 6/6 |
| 128 | 10 | 94.910 | 698.131 | 698.665 | 4256.768 | 7.36x | 6/6 |

Every cell is the median of 30 wall-clock calls: six independent workers, five warmups, then five counted calls. There were 1,200 counted calls and zero failed workers.

## ABI overhead

The wall timer surrounds the public Python-to-C ABI call; DVEB's native internal host-wall total is diagnostic. At `N=128`, one step, those medians are 9.638 ms and 9.626 ms respectively (0.13% wall overhead). For ten steps they are 94.910 ms and 94.892 ms (0.02%). The public device ABI therefore reaches the previously observed native-CUDA performance floor at the largest counted grid without a material abstraction penalty.

Protocol erratum: the frozen protocol called the internal field a CUDA-event time. Inspection of the qualified source confirms it is a monotonic host-wall timer spanning D2D input, kernels, D2D output, and stream synchronization. No result depends on treating it as the primary timer.

After timing, the combined v1/v2 suite exposed and fixed a v1-only odd-step allocation-handle bug at DVEB commit `1e7fec3`. The v2 numerical path was unchanged. One uncounted post-fix sentinel block at N=128 measured medians of 9.612 ms for one step and 94.573 ms for ten steps, within 0.4% of the frozen campaign. These sentinels confirm continuity but do not replace the frozen 30-observation results.

## Interpretation

The old conclusion that DVEB could not participate in E4 was an ABI-v1 limitation, not a CUDA-code-generation limitation. ABI v2 removes the mandatory host round trip while preserving caller ownership, streams, and reusable workspace. On this workload, generated DVEB CUDA is materially faster than the three ordinary-PyTorch resident formulations tested.

This does **not** make DVEB a universal GradFlow backend. E1--E3 remain as previously reported, and this addendum says nothing about automatic selection, arbitrary WENO order, Navier--Stokes, different boundary conditions, FP64, other GPUs, or end-to-end application latency.
