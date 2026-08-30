# G6 bounded profiler record

The frozen protocol selected `b256_r112`, the smallest primary geometric-mean
ratio, for comparison with the historical frozen R6Q binary at `N=128`, one
SSP-RK3 step, zero warmups, and one internal observation. Both lanes consumed
the same decompressed `profiler_input_n128.f32.gz`; its raw input SHA-256 is
`05ff02323c83be003761e34809fc8168149b1434cb82d02dfd6fc7cd608ff70e`.

## Nsight Systems 2025.3.2

| Kernel class | Frozen total (ns) | `b256_r112` total (ns) | Candidate / frozen |
| --- | ---: | ---: | ---: |
| face, 3 launches | 2,526,008 | 2,527,223 | 1.00048 |
| alpha, 9 launches | 1,295,468 | 1,293,580 | 0.99854 |
| update, 3 launches | 849,384 | 845,480 | 0.99540 |
| CFL, 1 launch | 35,393 | 34,816 | 0.98370 |
| finish CFL, 1 launch | 896 | 896 | 1.00000 |

This is direct evidence that the apparent fresh one-step advantage over the
historical binary is outside the numerical kernels. The rebuilt G6 binaries
query CUDA function/occupancy metadata before their counted event; the older
binary does not.

## Privileged Nsight Compute 2025.3.1 Basic

An explicitly authorized one-time `sudo` replay collected nine passes for all
17 numerical launches in each lane. Forge's persistent driver permissions
were not changed. Both final checksums were `7230127.1071257591`.

| Face metric | Frozen R6Q | `b256_r112` |
| --- | ---: | ---: |
| Registers/thread | 128 | 112 |
| Register block limit | 2 | 2 |
| Shared-memory block limit | 16 | 16 |
| Warp block limit | 6 | 6 |
| Theoretical occupancy | 33.33% | 33.33% |
| Median achieved occupancy | 32.39% | 32.44% |
| Median SM compute throughput | 72.99% | 73.18% |
| Median DRAM throughput | 16.44% | 16.42% |
| Median L2 throughput | 15.53% | 15.51% |
| Three-face replay duration | 3.17 ms | 3.16 ms |

The 16-register reduction does not admit a third resident block and does not
change occupancy or throughput materially. Nsight Compute's application times
of 2950.867 and 2950.767 ms include counter replay and are not benchmarks.

Raw `.nsys-rep`, `.sqlite`, `.ncu-rep`, logs, CSV exports, reduced JSON, exact
binaries, compiler logs, source snapshot, and compressed input accompany this
record. Their identities are fixed by `SHA256SUMS`.
