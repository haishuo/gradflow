# G5 bounded profiler record

Nsight Systems 2025.3.2 profiled one fresh P1 process at `N=128`, one SSP-RK3
step, zero warmups, and one internal CUDA-event observation. The input SHA-256
is `05ff02323c83be003761e34809fc8168149b1434cb82d02dfd6fc7cd608ff70e`.
The exact raw FP32 input is stored as `profiler_input_n128.f32.gz`; decompress
it before replaying the recorded profile command. Compression changes storage
only, not the hash above.

The trace contains 20 numerical launches: nine `pencil_kernel`, nine
`alpha_kernel`, one `cfl_kernel`, and one `finish_cfl_kernel`. Kernel totals:

| Class | Instances | Total ns | Median ns |
| --- | ---: | ---: | ---: |
| pencil | 9 | 11,433,242 | 1,473,584 |
| alpha | 9 | 1,287,405 | 178,786 |
| CFL scan | 1 | 34,720 | 34,720 |
| CFL finish | 1 | 864 | 864 |

The compiled pencil kernel reports 128 registers/thread, one barrier, and no
spills. Its dynamic shared memory is 2.5 KiB at this size.

In launch order, pencil durations repeat x, y, z for each RK stage. Summed by
direction they are 1,340,238 ns for x, 5,757,437 ns for y, and 4,335,567 ns
for z. This is consistent with coalesced x-pencil access and strided y/z
access in the frozen component-major `(z,y,x)` layout.

The initial unprivileged Nsight Compute 2025.3.1 Basic collection returned:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

An explicitly authorized one-time `sudo` replay was then run on the same
binary, input, and configuration. Forge's persistent driver permissions were
not changed. Nsight Compute collected nine passes for each of all 20 numerical
kernels. The final state checksum remained `7230127.1071257591`.

| Privileged artifact | SHA-256 |
| --- | --- |
| `ncu_privileged_p1_n128_s1.ncu-rep` | `163adf2dec17ea7e6137156759d0ff13a5d38ffd98a25c0cd16519f0257bf8ee` |
| `ncu_privileged_p1_details.csv` | `1b4e5f6bb4b21e814cbc5863e9564e7a33491d12c774a25dc9f504ae67326cc9` |
| `ncu_privileged_p1_summary.json` | `c0b710fd599a867d1d715840f21dd9c2325a72b753dd5ffe8a3d7505b96ff088` |

The pencil kernel is register-limited: 128 registers/thread permits two blocks
per SM, versus shared memory's four-block limit and the warp limit of six.
Theoretical occupancy is 33.33%. Axis medians and replay-duration sums:

| Axis | Three-launch duration (ms) | L2 | DRAM | L1/TEX | SM compute | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| x | 1.660 | 6.30% | 13.53% | 8.58% | 48.25% | 32.29% |
| y | 6.570 | 64.53% | 5.70% | 24.98% | 12.27% | 27.33% |
| z | 5.050 | 91.81% | 10.25% | 56.17% | 18.74% | 27.71% |

This confirms that y/z pencils stress cache-side memory service while the x
pencil remains substantially more compute-active. The low DRAM percentages
rule out off-card bandwidth saturation as the immediate bottleneck.

The application-reported 3.522 seconds is profiler replay overhead and is not
a benchmark observation. No source was changed or tuned in response.
