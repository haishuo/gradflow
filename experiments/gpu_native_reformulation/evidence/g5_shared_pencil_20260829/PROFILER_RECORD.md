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

Nsight Compute 2025.3.1 Basic collection was attempted once. It returned:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

Permissions were not changed and no source was tuned after profiling.
