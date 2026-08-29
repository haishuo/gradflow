# G4 bounded profiler record

Date: 2026-08-29 (UTC)

Input: deterministic `N=128` unique-cell vortex, SHA-256
`05ff02323c83be003761e34809fc8168149b1434cb82d02dfd6fc7cd608ff70e`.

Profiler observations are post-campaign diagnostics and are excluded from the
randomized timing statistics.

## Nsight Systems

The system-default Nsight Systems 2022.4 executables completed collection but
could not import their QDSTRM files:

```text
Importer error status: The importer binary and its dependencies were not found.
Unable to retrieve the importer version: skipping importation of the QDSTRM file.
```

Those raw `nsys_*_n128_s1.qdstrm` files are preserved. The CUDA 13 installation
provided Nsight Systems 2025.3.2, which successfully produced the preserved
`nsys2025_*_n128_s1.nsys-rep` and SQLite exports.

Both lanes issued 17 numerical kernels: one CFL scan, one CFL finalization,
nine line-alpha kernels, three spatial kernels, and three stage-update kernels.

Kernel summaries from the `N=128`, one-step traces:

| Kernel class | Face-once total | Cell-recompute total |
|---|---:|---:|
| Dominant spatial kernel | 2.512 ms (`face_kernel`) | 7.349 ms (`rhs_kernel`) |
| Line-alpha reductions | 1.285 ms | 1.401 ms |
| Stage updates | 0.847 ms | 0.547 ms |
| CFL scan and finish | 0.036 ms | 0.042 ms |
| Sum of numerical kernels | 4.680 ms | 9.338 ms |

The face-owned spatial kernel is about 2.93 times faster than the
cell-recompute RHS kernel. Face-once pays back part of that gain in its update
kernel, which must difference the stored directional faces and is about 1.55
times slower than the control update. The complete traced kernel sum remains
about 2.00 times faster. This directly localizes the schedule effect without
attributing it to fewer RK stages, line reductions, or launches.

The exact generated CSV summaries are preserved as
`nsys_face_once_kernel_summary.csv` and
`nsys_cell_recompute_kernel_summary.csv`.

## Nsight Compute

Both the system-default Nsight Compute 2022.4 and the CUDA 13 Nsight Compute
2025.3.1 attempts were bounded to the Basic set and one `N=128`, one-step
process per lane. Both lane attempts failed with the same machine-permission
condition:

```text
ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

No kernels were counted and no `.ncu-rep` was produced. Permissions were not
changed. Consequently G4 has launch and kernel-duration evidence but no
hardware-counter claims about achieved occupancy, bandwidth, issue rate, or
cache behavior.
