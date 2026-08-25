# Shu Euler WENO: Fortran versus direct PyTorch

This experiment asks when an ordinary-PyTorch GPU implementation becomes
faster than the repaired Jiang--Shu Fortran CPU implementation while keeping
the numerical work matched. It is an exploratory one-shot ablation, not a
publication benchmark: every grid has one measured solve, not 30 repetitions.

## Matched formulation

The 2-D PyTorch path translates the repaired ancestral program rather than the
scalar Gottlieb specialization. Both sides use:

- the compressible Euler conserved variables in IEEE float32;
- duplicated endpoints on the periodic `[0, 10]` grid and the Fortran
  boundary routine's endpoint-copy convention;
- finite-difference WENO-5 reconstruction in Roe characteristic fields;
- the original four-flux-difference nonlinear correction algebra;
- `epsilon = 1e-6`;
- a separate line-wise global Lax--Friedrichs speed for each characteristic
  family, enlarged by 10 percent;
- the original centered flux term; and
- one complete three-stage SSP-RK3 step with the same CFL calculation.

`shu_euler_torch.py` uses slicing, concatenation, elementwise operations, and
reductions. It contains no handwritten CUDA, Triton, C++, or custom operator.
TorchInductor compiles the same source used by eager PyTorch.

The 3-D extension is an experimental descendant, not an authentic source from
Professor Shu. It adds the z-momentum equation, the second shear
characteristic family, a z flux sweep, 5-by-5 Roe eigenvectors, and the 3-D CFL
sum. The matched test is the original isentropic vortex extruded uniformly in
z with zero z velocity. The z sweep is executed in full even though the exact
initial state is z-invariant.

## Correctness checks

The authoritative original remains untouched under `references/`. The 2-D
CPU comparator is the dynamically allocated, fixed-form-repaired descendant in
`experiments/fortran_scaling/`. The 3-D CPU comparator is the modern Fortran
extension in this directory.

After one complete step:

| Comparison | Grid intervals | float32 Linf |
|---|---:|---:|
| repaired 2-D Fortran vs PyTorch eager CPU | 10 x 10 | `4.768371582e-7` |
| 3-D Fortran extension vs PyTorch eager CPU | 6 x 6 x 6 | `7.152557373e-7` |
| PyTorch eager CPU vs compiled CUDA, 2-D | 40 x 40 | `4.768371582e-7` |
| PyTorch eager CPU vs compiled CUDA, 3-D | 8 x 8 x 8 | `2.384185791e-7` |

Both Fortran/PyTorch comparisons include initialization, the CFL-limited
timestep, all three RK stages, and the final duplicated-endpoint update.
Float64 autograd through the 2-D RHS also produced finite gradients. These are
bounded regression checks, not a general proof of the new 3-D mathematics.

Run the local CPU parity gate with:

```text
make
make PYTHON=/path/to/torch/python parity
```

## Timing definitions

Hardware was an AMD Ryzen 5 7600X CPU and NVIDIA GeForce RTX 5070 Ti (16,303
MiB), using GNU Fortran 13.3.0 and PyTorch 2.13.0+cu130. Fortran used
`-O3 -march=native` and one CPU thread. Solution-file output was disabled.

- **Fortran process** is wall time for a fresh process: input parsing,
  allocation, initialization, CFL calculation, and one full RK3/WENO step.
- **GPU step** is synchronized wall time for one compiled full RK3/WENO step.
- **GPU device-init + step** also includes initialization and CFL calculation
  performed directly on the GPU. It excludes Python import and compilation.
- **Transfer-inclusive estimate** sums separately measured one-shot CPU
  PyTorch preparation, pageable H2D transfer, the recorded GPU step, and D2H
  return of the complete state. It is an additive diagnostic, not a second
  solver run or a claim of overlapped production I/O.

Compilation is reported separately. A fresh observed 2-D dynamic compilation
took 31.61 seconds; a fresh 3-D fullgraph probe took 70.09 seconds. Cache state
and shape guards materially changed later compilation times. A 73.29-second
3-D shape-generalization compile contaminated the raw 8-cubed record; that
record is preserved but rejected, and a separately identified calibrated
8-cubed measurement replaces it in the tables below.

## One-shot 2-D results

Times are milliseconds. "Transfer speedup" is Fortran process time divided by
the transfer-inclusive estimate; values above one favor the GPU path.

| N x N | Fortran process | GPU step | GPU device-init + step | Transfer-inclusive | Transfer speedup |
|---:|---:|---:|---:|---:|---:|
| 32 | 1.841 | 0.663 | 1.282 | 2.110 | 0.87x |
| 64 | 2.086 | 0.613 | 1.032 | 1.515 | 1.38x |
| 128 | 4.847 | 0.772 | 1.441 | 1.701 | 2.85x |
| 256 | 13.855 | 0.951 | 1.493 | 2.431 | 5.70x |
| 512 | 51.094 | 3.658 | 4.243 | 10.228 | 5.00x |
| 1,024 | 273.872 | 14.526 | 15.608 | 44.061 | 6.22x |
| 2,048 | 1,471.507 | 58.596 | 61.219 | 186.646 | 7.88x |
| 4,096 | 6,626.866 | 235.055 | 246.261 | 864.876 | 7.66x |

The synchronized GPU step beat the Fortran process at every tested grid down
to the minimum valid N=4. At tiny grids, process-launch and initialization
noise is of the same order as the computation: one-shot device-init results
alternated around parity below N=32. With a complete CPU-preparation and PCIe
round trip included, **N=64 was the first measured GPU win**, and every larger
tested grid also won. Without the round trip, N=32 was the first clear and
sustained win.

At N=4,096 the compiled GPU step was 28.19 times faster than the Fortran
process; on-device initialization plus the step was 26.91 times faster. Even
after the deliberately pessimistic full-state round trip, the GPU path was
7.66 times faster. Peak CUDA allocation was 10,747,851,264 bytes, so larger
2-D cases rapidly approach this GPU's memory limit even though the CPU version
can allocate much larger host grids.

## One-shot 3-D results

| N cubed | Fortran process | GPU step | GPU device-init + step | Transfer-inclusive | Transfer speedup |
|---:|---:|---:|---:|---:|---:|
| 4 | 1.897 | 0.788 | 1.444 | 2.220 | 0.85x |
| 8 (corrected) | 2.961 | 1.138 | 2.055 | 2.596 | 1.14x |
| 12 | 5.026 | 1.175 | 1.872 | 1.774 | 2.83x |
| 16 | 8.904 | 0.998 | 1.462 | 1.531 | 5.81x |
| 24 | 27.567 | 1.032 | 1.497 | 1.811 | 15.23x |
| 32 | 54.407 | 1.534 | 2.256 | 2.675 | 20.34x |
| 48 | 173.055 | 3.899 | 4.821 | 5.890 | 29.38x |
| 64 | 398.140 | 9.138 | 10.033 | 16.653 | 23.91x |
| 96 | 1,323.557 | 29.851 | 31.256 | 56.877 | 23.27x |
| 128 | 3,154.830 | 71.716 | 73.372 | 142.503 | 22.14x |

The transfer-inclusive crossover was observed between N=4 and N=8; because
these are sub-3-millisecond single observations, **N=12 is the first convincing
3-D GPU win**. From N=24 through N=128, the GPU won by 15--29 times even with
the full-state PCIe round trip. The warm step-only advantage reached 43.99
times at N=128.

## What the result does and does not mean

Yes: once TorchInductor has compiled this direct formulation, the GPU really
is faster, and 3-D exposes a much larger throughput advantage. PCIe transfers
reduce the advantage but do not erase it beyond small grids.

A one-step job launched from cold is different. If the 31.61-second 2-D
compile is charged to one step, the CPU remains faster even at N=4,096. At
that grid the compile is amortized after roughly five timesteps. At N=2,048 it
takes roughly 23 timesteps. The fresh 70.09-second 3-D compile is amortized at
N=128 after roughly 23 timesteps. Real CFL-limited simulations normally use
many steps, but compile amortization must be stated rather than hidden.

The raw JSON is committed under `results/`. `rough_3d_2026-08-25.json` retains
the rejected recompile-contaminated N=8 observation;
`rough_3d_n8_corrected_2026-08-25.json` is its explicit replacement. No result
is an average, confidence interval, or stable publication number.

This comparison also does not establish a best-possible CPU result: it retains
the ancestral program's single-threaded line loops and does not add OpenMP,
SIMD-specific rewriting, or another multicore backend.

## Redistribution status

The PyTorch translation and 3-D Fortran extension are derived from the locally
preserved Jiang--Shu program. No license or public-redistribution permission
for that source was found in the supplied material. No new license conclusion
is inferred for these descendants; public release remains subject to the same
unresolved rights review recorded in `references/README.md`.
