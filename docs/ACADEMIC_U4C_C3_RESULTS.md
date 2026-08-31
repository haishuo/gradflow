# Academic U4-C C3 deployment-endpoint results

Status: **complete at the sole C2-admitted size**.

Date: 2026-08-30 (UTC)

`N=8192` was both the smallest and largest correctness-admitted C2 size. C3
therefore reports one size and does not substitute excluded larger grids.

## Transfer-inclusive CUDA endpoint

This endpoint begins with the frozen state in pageable CPU memory and includes
host-to-device transfer, one WENO-JS5 RHS, full device-to-host return, and
synchronization. State creation and compilation are outside the clock.

| lane | median (ms) | minimum (ms) | maximum (ms) |
|---|---:|---:|---:|
| OpenSBLI OPS CUDA | `0.043990` | `0.039790` | `0.047490` |
| compiled PyTorch/TorchInductor CUDA | `0.0686995` | `0.064100` | `0.092040` |

The descriptive median ratio `OpenSBLI / PyTorch` was `0.640325`, or about a
`1.56x` OpenSBLI advantage. Both returned full arrays passed the frozen C2
pointwise and conservation gates.

## AOT preparation and qualification

The fixed-shape float64 PyTorch AOTInductor package built successfully and
passed the same correctness gate (`maximum_normalized=2.51213e-14`,
`RMS_normalized=4.26528e-15`). Its package was 466,785 bytes with SHA-256
`fc6337f4bd7abc30779fdaa8229bc437395610d97a73c46dc605ffd12c173baf`.

The package's internal export and compilation took `0.197` and `6.821`
seconds, respectively (`7.156` seconds inside the builder; `9.243` seconds for
the complete builder process). The C3 OpenSBLI path took `0.651` seconds for
symbolic generation, `0.018` seconds for retained instrumentation, and `1.684`
seconds for OPS translation plus native CUDA build. These are one-off observed
preparation costs, not benchmark distributions.

## Prepared fresh-process launch to answer

Parent wall time starts before process creation and ends after a finite CPU
checksum of the full RHS is received. Prior artifact construction is excluded.

| prepared artifact | observations (s) | median (s) |
|---|---|---:|
| OpenSBLI native executable | `0.213531`, `0.214370`, `0.214216` | `0.214216` |
| PyTorch AOTInductor package | `1.467180`, `1.481119`, `1.461569` | `1.467180` |

The median ratio `OpenSBLI / PyTorch AOT` was `0.146005`, or about a `6.85x`
OpenSBLI launch-to-answer advantage. AOT removed JIT compilation from the run,
but it did not remove Python, PyTorch, CUDA-runtime, package-loading, and tensor
startup costs. At this small admitted grid those fixed costs dominate the
actual resident kernels.

## Combined interpretation

The three endpoints answer different questions and must not be collapsed:

- resident CUDA: OpenSBLI was about `3.23x` faster;
- pageable transfer-inclusive CUDA: OpenSBLI was about `1.56x` faster; and
- prepared fresh launch: OpenSBLI was about `6.85x` faster.

This is a useful external control, not a general verdict on PyTorch. It shows
that ordinary compiled PyTorch captures the matched nonlinear operator
correctly and reaches the same broad performance scale, while a generated
native external system retains meaningful kernel and deployment advantages on
this case. The larger-grid performance question remains open because the
prospective cross-implementation bounds excluded those cells.

## Evidence

Frozen evidence is in
`experiments/academic_u4c/evidence/u4c_c3_20260830/`. The architecture-specific
AOT package is retained outside the repository at the path recorded in
`endpoints.json`; its hash and build recipe are retained. Run
`python experiments/academic_u4c/verify_endpoints.py` for offline evidence
verification (the package itself is not required by the verifier).
