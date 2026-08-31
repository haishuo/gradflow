# Academic U5 stable-PyTorch replication results

Status: **complete on Forge with PyTorch 2.13.0**.

Date: 2026-08-31 (UTC)

## Decision

The central numerical and graph-capture findings reproduce on the current
stable release, but the performance surface does not reproduce quantitatively.
PyTorch 2.13 makes the generated CPU implementation substantially faster and
the CUDA implementation substantially slower than the dated development build
on this RTX 5070 Ti. The result narrows the stable float32 `64^3` GPU advantage
and removes a universal float64 GPU win at that shape. It also fixes the U4-F
batched CPU compiler assertion and reproduces the CUDA backend transition:
DVEB wins small batches while PyTorch wins the three largest batches.

This is evidence that compiler version is an experimental variable, not a
packaging detail. The paper must use the stable-release observations as its
primary toolchain result and retain the development-build comparison as a
version-sensitivity study.

## Environment and execution completeness

| Item | Stable U5 value |
|---|---|
| PyTorch | `2.13.0+cu130` |
| PyTorch commit | `cf30153c4c131c8164ee7798e5022d810682e2cb` |
| Triton | `3.7.1` |
| Python | `3.12.3` |
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0 |
| CUDA runtime | `13.0` |

- The complete A1 characterization executed.
- All 46 A2 configurations and all 90 eligible workers completed.
- Every eligible compiled A2 worker reported one graph and zero graph breaks.
- All three AOT packages built and passed JIT/AOT correctness.
- Both three-repetition deployment slices completed for all eight eligible
  endpoints.
- The complete A3 inverse, gradient, resolution, CPU, and CUDA campaign passed.
- All 12 U4-F device/batch cells passed correctness and were timed.
- The final CUDA-visible repository regression passed 355 tests, with 12
  expected external-ABI skips and one existing PyTorch deprecation warning, in
  53.81 seconds.

No canonical mathematics, tolerance, shape, warmup, sample count, or compiler
option changed.

## Numerical reproduction

The A1 order set, roundoff sweeps, and epsilon sweeps reproduced exactly.
Selected NumPy/LAPACK-derived condition diagnostics differed by at most
`4.70e-7` relative, while exact coefficient bit widths and all WENO numerical
outputs retained their conclusions. This is a conditioning-diagnostic
implementation difference, not a change to generated rational coefficients.

A2 retained 34 correctness exclusions versus 35 previously. The only removed
exclusion was characteristic WENO-11 float64 CUDA eager at `32^3`; no new
exclusion appeared. The important float32 one-dimensional CUDA boundary
persisted. Fast but inadmissible lanes remain excluded.

## Stable arbitrary-order performance

The table reports the fastest admitted CPU and resident CUDA lane for the
scalar Burgers `64^3` RHS. CPU may use one or six threads according to the
frozen matrix. CUDA may be eager or compiled when that is the faster admitted
lane.

| Order | Dtype | CPU ms | CUDA ms | CPU/CUDA | Stable CUDA lane |
|---:|:---:|---:|---:|---:|:---|
| 5 | f32 | 3.690 | 0.838 | 4.40x | compiled |
| 7 | f32 | 6.293 | 0.669 | 9.41x | compiled |
| 9 | f32 | 9.284 | 1.238 | 7.50x | compiled |
| 11 | f32 | 10.944 | 1.312 | 8.34x | compiled |
| 13 | f32 | 16.344 | 1.059 | 15.43x | compiled |
| 15 | f32 | 14.810 | 2.394 | 6.19x | compiled |
| 5 | f64 | 4.957 | 2.745 | 1.81x | eager |
| 7 | f64 | 9.383 | 5.238 | 1.79x | eager |
| 9 | f64 | 16.200 | 9.170 | 1.77x | eager |
| 11 | f64 | 16.452 | 14.284 | 1.15x | compiled |
| 13 | f64 | 29.766 | 22.079 | 1.35x | eager |
| 15 | f64 | 29.336 | 31.776 | **0.92x** | eager |

At this shape, stable float32 resident speedups are `4.40x--15.43x`; including
pageable round trips they are `3.61x--13.13x`. Stable float64 spans
`0.92x--1.81x` resident and `0.92x--1.71x` with transfers. Thus WENO-15
float64 is an observed CPU win on this consumer GPU. The consumer-GPU FP64
hardware limitation still prevents extrapolation to A100/H100-class systems,
but it does not permit relabelling this local loss as a win.

The fixed float32 scale slice now places WENO-5's transfer-inclusive 3-D
crossover between `32^3` and `64^3`; at `32^3` resident CPU/CUDA are virtually
tied and transfers favor CPU. WENO-15 crosses between `16^3` and `32^3`.

Relative to the development build, the fastest `64^3` compiled CPU medians
fell to `0.29--0.72` of their former values. CUDA resident medians rose to
`1.33--3.76` of their former values. These sequential observations do not
isolate a single causal compiler change, but they decisively prohibit treating
the nightly timing table as stable-release performance.

Compilation also became more expensive at high order. Stable CUDA first-call
time rose from `6.29 s` at float32 WENO-5 to `113.33 s` at WENO-15, and from
`6.57 s` to `130.93 s` in float64.

## AOT and deployment

| Order | Build s | JIT ms | AOT ms | Warm decision |
|---:|---:|---:|---:|:---|
| 5 | 10.696 | 0.828 | 0.792 | unresolved |
| 11 | 31.506 | 1.302 | 1.267 | unresolved |
| 15 | 62.970 | 2.388 | 2.330 | unresolved |

Unlike the development build, no stable warm JIT/AOT pair reached the frozen
five-percent materiality threshold. AOT's demonstrated value remains removal
of runtime compilation, not a faster steady-state kernel.

For isolated empty caches at `64^3`, the best JIT launch-to-answer median was
`10.457 s` versus `5.799 s` for AOT at order 5 (`1.80x` reduction), and
`106.208 s` versus `5.782 s` at order 15 (`18.37x`). AOT remained beneficial,
but stable package loading/startup was slower than in the development-build
record. Both sets of measurements remain preserved.

## Differentiation reproduction

The mathematical A3 result reproduced exactly: autograd/LBFGS returned
`1.100000846459`, golden section returned `1.099999995120`, and the best
centered-difference gradient disagreement remained `4.90e-11` relative.
CPU and CUDA eager and compiled lanes remained admitted with one graph and zero
breaks.

| Device | Stable compile s | Development compile s | Stable compiled forward ms | Stable objective+gradient ms |
|:---|---:|---:|---:|---:|
| CPU | 508.286 | 504.493 | 2.430 | 4.940 |
| CUDA | 445.078 | 339.093 | 2.942 | 8.205 |

Stable CUDA warm differentiation improved relative to the old build, while
its first compile became slower. Stable CPU remained the faster compiled lane
for this small float64 inverse problem. The derivative-free solution is still
more accurate and much more appropriate for the demonstrated one-off solve;
differentiability is correct, not shown superior here.

## Stable U4-F backend regime

PyTorch 2.13 compiled every batched CPU shape as one graph. The previous
development-build scheduler assertion for batches above one did not reproduce.

| Batch | CPU PyTorch/DVEB | CPU decision | CUDA PyTorch/DVEB | CUDA decision |
|---:|---:|:---|---:|:---|
| 1 | 1.956 | DVEB | 4.431 | DVEB |
| 4 | 1.841 | DVEB | 1.960 | DVEB |
| 16 | 1.786 | DVEB | 1.184 | DVEB |
| 64 | 2.083 | DVEB | 0.907 | PyTorch |
| 256 | 2.189 | DVEB | 0.823 | PyTorch |
| 1024 | 2.581 | DVEB | 0.825 | PyTorch |

DVEB wins every tested one-thread CPU batch. On CUDA, DVEB wins through batch
16 and PyTorch wins at batches 64, 256, and 1024. The precise medians moved,
and batch 16 changed from unresolved to a DVEB win, but the large-batch
PyTorch transition reproduced. This is the strongest current evidence for
GradFlow's backend-neutral premise: backend choice depends on the submitted
surface and toolchain, and neither PyTorch nor DVEB is globally correct.

## Scientific interpretation

U5 strengthens four claims:

1. exact-generated WENO-JS order 5--15 remains numerically reproducible;
2. ordinary PyTorch still captures all registered forward and differentiated
   programs as one graph with zero breaks;
3. AOT still converts runtime compilation into preparation, although its warm
   kernel advantage is not material on stable PyTorch; and
4. the matched DVEB/PyTorch regime boundary survives a major PyTorch version
   change.

It weakens any claim based on a single PyTorch build's absolute speedup. CPU,
CUDA, compilation, AOT loading, and even the best admitted CUDA lane changed
materially. The paper can now make a stronger systems claim—toolchain version
is part of the method and must be frozen—but must use smaller, stable-release
performance claims.

U5 remains one-machine evidence. It does not replace second-machine
replication, datacenter-FP64 qualification, independent CFD review, or the
rights decision.

## Evidence

The complete records are under
`experiments/academic_u5/evidence/u5_20260831/`. They include the package and
driver environment, full A1/A2/A3/U4-F records, raw worker output, AOT package
hashes, stable/development comparison, commands, and a SHA-256 manifest.

Verify with:

```bash
python3 experiments/academic_u5/verify_u5.py \
  experiments/academic_u5/evidence/u5_20260831
```
