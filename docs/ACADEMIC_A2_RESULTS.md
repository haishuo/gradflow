# Academic A2 arbitrary-order performance results

Status: **complete on the primary Forge machine**.

Date: 2026-08-30 (UTC)

## Result in one paragraph

Maintainable exact-generated PyTorch WENO-JS compiled successfully as one
full graph with zero graph breaks for all 90 protocol-eligible CPU/CUDA
workers. For the scalar `64^3` RHS, admitted compiled CUDA was substantially
faster than the fastest one- or six-thread compiled CPU observation at every
order and dtype: `22.9x--44.0x` resident (`12.7x--35.8x` including pageable
round trips) in float32 and `5.6x--9.4x` resident (`5.4x--8.3x` including
round trips) in float64. The corresponding one-dimensional float32 study did
not establish a GPU win: CPU won every admitted scale point, and many CUDA
outputs at small spacing failed the frozen cross-device roundoff tolerance.
Prepared AOT removed compilation from invocation and made clean-cache
launch-to-answer latency `4.7x` lower than JIT for WENO-5 and `42.7x` lower
for WENO-15 at `64^3`, while warm AOT kernel improvements were modest and
order-dependent. Higher-order characteristic Euler exposed strict
conservation/roundoff limits and therefore supplies failures, not performance
claims. These are bounded results on one Ryzen 7600X / RTX 5070 Ti machine.

## Execution and audit completeness

- All 46 registered mathematical configurations completed.
- All 90 protocol-eligible device workers completed; no worker process failed
  and no OOM occurred.
- One accidentally launched characteristic `64^3` CPU worker is retained with
  `protocol_eligible=false` and excluded.
- Every compiled worker reported one unique graph and zero graph breaks.
- Five warmups and 20 randomized eager/compiled pairs were retained per
  admitted warm endpoint; no outlier was deleted.
- Three prepared AOT packages built and passed parity.
- Both prepared-cache and isolated-empty-cache fresh-process slices completed.
- The analysis retains 35 endpoint exclusions produced by the frozen
  correctness and reference-health gates.

The protocol and the two scope corrections are in
`ACADEMIC_A2_PROTOCOL.md` and `ACADEMIC_A2_HARNESS_CORRECTION.md`.

## Machine and software

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 5 7600X, 6 cores / 12 hardware threads |
| GPU | NVIDIA GeForce RTX 5070 Ti, compute capability 12.0 |
| Driver | 580.173.02 |
| Operating system | Linux 6.8.0-138-generic, x86-64, glibc 2.39 |
| Python | 3.11.13, conda-forge build |
| PyTorch | 2.9.0.dev20250705+cu128 |
| PyTorch CUDA runtime | 12.8 |
| CPU thread observations | 1 and 6 PyTorch threads; one interop thread |

This consumer GPU's float64 hardware is not representative of an A100/H100.
A2 measures the machine; it does not attribute all float64 behavior to the
algorithm or extrapolate to data-center hardware.

## Scalar cross-order core

The table reports milliseconds for one scalar Burgers semidiscrete RHS at
`64^3`. `CPU` is the faster admitted compiled result from one or six threads.
`CUDA resident` uses an already resident state and CUDA events. `CUDA + copy`
adds pageable host-to-device input and device-to-host output. Speedups divide
the CPU median by the corresponding CUDA median. These are separate device
samples, not paired cross-device confidence intervals.

| Order | Dtype | CPU ms | CUDA resident ms | CPU / CUDA | CUDA + copy ms | CPU / CUDA + copy |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 5 | f32 | 5.100 | 0.223 | 22.85x | 0.402 | 12.69x |
| 7 | f32 | 10.821 | 0.251 | 43.07x | 0.425 | 25.45x |
| 9 | f32 | 23.233 | 0.528 | 43.99x | 0.713 | 32.59x |
| 11 | f32 | 24.813 | 0.611 | 40.59x | 0.805 | 30.81x |
| 13 | f32 | 35.004 | 0.795 | 44.02x | 0.979 | 35.77x |
| 15 | f32 | 51.016 | 1.751 | 29.14x | 1.930 | 26.44x |
| 5 | f64 | 8.793 | 1.328 | 6.62x | 1.590 | 5.53x |
| 7 | f64 | 17.542 | 1.860 | 9.43x | 2.125 | 8.25x |
| 9 | f64 | 45.045 | 5.413 | 8.32x | 5.681 | 7.93x |
| 11 | f64 | 43.602 | 6.828 | 6.39x | 7.106 | 6.14x |
| 13 | f64 | 51.166 | 9.210 | 5.56x | 9.475 | 5.40x |
| 15 | f64 | 77.374 | 10.402 | 7.44x | 10.681 | 7.24x |

All scalar `64^3` lanes in this table passed the frozen parity and conservation
gate. The nonmonotone order-to-order increments are measured compiler
outcomes, not evidence that mathematical work decreases with order.

### Scaling, memory, and compilation

For float32 `64^3`, CUDA first-call wall time rose from `1.103 s` at order 5
to `75.251 s` at order 15, while the warm compiled RHS rose from `0.223 ms` to
`1.751 ms`. Incremental compiled peak allocation rose from `40.9 MB` to
`139.5 MB`. The float64 endpoints were `1.328--10.402 ms` and
`81.8--278.9 MB`.

Compiled execution won every one of the 48 frozen CPU eager/compiled paired
comparisons. It also won all 18 cross-order CUDA pairs in which both lanes
were correctness-admitted. Six CUDA pairs were not decided because the
compiled float32 1-D lane was excluded. Six CPU threads did not produce a
general improvement over one thread: for the cross-order compiled core, the
six/one-thread median ratio ranged from `0.979` to `1.273`, with most values
near one and the largest regressions at orders 7 and 11 in `64^3`. A2 therefore
does not claim useful CPU thread scaling for this generated implementation.

## Float32 scale and crossover slice

Each entry is the fastest correctness-admitted warm lane on that device.
CUDA/CPU below one favors CUDA. `+copy` includes both pageable transfers.

| Order | Grid | CPU ms | CUDA resident ms | CUDA/CPU | CUDA + copy ms | CUDA+copy/CPU |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 5 | 1-D, 128 | 0.0346 | 0.1005 | 2.91 | 0.1189 | 3.44 |
| 5 | 1-D, 512 | 0.0403 | 0.6734 | 16.71 | 0.6980 | 17.33 |
| 5 | 1-D, 2,048 | 0.0564 | 0.6617 | 11.73 | 0.6905 | 12.24 |
| 5 | 1-D, 8,192 | 0.1267 | 0.6749 | 5.33 | 0.7056 | 5.57 |
| 5 | 1-D, 32,768 | 0.4002 | 0.6647 | 1.66 | 0.7086 | 1.77 |
| 5 | `16^3` | 0.1258 | 0.1907 | 1.52 | 0.2173 | 1.73 |
| 5 | `32^3` | 0.5786 | 0.2112 | 0.365 | 0.2589 | 0.448 |
| 5 | `64^3` | 5.0998 | 0.2232 | 0.0438 | 0.4019 | 0.0788 |
| 5 | `96^3` | 20.1740 | 0.7112 | 0.0353 | 1.0924 | 0.0541 |
| 15 | 1-D, 128 | 0.0913 | 0.3303 | 3.62 | 0.3517 | 3.85 |
| 15 | `16^3` | 0.8700 | 0.6955 | 0.799 | 0.7324 | 0.842 |
| 15 | `32^3` | 6.0865 | 0.7509 | 0.123 | 0.8062 | 0.132 |
| 15 | `64^3` | 51.0161 | 1.7509 | 0.0343 | 1.9296 | 0.0378 |
| 15 | `96^3` | 187.1794 | 7.8893 | 0.0421 | 8.2890 | 0.0443 |

For WENO-5, the observed 3-D crossover lies between `16^3` and `32^3`, even
with transfers included. WENO-15 already favors CUDA at the smallest tested
3-D grid. In 1-D, CPU won every admitted point. WENO-15 CUDA at 1-D sizes
`512--32768` is intentionally absent because no CUDA lane passed parity.

This is a shape-and-formulation result, not a claim that GPU scaling is
superlinear or that work ceases to scale with cells. Both devices perform more
work as the grid grows; the GPU amortizes fixed costs and exposes more parallel
capacity.

## Float32 one-dimensional exclusions

The strict cross-device gate exposed a real numerical boundary:

- at WENO-5 sizes `512--32768`, eager CUDA passed but compiled CUDA failed;
- at the cross-order `N=8192` point, orders 5 and 7 retained only eager CUDA;
- at orders 9, 11, 13, and 15, neither CUDA lane passed; and
- for WENO-15, neither CUDA lane passed sizes `512--32768`.

For example, WENO-5 compiled CUDA at `N=8192` had normalized maximum/RMS
differences `6.217e-4 / 1.032e-4`, against frozen bounds `5e-5 / 5e-6`.
WENO-15 compiled CUDA at the same size had `8.548e-4 / 1.161e-4`.
Every listed output remained finite and conservative. A2 establishes failure
of the declared cross-device agreement contract, not which rounded float32
result is closer to an infinite-precision solution. No threshold was relaxed
and no failed lane was timed or used to infer a crossover.

All scalar float64 1-D cross-order CUDA lanes passed. At `N=8192`, compiled
CUDA became faster than CPU for orders 11, 13, and 15 resident and with
transfers, but CPU won orders 5, 7, and narrowly order 9. This float64 result
does not repair or replace the float32 exclusions.

## Prepared AOT

The fixed-shape packages implement scalar float32 `64^3`. Build preparation
is outside warm and launch-to-answer invocation, but is reported in full.

| Order | Total build s | Package | Load ms | JIT / AOT resident ms | AOT/JIT | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 5 | 10.981 | 0.92 MB | 4.55 | 0.2106 / 0.1668 | 0.784 | AOT win |
| 11 | 27.363 | 2.19 MB | 2.32 | 0.5964 / 0.5514 | 0.921 | AOT win |
| 15 | 57.134 | 4.68 MB | 4.06 | 1.7322 / 1.6790 | 0.969 | unresolved |

Transfer-inclusive AOT/JIT median ratios were `0.892`, `0.943`, and `0.968`;
the first two are resolved AOT wins and order 15 remains unresolved under the
frozen five-percent rule. AOT's main demonstrated benefit is therefore removal
of compilation from invocation, not a universal faster steady-state kernel.

## Fresh-process launch to answer

Parent wall time starts before process creation and ends after the CPU checksum
returns. It includes Python startup, imports, input construction, compilation
or package loading, transfers, one RHS, output return, and process teardown.
Each value is the median of three independent processes.

### Prepared persistent compiler cache

| Order | Grid | CPU JIT s | CUDA JIT s | CUDA AOT s |
| ---: | :--- | ---: | ---: | ---: |
| 5 | 1-D, 8192 | 4.757 | excluded | not built |
| 5 | `64^3` | 4.945 | 3.480 | 1.656 |
| 15 | 1-D, 8192 | 5.766 | excluded | not built |
| 15 | `64^3` | 7.842 | 6.254 | 1.666 |

### Isolated empty compiler cache per process

| Order | Grid | CPU JIT s | CUDA JIT s | CUDA AOT s |
| ---: | :--- | ---: | ---: | ---: |
| 5 | 1-D, 8192 | 7.665 | excluded | not built |
| 5 | `64^3` | 11.075 | 7.752 | 1.657 |
| 15 | 1-D, 8192 | 19.228 | excluded | not built |
| 15 | `64^3` | 58.742 | 71.183 | 1.666 |

At `64^3`, AOT reduced the isolated-cache launch-to-answer endpoint by `4.68x`
against CUDA JIT for order 5 and `42.73x` for order 15. It was `6.69x` and
`35.26x` faster than the corresponding CPU JIT observations. These invocation
ratios exclude the explicitly reported one-time package build. The isolated
AOT values replicate the prepared-cache AOT values within about one percent,
confirming that package invocation does not rely on a populated JIT cache.

The prepared-cache JIT table must not be called cold compilation. The isolated
table is the clean-cache deployment endpoint; the core first-call records are
in-process compile-and-first-execute observations.

All three prepared-cache configurations and seven of eight isolated-cache
configurations returned an identical float64 checksum across their three
processes. The isolated WENO-15 `64^3` CUDA JIT lane returned checksums
`4.542743e-6`, `4.542743e-6`, and `4.852961e-6`. Every output was finite and
the corresponding core lane passed full-array parity; the approximately
`3.10e-7` sum difference is retained as compile-to-compile floating-point
variability. The isolated AOT lane returned an identical checksum in all three
processes.

## Characteristic Euler transfer slice

Only the order-five characteristic configurations passed all frozen
preconditions. At `32^3`:

| Dtype | Fastest compiled CPU ms | Compiled CUDA resident ms | CUDA + copy ms | CPU/CUDA resident |
| :--- | ---: | ---: | ---: | ---: |
| f32 | 17.368 | 0.402 | 0.534 | 43.20x |
| f64 | 26.604 | 1.608 | 1.811 | 16.55x |

The registered CUDA-only order-five float32 `64^3` point passed at `1.952 ms`
compiled resident versus `16.363 ms` eager resident. Its accidentally launched
CPU counterpart is excluded by design.

No higher-order characteristic performance claim is admitted:

- order-11 float32 and order-15 float32/float64 CPU references failed the
  frozen conservation precondition at `32^3`;
- order-15 float32 also failed that precondition at `64^3`; and
- for order-11 float64, the CPU reference passed but both CUDA results failed
  the strict conservation test despite approximately `1e-14` normalized
  pointwise differences.

Some individual compiled CUDA outputs were conservative where their CPU
reference was not. The raw worker initially timed them; the deterministic
analysis excludes the entire configuration because an unhealthy reference
cannot qualify a performance lane. This audit rule is documented rather than
hidden.

## Fixed historical controls: separate workloads

These controls are not divided into the scalar-RHS timings. They evaluate the
fixed float32 three-dimensional Shu Euler WENO-5 SSP-RK3 workload and retain
their original endpoints.

| Control | Grid/work | Endpoint | Median | Meaning |
| :--- | :--- | :--- | ---: | :--- |
| DVEB device ABI v2 | `128^3`, 1 step | resident public ABI | 9.638 ms | qualified generated-native control |
| AOT PyTorch in E4 | `128^3`, 1 step | resident | 69.790 ms | same fixed Shu workload |
| Eager PyTorch in E4 | `128^3`, 1 step | resident | 425.656 ms | same fixed Shu workload |
| G4 native face-once | `128^3`, 1 step | CUDA numerical loop | 5.021 ms | non-admitted schedule control |
| G4 native cell-recompute | `128^3`, 1 step | CUDA numerical loop | 9.608 ms | exact-math schedule control |
| Final DVEB generated CUDA | `128^3`, 1 step | fresh process | 245.577 ms | within 0.1% of 245.414-ms ceiling |
| Original bakeoff Fortran | duplicated-endpoint `N=128`, 1 step | fresh process to host state | 3153.075 ms | historical CPU deployment control |

The E4 records show DVEB `2.53x--7.36x` faster than that study's AOT path over
its frozen resident grid/step points. This remains evidence that generated
native CUDA can occupy a useful middle level for one matched WENO-5 workload;
it is not an arbitrary-order DVEB result. G4's faster face-once candidate is a
causal schedule result and remains non-admitted for reasons recorded in G3.
The original automatic bakeoff's exact DVEB artifact came from an uncommitted
compiler state; the later final requalification is the source-reproducible
deployment control.

## Scientific interpretation

1. **Ordinary PyTorch is a viable arbitrary-order execution source.** The
   compiler captured the generated order-5--15 graphs without bespoke CUDA or
   graph breaks, and warm `64^3` GPU execution was strongly favorable.
2. **There is no universal GPU answer.** CPU dominates the admitted small and
   one-dimensional float32 cases; 3-D creates the parallel workload that makes
   this consumer GPU useful.
3. **Order changes both arithmetic and deployment economics.** Warm time,
   memory, first compilation, and package size all grow. By order 15, removing
   JIT compilation from invocation matters far more than AOT's small warm
   kernel difference.
4. **FP64 is slower but not a universal local loss.** Despite the RTX 5070 Ti's
   consumer FP64 limits, compiled CUDA still won the scalar `64^3` and
   admitted characteristic order-five comparisons. Data-center replication is
   needed before a broader FP64 conclusion.
5. **Numerical limits are part of the performance result.** The float32 1-D
   and higher-order characteristic exclusions prevent benchmark speed from
   outranking correctness. They identify where precision/roundoff research is
   justified; they do not authorize a post-hoc mixed-precision rescue in A2.
6. **AOT is reasonable for a finite configuration family.** A package is
   specific to its exported graph and fixed shape, not one package per initial
   state. A library can prepare packages for supported equation/order/shape/
   dtype families and fall back to cached JIT or eager execution elsewhere.

## What A2 does not establish

A2 does not prove universal superiority over Fortran, CUDA, DVEB, JAX, or
finite-volume methods; production aerospace readiness; real-time complete CFD;
arbitrary boundary or equation support; characteristic robustness through
order 15 under the current strict float policy; performance on A100/H100 or
Apple GPU hardware; or novelty of WENO or PyTorch WENO itself.

## Evidence

Machine-readable raw observations, deterministic analysis, commands, and
hashes are under `experiments/academic_a2/evidence/a2_20260830/`. The offline
verifier and its regression test enforce completion, hashes, graph behavior,
cache-policy separation, AOT qualification, and the exclusion of the one
unregistered worker.
