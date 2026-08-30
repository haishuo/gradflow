# GradFlow Academic A2 arbitrary-order performance protocol

Status: **frozen before A2 harness implementation, compilation, or timing**.

Date: 2026-08-30 (UTC)

## Question

Across qualified finite-difference WENO-JS orders 5 through 15, how do
maintainable ordinary-PyTorch eager and compiler-generated implementations
behave on CPU and CUDA when formulation, precision, shape, state residency,
correctness gates, and timing endpoints are explicit?

A2 is the paper's principal performance experiment. It does not change WENO
mathematics, tune epsilon or precision, develop DVEB, extend native CUDA,
compare FD with FV, or implement an automatic selector.

## Fixed formulations

### Scalar subject

- exact-generated `WENOJS` orders `{5,7,9,11,13,15}`;
- periodic unique nodes;
- scalar inviscid Burgers flux `f(u)=u^2/2`;
- one global LF speed `max(abs(u))` per directional RHS;
- canonical epsilon `1e-29`, indicator scale 12, and nonlinear power two;
- logical single-owner interface-flux construction; and
- equal spacing `2*pi/N` in every direction.

The deterministic input is the smooth sign-changing field used by the frozen
face-ownership screen. Leading dimensions are physical dimensions, not
synthetic batch duplication.

### Characteristic subject

- existing generated face-frozen Roe-characteristic Euler RHS;
- orders `{5,11,15}`;
- duplicated periodic endpoints;
- ideal-gas gamma `1.4`, epsilon `1e-6`, per-line global LF enlargement `1.1`;
- three-dimensional periodic vortex; and
- the canonical component-major layout.

The characteristic slice tests transfer to a realistic system graph. It is
not merged numerically with the scalar timings.

## Frozen matrix

### S1. Cross-order core

For every order `{5,7,9,11,13,15}` and dtype `{float32,float64}`:

```text
scalar 1-D: N=8192
scalar 3-D: N=64
```

Execute CPU eager and compiled at `{1,6}` PyTorch threads, plus CUDA eager and
compiled. A lane that fails its correctness gate is recorded and not timed.

### S2. Crossover/scale slice

For scalar float32 orders `{5,15}`:

```text
1-D N = {128,512,2048,8192,32768}
3-D N = {16,32,64,96}
```

Reuse overlapping S1 points. Execute the same CPU and CUDA lanes. This slice
locates where device residency and expression order change the useful
execution choice; it does not train an automatic selector.

### E1. Characteristic transfer slice

At `32^3` unique cells, run orders `{5,11,15}` and both dtypes through the four
CPU/CUDA eager/compiled lanes. At `64^3`, add float32 CUDA eager/compiled for
orders `{5,15}`. The timed operation is one semidiscrete RHS, not an SSP-RK3
step.

### P1. Prepared AOT slice

Attempt fixed-shape CUDA AOTInductor packages for scalar float32 `64^3` at
orders `{5,11,15}`. Record export time, package-build time, package size/hash,
load time, graph parity, device-resident execution, and transfer-inclusive
execution. Package failure is a result and may not be replaced by a smaller
order or shape.

### C1. Fresh-process deployment slice

For scalar float32 orders `{5,15}`, shapes `N=8192` in 1-D and `N=64` in 3-D,
run three fresh processes per eligible lane:

- CPU compiled at six threads;
- CUDA compiled; and
- prepared CUDA AOT when available.

Parent-process wall time begins before process creation and ends after a
finite checksum is returned to the parent. This includes Python startup,
imports, input construction, compilation or package loading, transfers, one
RHS, output return, and process teardown. Preparation of an AOT package is
reported separately and excluded from this deployment endpoint by definition.

## Fixed historical controls

A2 imports rather than reruns the hash-qualified WENO-5 Shu-Euler deployment
bakeoff, DVEB device-resident ABI result, and G4 native face-once schedule.
They are fixed order-five system controls with their original full-step
endpoints. They must appear in a separate table and must not be divided into
the new scalar-RHS timings as if the workloads were identical.

No arbitrary-order native, DVEB, or Fortran implementation is required.
TorchInductor CPU is the generated-C++ CPU lane for the new cross-order matrix.

## Correctness gate

The CPU eager output for the same formulation, shape, and dtype is the timing
reference. Before timing:

1. require finite output and the periodic conservation bound
   `abs(sum(rhs)) <= 32*eps(dtype)*sum(abs(rhs))`;
2. compare every compiled CPU, eager CUDA, compiled CUDA, and AOT output with
   CPU eager;
3. require `torch.compile(fullgraph=True, dynamic=False)` to capture one graph
   with zero graph breaks; and
4. record, rather than suppress, compilation/export/runtime failure or OOM.

Normalized max/RMS thresholds are:

| Subject | float32 | float64 |
| --- | ---: | ---: |
| Scalar | `5e-5 / 5e-6` | `5e-11 / 5e-12` |
| Characteristic Euler | `3e-4 / 3e-5` | `5e-11 / 5e-12` |

Normalization uses `max(max(abs(reference)),1)`. Each lane is admitted
independently. No threshold, shape, or dtype may change after observation.

## Warm and transfer timing

For every admitted eager/compiled endpoint:

```text
warmups per lane                 = 5
randomized complete pair blocks = 20
random seed                     = 20260830
bootstrap resamples             = 20,000
thermal stop                    = 80 C
```

CPU resident timing uses `perf_counter_ns` around the call and retains every
sample. CUDA resident timing uses CUDA events. CUDA transfer-inclusive timing
uses wall time around pageable CPU input to CUDA, one RHS, output back to CPU,
and synchronization. CUDA input construction is outside that clock.

Eager and compiled lanes are randomized within complete pairs. Report each
sample, median, mean, MAD, sample standard deviation, paired ratio, bootstrap
95% interval, first-call wall time, graph behavior, peak allocated/reserved
memory, and recorded GPU telemetry. No outlier is removed.

For AOT, compare the admitted package against compiled CUDA with the same
warmup/repetition counts. AOT package preparation is never charged to warm or
fresh-process invocation, but its complete one-time cost is reported.

## Decision rules

A2 does not choose a universal winner. For each paired comparison, a resolved
win requires a median ratio below `0.95` and bootstrap upper bound below one;
the reverse win requires a ratio above `1.05` and bootstrap lower bound above
one. Otherwise the point is unresolved.

The final analysis must distinguish:

- arithmetic/order scaling from fixed overhead;
- CPU thread-count effects;
- CPU versus CUDA resident execution;
- transfer-inclusive versus resident CUDA;
- eager, JIT-compiled, prepared-AOT, and fresh-process endpoints;
- scalar versus characteristic graphs;
- float32 versus consumer-GPU float64; and
- observed failures from unsupported or untested points.

## Reproducibility and stop condition

Record exact source hashes, environment, CPU, GPU, driver/runtime, compiler
settings, raw repetitions, failures, and commands. Add an offline verifier and
regression test. A2 closes only when the frozen matrix is complete or every
missing point has an explicit recorded failure, results and limitations are
written, coherent local commits exist, and the working tree is clean.

Do not push without new explicit authorization. After A2, proceed directly to
the independently validated A3 differentiation use.
