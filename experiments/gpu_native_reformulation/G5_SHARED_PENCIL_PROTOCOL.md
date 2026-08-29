# G5 Shared-Pencil Memory-Recovery Protocol

Status: frozen before the G5 candidate is implemented or compiled.

Date: 2026-08-29 (UTC)

## Question and claim boundary

Can the fixed R6Q characteristic FD-JS-WENO-5 calculation retain most or all
of G4's face-once speed while removing its three full-grid directional
face-flux arrays?

G5 is an execution-schedule experiment motivated by G4's approximately 2x
speed / approximately 2x workspace tradeoff. It does not repair the G3 gate,
add autograd, admit a backend, change WENO mathematics, or begin
arbitrary-order work.

## Frozen candidate: P1

Candidate contract:

```text
p1_shared_pencil_unique_strict_f32_shu_fused_update_v1
```

P1 must retain R6Q's:

- unique component-major periodic state;
- strict FP32 Shu difference-form characteristic WENO-5 face algebra;
- line-family Lax--Friedrichs maxima and 1.1 enlargement;
- CFL policy and SSP-RK3 stages;
- arbitrary raw unique-state input and RHS-only mode; and
- compiler flags and `sm_120` target.

The only permitted schedule change is:

1. launch one block per periodic line and one thread per possible line
   coordinate, using a fixed 256-thread block for every tested size;
2. compute each line's directional faces once and retain its five-component
   fluxes in dynamic shared memory;
3. difference adjacent shared-memory faces into a full-grid partial-divergence
   scratch array;
4. write the x divergence, accumulate y, then consume z and fuse the existing
   SSP-RK3 stage formula into the z kernel; and
5. reuse otherwise-dead state-stage buffers as partial-divergence scratch.

The axis divergence must be accumulated in the same x, y, z order as R6Q and
multiplied by inverse spacing only after the final sum. No coefficient,
operation policy, launch geometry, layout alternative, block size, or shared
memory organization may be tuned after the first candidate result.

The one-block-per-line schedule is defined only for `N <= 256`. Longer and
segmented pencils are out of scope.

## Expected launch and workspace contracts

P1 retains one CFL scan and finish per step. Each RK stage launches three
line-alpha reductions and three directional shared-pencil kernels, for 20
numerical launches per complete step versus R6Q's 17.

P1 may allocate five complete unique state-sized arrays plus line-alpha
storage and one scalar timestep. It may not allocate a global face array or an
additional RHS array. At `N=128`, its declared peak bytes must be no more than
70% of R6Q's frozen declared peak. Failure stops performance measurement.

## Pre-timing forward gate

Compare P1 to frozen R6Q using exact shared FP32 input bytes:

1. periodic vortex, `N=32`, one and ten steps;
2. G3 perturbed vortex, `N=32`, one step;
3. G3 periodic dual-interface Shu--Osher-type specimen, `N=32`, ten steps;
4. G3 smooth entropy-wave RHS, `N=40`.

Record bitwise identity, maximum error, RMS error, finiteness, minimum density,
and minimum pressure. Full-step maximum error must not exceed `2e-5`; RHS
maximum error must not exceed `5e-5`. Every stepped state must remain finite
with positive density and pressure. A failure stops timing and is preserved.

This is schedule parity with R6Q, not independent backend qualification.

## Frozen performance matrix

Compare three lanes:

- `shared_pencil`: P1;
- `global_face_once`: frozen G4 R6Q;
- `cell_recompute`: frozen G4 control.

```text
N = {32, 64, 128, 192, 256}
steps = {1, 10}
warm-up processes per lane and configuration = 3
randomized counted triplets per configuration = 30
random seed = 20260829
bootstrap resamples = 20000
thermal stop = 80 C
```

Every triplet runs all three lanes in an independently shuffled order using
one exact deterministic vortex input. Every lane runs in a fresh process with
one internal CUDA-event observation. Record resident numerical-loop time,
external fresh-process time, all raw observations, declared peak bytes, and
the same machine/thermal/clock fields as G4. No outlier may be removed.

For every configuration report lane statistics and paired ratios. The primary
points are `(N,steps)=(128,1)` and `(128,10)`.

P1 establishes a successful memory-recovery Pareto result only if, at both
primary points:

- median `P1 / R6Q` resident time is at most `1.10`;
- the bootstrap 95% upper bound for that paired ratio is at most `1.15`;
- P1 declared peak bytes are at most 70% of R6Q; and
- P1 remains faster than cell-recompute by paired median.

Performance better than R6Q is reported if observed but is not required. The
full size matrix characterizes crossover and scaling.

## Bounded profiler record

After timing, use Nsight Systems 2025.3.2 on one `N=128`, one-step P1 process.
Record launch count and kernel-class durations alongside the frozen G4 traces.
Attempt one Nsight Compute 2025.3.1 Basic collection. If Forge again denies
performance counters, record the failure and do not change permissions.

No source may be changed in response to profiler or timing results during G5.

## Evidence and stop condition

Freeze protocol, source, build recipe, binary, compiler log, forward-gate
arrays, randomized observations, analysis, profiler record, environment, and
SHA-256 manifest. G5 ends with a speed-memory Pareto decision, verification,
documentation, and a clean local commit. Do not push without explicit
authorization.
