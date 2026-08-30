# G6 Exact-Math Occupancy Ablation Protocol

Status: frozen before any G6 candidate is implemented, compiled, or timed.

Date: 2026-08-30 (UTC)

## Question and claim boundary

Can launch geometry or a controlled compiler register ceiling improve the
resident performance of the fixed R6Q global-face-once characteristic
FD-JS-WENO-5 schedule without changing its mathematics, state layout, face
ownership, workspace, or numerical output contract?

G6 is a causal execution experiment. It does not admit R6Q as a backend,
repair the frozen G3 qualification failures, add autograd, begin arbitrary
order, or establish that maximum occupancy is maximum performance.

## Frozen source contract

Every G6 candidate retains R6Q's:

- unique component-major periodic state;
- strict FP32 Shu difference-form characteristic WENO-5 algebra;
- line-family Lax--Friedrichs maxima and 1.1 enlargement;
- SSP-RK3 and CFL policy;
- global directional face arrays and separate update kernel;
- arbitrary raw-state and RHS-only ABI;
- `sm_120`, `-O3`, `--fmad=false`, `--prec-div=true`, and
  `--prec-sqrt=true` compilation; and
- 256-thread CFL, line-alpha, and update kernels.

Only the `face_kernel` block size and whole-compilation `--maxrregcount`
setting may differ. The latter is whole-compilation because `nvcc` does not
offer a per-kernel command-line register ceiling; the other kernels' compiled
register and spill records must therefore also be retained.

The runner may report `cudaFuncGetAttributes` and
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` metadata before counted CUDA
events. No metadata query may occur inside the numerical loop.

## Frozen factorial

Cross:

```text
face block size = {64, 128, 256}
register policy = {uncapped, maxrregcount=112, maxrregcount=96}
```

Candidate IDs are `b64_u`, `b128_u`, `b256_u`, `b64_r112`, `b128_r112`,
`b256_r112`, `b64_r96`, `b128_r96`, and `b256_r96`. The exact frozen G5 R6Q
binary is a tenth `frozen_r6q` control. No block size, register ceiling,
launch bound, cache preference, or combined variant may be added after the
first candidate is run.

Compiler logs must record registers, stack, and spill loads/stores for every
kernel. A spilling candidate is not silently discarded: its correctness and
timing remain useful evidence about the occupancy tradeoff.

## Pre-timing gate

Before performance measurement:

1. compile all nine candidates from one hashed source and build matrix;
2. compare `b256_u` with the frozen R6Q binary on exact shared input bytes;
3. compare all candidates with frozen R6Q on:
   - periodic vortex, `N=32`, one and ten steps;
   - perturbed vortex, `N=32`, one step;
   - dual-interface Shu--Osher-type state, `N=32`, ten steps; and
   - smooth entropy-wave RHS, `N=40`;
4. record bitwise identity, maximum and RMS differences, finiteness, minimum
   density, and minimum pressure; and
5. require full-step maximum difference at most `2e-5`, RHS maximum difference
   at most `5e-5`, and finite positive stepped states.

Failure excludes that candidate from performance timing but preserves its
build and gate evidence. Thresholds may not be relaxed.

## Frozen performance campaign

For every passing candidate and the frozen control:

```text
N = {64, 128, 256}
steps = {1, 10}
warm-up processes per lane/configuration = 3
randomized complete lane blocks per configuration = 30
random seed = 20260830
bootstrap resamples = 20000
thermal stop = 80 C
```

Each complete block runs every eligible lane once in independently shuffled
order from one exact deterministic vortex input. Each observation is a fresh
process with one internal CUDA-event measurement. Preserve resident numerical
time, external process time, raw samples, order, temperature, clocks, power,
declared occupancy metadata, compiler resource data, and input hashes. No
outlier may be removed.

Primary points are `N=128`, one and ten steps. A candidate establishes a
meaningful exact-math occupancy improvement only if at both primary points:

- its paired median resident time divided by `frozen_r6q` is below `0.95`;
- the bootstrap 95% upper bound of that ratio is below `1.00`;
- it passes every forward gate; and
- it does not increase declared peak workspace.

Report all variants even if none meets this criterion. Faster performance at
one point is characterization, not selection.

For the bounded profiler comparison, define the "fastest passing G6
candidate" as the candidate with the smallest geometric mean of its paired
median resident `candidate / frozen_r6q` ratios at the two primary points;
break an exact tie by candidate ID. This ranking rule is frozen before timing.

## Profiler boundary

After timing, profile `frozen_r6q` and the fastest passing G6 candidate at
`N=128`, one step, using Nsight Systems 2025.3.2. Prepare one bounded script
for an explicitly authorized privileged Nsight Compute 2025.3.1 Basic
collection of both binaries. Profiler replay time is never a benchmark.

No source, build setting, or campaign result may change in response to timing
or profiler output. A warp-distributed face kernel, variable-lifetime rewrite,
layout change, or shared-memory design is a separately frozen later
experiment, not an amendment to G6.

## Stop condition

G6 ends after the gate, randomized campaign, profiler record, causal
interpretation, checksum verifier, regression test, and coherent clean local
commit. Do not push without explicit authorization.
