# GPU-native WENO reformulation exploration

## Status

This document defines an intentionally noncanonical research exploration.  It
does not change GradFlow's qualified numerical source, public API, or academic
claims.

GradFlow's engineering law remains:

> **Correctness > performance > convenience.**

That law governs admission into the project.  It does not prohibit a bounded
experiment from first constructing a deliberately unsafe, throughput-oriented
algorithm and then asking whether independent correctness can be recovered.
No result from this exploration is eligible for `src/gradflow/` until it has
passed a new mathematical and numerical qualification.

## Research question

If characteristic finite-difference WENO were designed around a modern GPU
rather than inherited from a line-oriented vector-CPU implementation, what
computation graph would minimize time to a complete conservative update, and
how much of that graph can be retained while recovering the independently
specified solution?

The purpose is not to make an existing implementation faster by changing an
unexamined tolerance.  The purpose is to expose the performance frontier first,
measure its numerical damage second, and then locate the smallest set of
correctness-restoring operations.

## Plausibility floor

Without a floor, the fastest inaccurate solver is one that returns its input.
Every exploratory candidate must therefore:

- advance the same declared conservation law;
- update the state through conservative face-flux differences;
- use a nonlinear WENO-type stencil weighting somewhere in the update;
- preserve explicit boundary and time-step semantics in its record;
- return the complete requested state; and
- disclose every approximation, precision change, and altered formulation.

Candidates may initially change characteristic treatment, flux splitting,
weight sharing, precision, endpoint storage, or time integration.  Such a
candidate is a different numerical contract, never a silent optimization.

## Four layers kept separate

1. **Continuum problem:** governing PDE and physical state.
2. **Exact-arithmetic discretization:** stencils, indicators, nonlinear
   weights, flux splitting, characteristic treatment, and time integration.
3. **Floating-point algorithm:** coefficient evaluation, normalization,
   operation ordering, precision, contraction, and exceptional-value policy.
4. **Execution schedule:** layout, tiling, face ownership, fusion, residency,
   launch structure, reductions, and communication.

The first ablations change only layer 4 where possible.  The reckless endpoint
also explores layers 2 and 3, but each change is named so it can later be
restored independently.

## Existing evidence and starting point

The strongest current native specimen is DVEB's matched 3-D Shu Euler WENO-5
CUDA path.  In GradFlow's frozen E4 campaign it reached `9.638 ms` for one
resident `N=128` SSP-RK3 step and was `7.24x` faster than the packaged
AOTInductor observation.  The full-array discrepancy remained below the
predeclared float32 gate.

That specimen is not a maximum-performance implementation.  Its CUDA source is
explicitly a `cell-recompute` layout:

- one thread owns one output cell;
- each cell reconstructs both adjacent faces in all three directions;
- neighboring cells therefore recompute a shared face;
- the six-point state and physical-flux data are recomputed for both faces;
- line-wise Lax--Friedrichs reductions are separate kernels;
- RHS and RK updates are separate global-memory passes; and
- the compiled RHS uses 188 registers per thread, although it reports no spill.

This is excellent correctness-oriented CUDA and a valid DVEB ceiling-distance
reference.  It leaves a separate, more aggressive face-reuse/dataflow question
open.

Frozen source identities inspected for this exploration:

| Source | SHA-256 |
| --- | --- |
| DVEB Shu ceiling `cuda.cu` | `c3964d31399bb4d2b68bdd2c33a70aa5263ea3b370a3d94e2dde2f169dfcfb6d` |
| DVEB Shu ceiling `shu_math.h` | `125dd8ec0d60cc4c965e1a8f804b12ae471cf73850e3484520cc400ae0db9009` |
| DVEB portable pipeline CUDA runtime | `f339f75c807ac3932ea080949a8bf3b6ef5ff33a4a4b7c7ebf3708934262106c` |
| GradFlow E4 result document | `0eae9e1c020170584eeb030b0b9ab44c7e3ffb66eedb906c3d068595bda2e04f` |

The source remains owned by and read from DVEB.  It is not copied into
GradFlow.

## Proposed maximal-throughput graph

The initial architecture hypothesis is a **resident, face-owned,
pencil-tiled, stage-fused GPU solver**.

### State and ownership

- Store only unique periodic nodes in component-major or measured blocked-SoA
  form; synthesize periodic halos during tile loading.
- Keep the state resident for the complete solve.
- Assign each physical face to exactly one computational owner.
- Compute each numerical face flux once and reuse it for the two adjacent cell
  divergences.
- Distribute the characteristic families, stencil candidates, and components
  within a warp rather than making one thread hold the complete face algebra.

### Directional pencil pipeline

For each spatial direction and RK stage:

1. A block loads one or more stencil pencils, including halo values, into
   shared memory with a direction-appropriate coalesced mapping.
2. Threads compute primitive data and line wave-speed candidates while the
   pencil is being loaded.
3. A block reduction produces the line speed when the frozen formulation
   requires it.
4. Warps evaluate distinct faces once.  State, physical flux, and Roe data are
   shared or shuffled rather than materialized as whole-grid tensors.
5. Adjacent face fluxes immediately become a directional divergence.
6. The x pass writes one partial RHS; the y pass accumulates it; the z pass
   consumes it and performs the RK update.  No full face-flux arrays and no
   separate final update pass are retained unless measurement proves them
   cheaper.

For line lengths that fit one block, the reduction and reconstruction can be
one launch.  Longer lines require a segmented reduction protocol and are a
separate shape regime.

### Launch and residency policy

- Capture the fixed-shape stage sequence in a CUDA graph or an equivalent AOT
  launch schedule.
- Allocate all scratch once.
- Perform CFL reduction, all stages, endpoint synchronization, and requested
  final materialization without host control in the numerical loop.
- Report process-entry and resident endpoints separately.

### What is deliberately not assumed

- Convolution is not presumed optimal.  It must beat direct face arithmetic
  after accounting for nonlinear weights and intermediate tensors.
- Tensor cores are not presumed useful.  Small fixed transforms only qualify
  if batching them as matrix operations wins after packing and precision cost.
- Higher occupancy is not presumed faster.  Register count, reuse, instruction
  issue, and memory traffic must be measured together.
- Precomputing primitive or physical flux arrays is not presumed better than
  recomputation.  Both are explicit candidates.

## Reckless numerical endpoint U0

The first intentionally unsafe endpoint combines the changes most likely to
make the GPU comfortable:

- unique periodic-node storage;
- float32 conserved state and flux arithmetic with contraction/fast intrinsic
  use declared;
- componentwise conservative reconstruction of the split physical fluxes;
- one density-derived smoothness sensor, with its nonlinear weights shared by
  all five conserved-flux components;
- one face-local Lax--Friedrichs speed obtained from the six-point face
  stencil, eliminating the separate line reduction;
- face-owned pencil fusion and device-resident time stepping; and
- one conservative Forward Euler update per declared step at the existing
  `CFL=0.1`, in place of three-stage SSP-RK3.

Reduced-precision indicators or raw weights are a measured U0 variant, not an
assumption.  FP16/BF16 may save arithmetic but can also add conversion traffic
and does not automatically engage tensor cores.

U0 therefore remains recognizably a conservative WENO-derived Euler solver,
but it deliberately gives up characteristic decoupling, separate wave-family
weights, the ancestral global split, and third-order time integration.  It is
expected to disagree with the Shu oracle and may fail strongly shocked flows.
Its value is to establish a plausible speed frontier, not to produce an
acceptable solver.

## Correctness-recovery ladder

After U0 timing is frozen, restore one contract feature at a time:

| Step | Restored property | Principal question |
| --- | --- | --- |
| R1 | strict float32 indicators, weights, and division semantics | Were reduced precision or fast intrinsics responsible for the error? |
| R2 | separate nonlinear weights per reconstructed component | Is shared sensing admissible? |
| R3 | Roe characteristic projection and back-projection | Is componentwise reconstruction the decisive failure? |
| R4 | ancestral per-line characteristic LF speeds and enlargement | What is the accuracy/performance cost of the global split? |
| R5 | qualified SSP-RK3 time integration | What was gained by altering the temporal graph? |
| R6 | canonical epsilon, duplicated-endpoint, and operation-order contract | Can the exact GradFlow oracle be recovered? |

The restoration order may be changed only before comparative results exist,
with a written causal reason.  If interactions are suspected, a small frozen
factorial follows the ladder; post-result tolerance relaxation is forbidden.

At each recovery step report:

- full-state discrepancy against the independent oracle;
- conservation and admissibility;
- smooth convergence and critical-point behavior;
- Sod and Shu--Osher behavior;
- gradient behavior if differentiability is retained;
- resident and complete latency;
- peak workspace;
- registers, spills, shared memory, and launch count; and
- the incremental accuracy recovered per unit of time added.

The first candidate that passes the complete independent gate becomes eligible
for a performance comparison.  Faster failing candidates remain scientifically
useful negative evidence but cannot become GradFlow backends.

## First implementation boundary

The first executable prototype is U0 in full.  This follows the declared
discovery order: construct the most GPU-comfortable plausible solver first,
freeze its throughput and output, then consult the oracle and recover
correctness.

U0 should be implemented with compile-time or generated switches for each
named relaxation.  The switches exist only to support the later recovery
ladder; they must not cause the first U0 result to be tuned against the oracle.

The retained qualified cell-recompute specimen is the initial hardware control.
Because U0 uses Forward Euler while the control uses SSP-RK3, their complete
step times are **not** an apples-to-apples solver speedup.  Before R5 restores
SSP-RK3, report U0 as an unsafe throughput frontier using:

- time per declared update;
- state cells advanced per second;
- directional faces evaluated per second;
- time to the same simulated final time under each candidate's declared CFL;
  and
- the resulting numerical discrepancy.

After U0 is frozen, the recovery/control sequence includes:

1. face-once global/read-only-cache scheduling;
2. face-once shared-pencil scheduling;
3. face-once shared-pencil scheduling with fused directional accumulation and
   update;
4. the R1--R6 numerical restorations; and
5. the fully restored face-once schedule against the qualified 188-register
   cell-recompute kernel.

Only the fifth comparison answers whether face reuse is an exact-contract
optimization.  The earlier U0 observations answer how much speed exists at the
unsafe frontier.

## G5 outcome: literal shared pencils do not recover the Pareto frontier

G5 implemented the first frozen shared-pencil/fused-update schedule after the
R6Q recovery. The P1 candidate retained R6Q arithmetic exactly on all five
forward specimens and reduced declared `N=128` peak allocation from
336,134,148 to 210,305,028 bytes.

It was nevertheless 2.619x and 2.701x R6Q resident time at the primary one-
and ten-step points, and cell-recompute was faster than P1 throughout the
matrix. Profiling found no spills. Privileged hardware counters subsequently
showed 33.33% register-limited theoretical occupancy and a shift from 6.30%
L2 throughput for x pencils to 64.53% for y and 91.81% for z, corroborating
direction-strided access in the existing state layout as the main cause. Full
evidence and the narrow claim boundary are in
`experiments/gpu_native_reformulation/G5_SHARED_PENCIL_RESULTS.md`.

This rejects the P1 schedule, not shared-memory tiling as a category. Any
future memory-recovery candidate must explicitly preserve coalescing in every
direction and must be preregistered as a separate experiment. R6Q global
face-once remains the current non-admitted throughput schedule control.

## G6 outcome: occupancy is not the optimization target

G6 held R6Q mathematics, layout, face ownership, and workspace fixed while
crossing 64/128/256-thread face blocks with uncapped, 112-register, and
96-register compilation. All nine candidates were bitwise identical to R6Q
on 45 forward comparisons. None passed the preregistered improvement rule at
both primary points.

The 112-register cap did not cross a residency threshold: 256-thread launches
still admitted two blocks per SM and retained 33.33% theoretical occupancy.
The 96-register cap raised theoretical occupancy to 41.67% for 64/128-thread
blocks only by spilling face-kernel live values, and was slower at moderate
and large grids. Privileged counters showed 32.39% versus 32.44% achieved
occupancy and approximately 73% SM throughput for frozen R6Q versus the
profiled 112-register candidate. Nsight Systems showed identical face-kernel
time within about 0.05%.

The experiment also exposed a measurement boundary: all rebuilt lanes moved
lazy function setup outside their event by querying CUDA function metadata.
The rebuilt uncapped 256-thread negative control and sustained timing separate
that common first-event effect from occupancy. The full correction and
evidence are in `experiments/gpu_native_reformulation/G6_OCCUPANCY_RESULTS.md`.

This closes simple block-size/register-cap tuning. A future attempt would need
a separately frozen change to live-value structure, such as warp-distributed
characteristic work; it cannot be described as merely "raising occupancy."

## Claim boundary

This exploration may establish that inherited execution structure left
performance unused, or that the apparent reformulations lose more through
memory traffic and synchronization than they save through face reuse.  Either
answer is useful.

It cannot initially establish a new WENO method, universal GPU optimality,
correctness, production readiness, arbitrary-order behavior, or portability.
Those require later qualification and a systematic prior-art audit.
