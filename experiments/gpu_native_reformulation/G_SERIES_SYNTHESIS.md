# Reckless-to-correct GPU-native WENO synthesis

Status: **G0--G6 complete; experimental program closed**.

Closure date: 2026-08-30 (UTC)

## The question

The G series deliberately reversed the usual development order. It first
asked what a recognizable WENO-derived Euler update would look like if GPU
comfort were the only concern, even at the cost of numerical fidelity. Only
after that unsafe implementation, its timing, and its output were frozen did
the study consult the independent GradFlow/Shu oracle and restore the
mathematical contract one property at a time.

This was not a search for permission to weaken WENO. It was a controlled way
to separate two kinds of inheritance:

1. mathematics that must survive because it determines the requested method;
2. CPU-era execution structure that may be changed without changing that
   mathematics.

The experiment is now sufficient to answer that bounded question. No G7 or
additional CUDA microarchitecture campaign is authorized by this closeout.

## Experimental arc

| Phase | Question | Result |
| --- | --- | --- |
| G0 | Where does the inherited graph duplicate work or materialize state? | Cell-owned execution reconstructed every interior face twice; face ownership, live-value distribution, reductions, and update fusion were plausible schedule hypotheses. |
| G1 | How fast is an intentionally unsafe GPU-comfortable endpoint? | U0 completed one `128^3` Forward-Euler update in `0.515 ms` with 72 face registers, but this was not mathematically comparable to the SSP-RK3 control. |
| G2 | How wrong is frozen U0? | Its one-step maximum state error was `3.27e-4`, but the more informative error was `0.839` of the oracle update RMS with update cosine `0.592`: U0 produced a substantially different increment. |
| G3 | Which restorations recover the Shu contract, and what do they cost? | Characteristic projection mattered, the ancestral line-family LF policy was decisive, and Shu difference form closed the remaining gap. R6 reached `3.51e-7` one-step maximum error and `4.709 ms` at `128^3`. |
| G3Q | Does the interface-extended R6Q pass a broad frozen gate? | State-level evidence was strong, but six RHS relative-RMS points and literal zero conservation budgets failed; no autograd ABI existed. R6Q was not admitted. |
| G4 | Is face ownership itself a real exact-math speed effect? | Yes, as a non-admission schedule control: face-once was `1.913x` and `1.993x` faster at the primary one- and ten-step points, with about `1.95x` the workspace. |
| G5 | Can a literal shared-pencil schedule recover the memory without losing speed? | No. It was bitwise identical and used 37.43% less memory than global face-once, but took `2.619x--2.701x` as long at the primary points. |
| G6 | Can block size or a compiler register ceiling recover performance through occupancy? | No. A 112-register cap left occupancy unchanged; a 96-register cap raised nominal occupancy only by spilling and lost about 2% at useful sizes. |

## What the reckless endpoint taught us

U0 was fast partly because it stopped doing the requested problem. It used
componentwise reconstruction, one shared density sensor, face-local LF
speeds, fast FP32 arithmetic, and Forward Euler. Its absolute state error can
look superficially small because one timestep changes an order-one state only
slightly. Relative to the update that the oracle was supposed to produce,
however, U0's error was 83.9% and its direction cosine was only 0.592.

That is the intended lesson of G1/G2: throughput from a numerically different
increment is not a WENO speedup.

The recovery ladder then separated valuable mathematics from replaceable
execution structure:

- strict arithmetic alone added cost without recovering the update;
- separate component weights did not bridge the componentwise and
  characteristic methods;
- Roe characteristic projection substantially corrected the update;
- the per-line characteristic LF policy and 1.1 enlargement reduced the
  remaining update discrepancy by about fourteenfold;
- restoring SSP-RK3 made the temporal endpoint comparable; and
- Shu difference form and its epsilon scaling reduced the one-step maximum
  discrepancy by about 51-fold from R5 while also running faster.

The correct conclusion is therefore not that the numerical method should be
made reckless. It is that unique periodic storage and one owner per
directional face can survive restoration of the Shu mathematics.

## The reusable exact-math result

The most important positive G-series observation is the G4 schedule control.
For the fixed FP32, periodic, three-dimensional characteristic WENO-5 problem
on Forge's RTX 5070 Ti, constructing each directional numerical face once
removed duplicated characteristic algebra and produced a sustained
approximately `1.9x--2.0x` resident speedup at moderate and large grids.

That gain was localized by profiling: the face-once three-stage face kernels
took 2.512 ms at `128^3`, while the cell-recompute RHS kernels took 7.349 ms.
Both schedules retained the same 17-launch class. The result was not obtained
by deleting Runge--Kutta stages, line-speed reductions, or mathematical work.

The speed is not free. Three five-component face arrays make the global
face-once schedule use approximately twice the large-grid workspace. Small
one-step jobs also retain a crossover: cell recomputation won at `8^3` and
`16^3`, while face-once won from the observed `32^3` point onward. Process
startup hides much of either kernel result for small command-line jobs.

This is a schedule result, not a backend admission. R6Q remains outside the
public GradFlow implementation because it failed the immutable G3Q aggregate
gate and provides no differentiable ABI.

## Why the two attempted follow-up optimizations stop here

G5 showed that eliminating the face arrays through a literal
one-block-per-line shared-memory pencil is the wrong exchange for this layout.
The candidate preserved R6Q output bitwise and reduced memory, but strided y/z
access moved the cost onto cache-side service. At `N=128`, three x pencils
took 1.660 ms under counter replay, versus 6.570 ms for y and 5.050 ms for z.
Low DRAM use showed that off-card bandwidth was not the immediate limit.

G6 then showed that occupancy is not an independent objective. The exact face
kernel achieved about 32.4% occupancy against a 33.33% theoretical ceiling
and about 73% SM throughput. Reducing 128 registers to 112 did not admit
another block and did not change speed. Reducing to 96 admitted more small
blocks only by spilling 80-byte stores and 88-byte loads per thread; the
result was slower at moderate and large grids.

A warp-distributed face or a different live-value decomposition could be a
legitimate new design, but it would be a substantial bespoke-CUDA project,
not completion of the present occupancy question. Its expected value is too
low relative to the unfinished GradFlow Academic core.

## Established, observed, and not established

### Established within the frozen experiments

- U0's numerical increment is materially different from the Shu oracle.
- Restoring characteristic reconstruction, ancestral LF policy, SSP-RK3, and
  Shu difference-form semantics recovers close forward agreement.
- Exact-math face ownership avoids duplicated characteristic reconstruction
  and produces an approximately twofold resident schedule advantage at the
  tested moderate/large grids.
- That global reuse costs approximately twofold workspace.
- The tested literal shared-pencil implementation is not Pareto-competitive.
- The tested block-size/register-cap occupancy intervention is ineffective.

### Observed but not promoted

- R6Q passes all measured full-state bounds and several independent smooth,
  shock, health, and sensitivity diagnostics.
- Its frozen aggregate qualification nevertheless fails, so these observations
  support a research hypothesis rather than a backend.
- A future compiler or generated backend may use the schedule insight only
  after passing its own mathematical, differentiation, and device gates.

### Not established

- a new WENO discretization;
- a production or differentiable CUDA backend;
- arbitrary-order face-once performance;
- superiority on another GPU, precision, equation, boundary, or layout;
- universal GPU superiority over CPU execution;
- optimality of the retained schedule; or
- a publication claim by this experiment alone.

## Disposition

The G-series source, binaries, raw outputs, profiler reports, environment
records, and manifests remain immutable under the existing evidence
directories. R6Q remains the native FP32 WENO-5 throughput schedule control,
not canonical GradFlow code. P1 and the G6 variants remain rejected research
specimens.

Further GPU-native work is deferred unless a later paper gate demonstrates a
specific missing comparator or a measured bottleneck that cannot be answered
from the present record. Any such work requires a separately named,
prospectively frozen protocol. It may not extend G6 post hoc.

GradFlow Academic now returns to its central ordinary-PyTorch question:
arbitrary-order construction, independent numerical validation,
differentiation, and endpoint-explicit performance relative to matched
baselines. The G series contributes one WENO-5 native control and a documented
execution-schedule lesson; it does not redefine the project around handwritten
CUDA.

## Evidence map

| Scope | Primary record |
| --- | --- |
| G0 audit | `G0_STATIC_AUDIT.md` |
| G1 protocol and immutable U0 | `G1_U0_PROTOCOL.md`, `evidence/g1_u0_20260829/` |
| G2 oracle damage | `G2_DAMAGE_PROTOCOL.md`, `evidence/g1_u0_20260829/g2_damage.json` |
| G3 recovery | `G3_RECOVERY_RESULTS.md`, `evidence/g3_recovery_20260829/` |
| G3 qualification | `G3_QUALIFICATION_RESULTS.md`, `evidence/g3_qualification_20260829/` |
| G4 schedule campaign | `G4_PERFORMANCE_RESULTS.md`, `evidence/g4_performance_20260829/` |
| G5 memory recovery | `G5_SHARED_PENCIL_RESULTS.md`, `evidence/g5_shared_pencil_20260829/` |
| G6 occupancy ablation | `G6_OCCUPANCY_RESULTS.md`, `evidence/g6_occupancy_20260830/` |

`verify_g_series.py` validates the immutable manifests, key causal values, and
the G3Q--G6 semantic verifiers without rerunning any performance campaign.
