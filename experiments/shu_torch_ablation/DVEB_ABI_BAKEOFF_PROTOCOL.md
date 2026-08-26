# Forced-target DVEB ABI bakeoff protocol

Status: **FROZEN BEFORE HARNESS IMPLEMENTATION OR COUNTED TIMING**.

Date: 2026-08-26

## Question

For the already matched three-dimensional Shu Euler JS-WENO-5 workload, can
DVEB portable ABI v1 compete with the eligible CPU, CUDA, and PyTorch
deployment choices when callable lanes receive the same caller-owned initial
state, every lane solves the same matched initial condition, and every lane
returns the complete answer?

This campaign asks where forced DVEB CPU and forced DVEB CUDA are useful. It
does not test or tune automatic placement. `DVEB Auto` is explicitly excluded
until an ABI-endpoint calibration protocol is frozen separately.

No lane must win everywhere. Small one-step calls, long calls, first CUDA use,
warm repeated use, and fresh applications are different deployment regions
and will not be collapsed into one number.

## Claim boundary

Any conclusion applies only to:

- float32 three-dimensional compressible Euler;
- the precise characteristic finite-difference JS-WENO-5 formulation below;
- cubic duplicated-periodic grids on `[0,10]^3`;
- one Ryzen 5 7600X / RTX 5070 Ti machine;
- the frozen software and artifact hashes; and
- the declared timing endpoint and preparation state.

The campaign cannot qualify general DVEB programs, arbitrary-order WENO,
Navier--Stokes, other boundaries, float64 GPU performance, H100-class GPUs,
Apple Silicon, gradients through native code, or a production selector.

## Frozen mathematical work

Every counted lane performs the same complete operation:

- five-component float32 compressible Euler with gamma 1.4;
- an isentropic vortex extruded uniformly in z with zero z velocity;
- duplicated periodic endpoints on every axis;
- dimension-by-dimension Roe characteristic finite-difference JS-WENO-5;
- Shu's central-flux-plus-nonlinear-correction reconstruction;
- `epsilon=1e-6` with the preserved indicator scaling;
- per-line and per-characteristic-family global LF speeds enlarged by 10%;
- CFL 0.1 recomputed from the current state before every timestep;
- three complete SSP-RK3 stages per timestep; and
- a finite complete final state, not a checksum-only result.

The primary input is constructed once in caller-owned pageable CPU memory in
component-major `(5, nz+1, ny+1, nx+1)` order with duplicated endpoints. The
timed call begins from that ready state for the first-call and warm-call
endpoints. A callable lane may not replace the input with its own initializer,
omit a z sweep, fix the timestep, change the nonlinear-weight algebra beyond
an already gated equivalent form, or leave the answer on a device when the
endpoint requires CPU output. The original Fortran executable and native
fresh-process diagnostics may construct the mathematically identical vortex
internally because they expose no arbitrary-state callable ABI; they are
consequently ineligible for the `Solver.run` endpoints.

The committed non-vortex arbitrary-state ABI gate remains separate evidence
that DVEB consumes its input. Reusing the common vortex here preserves direct
comparability with the Fortran and independently written native ceilings.

## Frozen source and artifact identities

The protocol begins from:

```text
GradFlow source commit  0f37bb463cf14dc690701071adcec7ad339e6295
DVEB source commit      f71d86717c065841c002b41287ff943e9f0a7898
DVEB ABI library        cfa939a5b492ed5711a432391d604ceda65ed55c6df7a4a77b6bfabdd7bd1b1c
DVEB ABI header         c14731d87423f95f9b19f216ddb7d4d2719e7196b6bd0d19205598ab23015c2a
DVEB program            c6e5bd916f951ff412eac99863a74f8c98e5e14b044097a7ad59fe26f704c381
DVEB math module        555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260
native CUDA ceiling     873a9227196664398012e7d42a27e29ec9cd3610c45a4c61ab40a0688aed3caa
```

Preparation must freeze copies outside the active repository and record the
SHA-256 of every executable, shared library, header, AOT package, source file,
protocol, worker, and analysis script actually used. The prepared manifest
also records both repository commits and refuses any later mismatch.

The DVEB executable's known NVCC local-symbol nondeterminism does not permit
substituting a rebuilt ABI library after preparation. The exact shared-library
hash above is the initial candidate. If rebuilding becomes necessary, stop,
amend the protocol before timing, rerun correctness, and record a new hash.

## Lanes

### Primary deployable lanes

1. **Fortran CPU** — the existing repaired/extended GNU Fortran executable.
   It participates only in the fresh-application endpoint because it has no
   arbitrary-state callable ABI.
2. **DVEB ABI CPU-6** — `backend="cpu-simd"`, six OpenMP workers.
3. **DVEB ABI CPU-12** — the same generated CPU code with twelve workers.
4. **DVEB ABI CUDA** — `backend="cuda-native"`; ABI v1 performs and charges
   its required H2D and D2H copies.
5. **Direct eager PyTorch CUDA** — ordinary direct PyTorch without compilation.
6. **Persistent-cache `torch.compile` CUDA** — the same direct source with a
   prepared persistent cache; any reconstruction or compilation occurring
   after the timed call begins is charged.
7. **Packaged AOTInductor CUDA** — a fixed-shape prepared package of the same
   direct operation.

CPU-6 and CPU-12 are retained independently; this protocol performs no new
pilot-based pruning. CPU affinity and worker counts are fixed, reported, and
may not be changed per grid after observations begin.

### Diagnostic lanes

8. **Independent OpenMP C++ ceiling**, where the existing matched ceiling
   exposes the required endpoint.
9. **Independent handwritten CUDA ceiling**.
10. **Cold `torch.compile`**, with a new empty cache and cache reuse disabled.
11. **Pristine AOT package load**, with a new empty extraction cache.

Diagnostic lanes establish native or preparation distance. They are not
silently promoted to the primary user-facing winner table.

The exact convolution feature-bank candidate is not repeated. It already lost
to direct eager PyTorch at the relevant N=64/128 one/ten-step observations and
used more peak GPU memory. Its committed result remains contextual evidence,
not a missing lane. Direct PyTorch CPU is a correctness authority rather than
a counted performance lane; Fortran and generated DVEB CPU provide the
declared CPU deployment comparisons.

## Preparation is recorded, not charged to prepared calls

Before any correctness or counted timing:

- build and hash the Fortran executable;
- copy and hash the DVEB ABI sidecar set;
- build and hash each fixed-shape AOT package;
- prepare each persistent TorchInductor cache;
- perform one explicit AOT extraction-cache preparation for the prepared-AOT
  lane;
- record build, export, cache preparation, package extraction, and DVEB AOT
  build duration independently; and
- capture compiler versions and build flags.

Ahead-of-time preparation is excluded from prepared invocation latency but is
never called free. Any compilation, extraction, cache miss, or library setup
that occurs after a measured invocation begins is charged to that invocation.
Cold-JIT and pristine-package diagnostics deliberately measure those costs.

No placement calibration occurs in this campaign.

## Correctness gate before timing

Before any counted performance observation, every eligible lane must return a
complete finite state agreeing with the direct PyTorch CPU authority at:

```text
N=6,   steps=1
N=6,   steps=10
N=32,  steps=1
N=128, steps=1
```

The frozen float32 bound is `rtol=0`, `atol=2e-5`. Report every pairwise
maximum error among DVEB CPU-6, DVEB CPU-12, DVEB CUDA, direct PyTorch CUDA,
compiled PyTorch, AOT PyTorch, Fortran, and the applicable ceilings. Checksums
may detect corruption but cannot replace full-array comparison.

The gate additionally verifies:

- duplicated endpoints in every returned state;
- no NaN or infinity;
- requested step count and CFL policy;
- no hidden compilation in prepared lanes;
- exact artifact hashes; and
- that ABI timing fields are nonnegative and nested within the surrounding
  Python call interval.

Any correctness failure stops the campaign. It is not averaged away or
classified as a performance loss.

## Timing endpoints

### E1 — fresh application, CPU result

An external monotonic clock starts immediately before process creation and
stops after successful process exit. It includes imports or native loading,
state construction, backend setup, CUDA initialization, required transfers,
the complete solve, CPU result materialization, finite/checksum validation,
and teardown.

This preserves continuity with the earlier deployment bakeoff. It answers,
“How long does a standalone invocation take?” The worker also reports an
internal timestamp ending when the complete CPU state becomes available so
validation and teardown can be separated diagnostically.

### E2 — first `run`, CPU state already available

A fresh worker imports its framework, constructs and validates the CPU input,
constructs the solver/backend configuration, and then performs no numerical
backend call. The monotonic timer surrounds exactly the first call that
advances the ready CPU state and returns a complete CPU result.

For the native DVEB lanes this is the wall interval around `Solver.run`. The
public direct-PyTorch solver deliberately forbids hidden device transfers, so
the PyTorch CUDA lanes use a named deployment adapter whose `run` consists of
explicit H2D, the canonical numerical call, synchronization, and explicit D2H.
This adapter is benchmark infrastructure, not a new public GradFlow backend.
Artifact-manifest verification performed by solver or adapter construction is
excluded and reported as setup. DVEB's lazy `ctypes` library load, first CUDA
runtime use, H2D/D2H copies, and numerical execution occur inside the first
call and are charged if the implementation performs them there. No earlier
CUDA tensor, allocation, synchronization, or kernel call is permitted in a
cold worker, except that loading an AOT package as configuration may initialize
its loader; pristine package loading remains a separate diagnostic.

E2 is measured in a new process for every observation so every CUDA
observation is genuinely first-use. It answers, “The application is running
and has an initial state; what happens the first time the user presses Run?”

### E3 — warm repeated `run`, CPU state already available

The framework, implementation, CUDA context, package, and any permitted cache
are already loaded. One full uncounted call using a separate output establishes
the warm state. Each counted timer then surrounds one complete call from the
same immutable caller-owned CPU input through a newly materialized CPU output.
Required H2D/D2H copies remain charged on every CUDA call.

Warm observations are gathered in independent worker blocks rather than one
unbounded process: six workers per lane and point, each with one uncounted
warmup followed by five counted calls, yield 30 counted observations. Output
objects are released between calls and allocator state is recorded; the input
is never replaced by a previous result.

E3 answers, “What does the next solve cost in an already-running application?”

### E4 — resident numerical execution

Where a public lane accepts an already-resident state and may return a
resident result, time the synchronized CFL-plus-SSP-RK3 interval without H2D
or D2H. This is a separate throughput endpoint and never substitutes for E1,
E2, or E3.

DVEB ABI v1 does not accept a device pointer. Its internally reported CUDA
`execution_seconds` is retained as a generated-kernel diagnostic, but DVEB is
marked **unsupported as a public E4 lane**. It cannot win E4 until a future ABI
version accepts and returns device-resident state. No transfer-subtracted
estimate may be presented as an E4 observation.

## Counted points

One uncounted capacity pilot may attempt `N={8,16,32,64,96,128,160}` at one
and ten steps. Pilot values establish only failure and memory-safe frontiers;
they support no performance claim.

Primary counted points are:

```text
steps=1:  N={8,16,32,64,96,128}
steps=10: N={16,32,64,128}
```

E1 and E2 receive one uncounted infrastructure smoke test and 30 independent
counted processes for every eligible primary lane and point. E3 receives the
six-by-five blocked design above. E4 uses six independent workers per lane and
point; each receives five warmup calls followed by one randomized five-call
counted block, yielding 30 counted calls where supported.

Cold `torch.compile` is limited to three independently empty-cache observations
at `(N,steps)={(64,1),(128,1),(128,10)}` because it characterizes compilation,
not steady runtime. Pristine AOT loading receives five independent empty-cache
observations at those points. Their smaller sample counts are explicit and
they are excluded from median-winner classification.

If a lane cannot execute a point, record the exact failure as unsupported,
out-of-memory, or defective. Do not resize the grid, change precision, reduce
the step count, or substitute an internal timer. The common capacity frontier
and every lane's independent frontier are separate results.

## Ordering, process isolation, and machine control

For E1 and E2, each repetition is a randomized block containing every eligible
lane at that point. The random seed is frozen in the preparation manifest.
Warm-worker and resident blocks are independently randomized. CPU and CUDA
blocks alternate where practical so thermal drift cannot align with one lane.

The campaign fixes and records:

- `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, and static scheduling;
- CPU model, physical/logical topology, affinity, governor, and observed
  frequencies;
- GPU model, driver, CUDA toolkit/runtime, clocks, temperature, power, and
  throttle reasons;
- PyTorch, Python, GNU compiler, glibc, and kernel versions;
- TF32 disabled and highest float32 matmul precision;
- allocator/cache environment variables;
- background-load observations; and
- peak host RSS and GPU allocated/reserved memory where available.

Telemetry collection occurs outside timed intervals. No individual timing is
dropped as an outlier. A documented machine interruption invalidates and
repeats the entire randomized block, preserving both the rejected block and
the reason.

## Statistics and decision language

Retain every raw observation. For each lane, point, and endpoint report count,
minimum, p05, median, mean, p95, maximum, median absolute deviation, peak
memory, and paired block wins where pairing exists. Report ratios and absolute
differences; do not report a ratio alone for sub-millisecond effects.

For each primary endpoint and point:

- **winner** means the lowest median eligible primary lane;
- **competitive** means median no more than `1.10` times the lowest median;
- differences below `0.25 ms` are practical ties unless the complete paired
  observations clearly separate them;
- **DVEB ceiling distance** is a forced DVEB lane's median divided by its
  matched native ceiling at the identical endpoint, when that ceiling exists;
  and
- cold, first-call, warm, and resident results must remain separate tables.

A bounded “DVEB has a useful region” conclusion requires correctness plus
either competitiveness at two adjacent counted grid sizes in one step stratum
or competitiveness in both step strata at one counted grid size. An isolated
sub-0.25-ms win is reported as a tie, not validation. Failure to meet this rule
is retained as a negative result; no lane, threshold, or point may be changed
after viewing counted data.

## Required output record

Commit:

- preparation manifest with all hashes and excluded preparation durations;
- pairwise full-array errors and hashes for losslessly compressed correctness
  arrays archived outside Git; the preparation manifest must identify and hash
  that archive so the large binary states do not enter ordinary repository
  history;
- every raw timing observation with block/order identifiers;
- stdout/stderr and exact command/environment for every failure;
- capacity and memory records;
- machine and thermal telemetry;
- analysis code and machine-readable summary; and
- a prose result distinguishing E1, E2, E3, and E4 conclusions.

The report must disclose that DVEB's CUDA E4 value is internal diagnostic
timing rather than a callable resident-state interface. It must also disclose
that the first-call and warm endpoints assume an already-running application
and therefore exclude Python import and initial-state construction.

## Stop condition

This turn stops after the protocol is committed with a clean tree. Do not
implement the harness, prepare packages, calibrate automatic placement, or run
capacity/correctness/performance observations yet.

The future measurement campaign stops after all gates and declared points
complete or fail explicitly, raw/analyzed evidence is committed, and the tree
is clean. Do not modify DVEB during measurement. Do not begin WENO-11/WENO-15,
device-pointer ABI v2, a new optimization campaign, or publication claims.

Do not push either repository without new explicit authorization.
