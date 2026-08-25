# DVEB-inclusive matched deployment bake-off

Status: **FROZEN BEFORE COUNTED TIMING**.

Date: 2026-08-25

## Question

Where, if anywhere, does the current DVEB implementation compete with or beat
the eligible GradFlow deployment alternatives for the same mathematical
workload?

DVEB does not have to win every region.  Small, medium, large, resident, and
CPU-originating jobs are distinct regions and will be reported separately.

## Current capability boundary

The new matched DVEB portable pipeline supports the three-dimensional Shu
Euler WENO-5 workload only.  This campaign does not pretend it supports the
two-dimensional implementation or GradFlow's scalar pointwise oracle.

- **1-D:** DVEB trunk 001 is retained as separate scalar-WENO evidence.  It
  recorded a permanent NO-GO: compiled direct PyTorch was approximately
  1.9--2.1 times faster than that DVEB implementation.  The formulation and
  dtype differ from this 3-D campaign, so those numbers are contextual rather
  than merged.
- **2-D:** no matched DVEB portable artifact exists; status is unsupported and
  untested.
- **3-D:** this campaign performs the matched comparison.

GradFlow does not modify DVEB.  The input executable is the already-built
FMA-enabled artifact from DVEB commit `4c6a330`/branch state `5dddf95`, with
SHA-256:

```text
389946b2f1fe6c2df180e910f965814f3b6f67100ff5a73c38adfee1b88424a3
```

The module used to produce it has SHA-256:

```text
555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260
```

At campaign setup the DVEB repository had unrelated in-progress automatic
placement changes.  GradFlow treats DVEB as read-only and refuses any binary
whose hash differs from the frozen value above.

## Frozen mathematics

All 3-D lanes execute:

- float32 compressible Euler with five conserved components;
- the periodic isentropic vortex extruded uniformly in z;
- duplicated endpoints on `[0, 10]^3`;
- dimension-by-dimension finite-difference WENO-5;
- Roe characteristic projection at every face;
- Shu's central-flux-plus-nonlinear-correction algebra;
- `epsilon = 1e-6`;
- per-line, per-family global Lax--Friedrichs speeds enlarged by 10 percent;
- CFL 0.1 recomputed before every timestep;
- three complete SSP-RK3 stages per timestep; and
- complete final-state materialization in pageable host memory.

The authority is `shu_euler_torch.py` at GradFlow commit `948af9b`.  No lane
may substitute componentwise reconstruction, local LF, a fixed timestep,
unique periodic nodes, a reduced z calculation, or another dtype.

## Lanes

Primary deployable lanes:

1. repaired/extended Fortran CPU;
2. DVEB generated OpenMP C++ (`--target cpu`);
3. DVEB generated CUDA (`--target cuda`);
4. direct eager PyTorch CUDA;
5. prepared fixed-shape PyTorch AOTInductor CUDA.

Diagnostic ceilings:

6. separately authored matched OpenMP C++;
7. separately authored matched CUDA.

The diagnostic ceilings determine DVEB's implementation distance but are not
treated as high-level user-facing systems.  The convolutional PyTorch lane,
persistent `torch.compile` cache, and cold JIT are retained from the earlier
one-shot campaign but are not repeated: they were already dominated at the
accepted points or answer compilation questions rather than prepared runtime
placement.  Their prior records remain committed.

DVEB automatic placement is not included because it does not yet exist in a
committed, qualified artifact.  `min(DVEB CPU, DVEB CUDA)` will be reported as
an offline oracle envelope, clearly labeled as such; it is not attributed to
a selector.

## Correctness gate

Before counted timing, the exact frozen DVEB binary must be checked against
the independent PyTorch CPU result using complete arrays at:

- `N=6`, one step;
- `N=6`, ten steps;
- `N=32`, one step; and
- `N=128`, one step.

The existing justified float32 bound is `2e-5`; observed discrepancies and
CPU/CUDA differences are reported rather than hidden behind the bound.

The native ceiling and Fortran/PyTorch parity gates must also continue to
pass.  Checksums alone are not a correctness oracle.

## Timing endpoints

Two endpoints remain distinct:

1. **External fresh process:** an external monotonic clock surrounds process
   creation through successful exit.  It includes Python import for PyTorch,
   native loader/runtime initialization, pageable CPU initialization,
   required transfers, the solve, final host materialization, validation, and
   teardown.
2. **Resident execution:** the lane's internally synchronized interval after
   state placement and implementation loading.  It includes CFL plus all RK3
   stages for every requested step and the terminal synchronization, but not
   initial H2D or final D2H.

Fortran exposes only the complete fresh-process endpoint.  AOT package build
and one explicit extraction/cache preparation run are deployment preparation,
excluded from prepared run latency and reported separately.  Any compilation
inside an ordinary measured process would be charged, but no such lane is in
the primary campaign.

## Points and repetitions

One uncounted pilot may probe
`N in {8, 16, 32, 48, 64, 96, 128, 160}` for one step to establish failures
and memory safety.  Pilot values cannot support final performance claims.

Primary counted points:

- one step at `N in {8, 16, 32, 64, 96, 128}`;
- ten steps at `N in {32, 64, 128}`.

Every eligible lane receives one uncounted warmup and 30 counted fresh-process
runs per point.  Lane order is randomized within each repetition using a
fixed recorded seed.  A prepared AOT package and cache are specific to each
shape; build/preparation records and hashes are retained.

If a lane cannot execute a point, record the exact failure and classify it as
unsupported, out-of-memory, or defective.  Do not silently shrink the grid.
The common memory-safe frontier and each lane's independent capacity frontier
are separate results.

## Region and decision terminology

For each point and endpoint:

- **wins:** lowest median among eligible deployable lanes;
- **competitive:** median no more than 1.10 times the best deployable median;
- **specialist distance:** DVEB target median divided by its matched native
  ceiling median;
- **DVEB oracle envelope:** lower observed median of forced DVEB CPU and
  forced DVEB CUDA; and
- **selector required:** which forced DVEB target produced that envelope.

Differences smaller than 0.25 ms are reported as practical ties unless the
paired distribution is sufficiently separated to justify a stronger claim.
Means, medians, minima, maxima, p95, raw observations, and absolute
differences are retained.

The campaign may conclude that DVEB is valuable in only one bounded region.
It may also conclude that matching the handwritten ceiling is insufficient
to win complete latency because initialization, process startup, transfer, or
another system dominates.

## Environmental record

Record CPU model/topology, CPU frequency governor, OpenMP environment and
affinity, GPU model, driver, toolkit, GPU clocks/temperature/throttle state,
PyTorch version, compiler versions, binary/package hashes, random seed, and
peak memory where exposed.

The campaign runs on the available Forge machine only.  It makes no claim for
H100, other consumer GPUs, Apple Silicon, or other CPUs.

## Stop condition

Stop after correctness passes, pilot capacity is recorded, all primary
points complete or have explicit failures, raw and analyzed results are
committed, the dimensional capability boundary is documented, coherent local
commits exist, and the GradFlow working tree is clean.

Do not modify DVEB.  Do not push either repository without new explicit
authorization.  Do not begin WENO-11/WENO-15 or a new optimization campaign.
