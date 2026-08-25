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

GradFlow does not modify DVEB. The input executable is a hash-frozen copy of
the first automatic-placement artifact produced after DVEB commit `9b551ef`
(with the placement implementation still uncommitted in that checkout). The
executable, program, and mathematical-module SHA-256 values are respectively:

```text
884d874308dc7b1fd12f56491ae9addd85d1872ffcea4c2f26a0157c9c55c03c
c6e5bd916f951ff412eac99863a74f8c98e5e14b044097a7ad59fe26f704c381
555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260
```

The separately authored diagnostic ceiling was relinked by the same DVEB
development work and is frozen independently with SHA-256:

```text
873a9227196664398012e7d42a27e29ec9cd3610c45a4c61ab40a0688aed3caa
```

At campaign setup the DVEB repository had in-progress automatic-placement
changes. GradFlow treats DVEB as read-only, copies the executable before use,
and refuses any artifact whose hash differs from the frozen values above. The
uncommitted compiler state is a provenance limitation; this campaign qualifies
the exact executable, not a claim that DVEB can yet reproduce it from a clean
commit.

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
2. DVEB automatic placement using a hash-verified calibration model;
3. direct eager PyTorch CUDA;
4. prepared fixed-shape PyTorch AOTInductor CUDA.

Diagnostic ceilings:

5. DVEB forced six-thread CPU execution through its calibration-only hook;
6. DVEB forced CUDA execution through its calibration-only hook;
7. separately authored matched OpenMP C++;
8. separately authored matched CUDA.

The diagnostic ceilings determine DVEB's implementation distance but are not
treated as high-level user-facing systems.  The convolutional PyTorch lane,
persistent `torch.compile` cache, and cold JIT are retained from the earlier
one-shot campaign but are not repeated: they were already dominated at the
accepted points or answer compilation questions rather than prepared runtime
placement.  Their prior records remain committed.

The forced DVEB lanes are diagnostic and do not represent public user syntax.
Their lower median is reported as an offline two-target oracle envelope. The
automatic lane is the user-facing result and its selected target is recorded
on every run.

Before correctness or counted timing, GradFlow calibrates the frozen DVEB copy
without writing to the DVEB repository. A screen at the smallest calibration
sizes plus a timed `N=128`, ten-step probe eliminates the one- and two-thread
schedules before the full calibration: the one-thread probe alone took
107.16 seconds end-to-end after initialization. The retained contenders are
four, six, and twelve CPU threads plus CUDA. Calibration uses one warmup and three
observations at every pilot one-step size and every counted ten-step size. Its
raw measurements, model, hashes, and elapsed deployment work are retained.
Calibration is machine- and artifact-specific deployment preparation; it is
excluded from run latency, just like AOT compilation, and is never described
as free.

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

Every deployable lane receives one uncounted warmup and 30 counted fresh-process
runs per point. Diagnostic forced-target and ceiling lanes are exercised in
the correctness gate, pilot, calibration, and retained earlier ceiling study;
they are not repeated 30 times because they are not competing user-facing
systems. Lane order is randomized within each repetition using a fixed
recorded seed. A prepared AOT package and cache are specific to each
shape; build/preparation records and hashes are retained. The automatic DVEB
selector uses separate calibration observations and the 30 counted
observations remain independent.

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
