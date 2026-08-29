# G4 Face-Once Schedule Performance Protocol

Status: frozen before the G4 control adaptation is built or any G4 timing is
observed.

Date: 2026-08-29 (UTC)

## Question and claim boundary

Does the R6Q unique-cell, face-once execution schedule reduce elapsed time and
workspace relative to the exact-math duplicated-endpoint, cell-recompute CUDA
schedule on the same three-dimensional periodic Shu characteristic
FD-JS-WENO-5 workload?

R6Q did not pass every frozen G3 criterion and has no autograd ABI. G4 is
therefore a **performance characterization of a non-admitted research
candidate**, not a qualified-backend bakeoff. A favorable G4 result cannot
erase the G3 non-admission decision or be reported as a production GradFlow
speedup.

No source, launch geometry, compiler flag, size, repetition count, threshold,
or metric may be tuned after primary results exist.

## Compared schedules

### Face-once candidate

Use the exact checksummed R6Q artifact frozen by G3:

```text
r6q_arbitrary_state_rhs_unique_strict_f32_shu_face_once_v1
```

It stores `N^3` unique periodic cells and one five-component numerical flux
for every cell-owned directional face.

### Cell-recompute control

Start from the separately authored DVEB native ceiling at DVEB commit
`bd4bc791b6e8f4a2ba2b0b28ecdb3086a4d3d97c`. Preserve the upstream hashes in a
provenance record. The GradFlow-local G4 copy may change only:

- its CLI to read the same component-major unique-cell FP32 input as R6Q;
- deterministic expansion to its native `(N+1)^3` duplicated periodic layout;
- metadata identifying the G4 control contract; and
- its resident-loop timer from synchronized host time to CUDA events.

Its CFL, line-speed, characteristic reconstruction, RHS, SSP-RK3, launch
geometry, duplicated storage, cell-owned reconstruction, and compiler
optimization flags must not change. It remains a cell-recompute control, not a
new candidate.

## Pre-timing validity gate

Before timing, use the same `N=32` unique periodic vortex input for both lanes.
Run one step, crop the control's duplicated result to unique cells, and require:

- both outputs finite with positive density and pressure;
- candidate versus control maximum absolute difference at most `2e-5`; and
- each output file to have its declared exact byte length.

Failure stops G4. No performance result from a failed control pair is counted.

## Frozen primary matrix

```text
N = {8, 16, 32, 64, 128, 192, 256}
steps = {1, 10}
warm-up processes per lane and configuration = 3
paired counted repetitions per configuration = 30
random seed = 20260829
```

For each size, generate one deterministic periodic-vortex unique-cell FP32
input and reuse its exact bytes for both schedules. Input files are temporary
campaign material and are hashed in the result record; they need not be
committed because the generator and hashes are frozen.

Every repetition contains both lanes. Their order is independently shuffled
with the frozen seed. Each lane runs in a fresh process with one internal
observation. No result is discarded as an outlier.

## Timing endpoints

Record both endpoints for every observation:

1. **Resident numerical loop:** CUDA-event elapsed time beginning after device
   allocation and input upload and ending after the requested synchronized
   SSP-RK3 steps. This is the primary schedule metric.
2. **Fresh process:** parent-observed wall time from process launch through
   exit. This includes dynamic loading, CUDA context creation, CPU input read
   and adaptation, device allocation and transfers, numerical execution,
   result download, checksum, and teardown. This is a deployment diagnostic,
   not a pure schedule metric.

Both programs copy the final state back and check finiteness after the resident
timer. Neither writes the result during counted observations.

For each lane and endpoint report all raw observations, median, mean, sample
standard deviation, minimum, maximum, median absolute deviation, and bootstrap
95% confidence interval for the median. Also report paired
`cell_recompute / face_once` ratios and order-stratified medians. Bootstrap
uses 20,000 resamples and seed 20260829.

The face-once scheduling hypothesis is supported at the two preregistered
primary points `(N,steps)=(128,1)` and `(128,10)` only if the paired resident
median ratio exceeds `1.10` and its bootstrap 95% lower bound exceeds `1.0` at
both points. All other sizes characterize scaling and crossover behavior.

## Memory, machine, and thermal records

Record declared peak allocated bytes for both schedules. This is an allocation
account, not measured peak physical memory.

Before and after every counted pair, record GPU temperature, P-state, SM and
memory clocks, power draw, utilization, and memory use. Record driver, CUDA
compiler, GPU identity, CPU identity, operating system, source hashes, binary
hashes, build commands, and git identities.

Stop the campaign without replacing observations if:

- GPU temperature reaches 80 C;
- either process fails or reports nonfinite output;
- input or output sizes are inconsistent; or
- the machine reboots or loses the CUDA device.

No clock locking, power-limit change, or cooling intervention is part of G4.

## Bounded profiler record

After the timing campaign, profile one `N=128`, one-step observation per lane.
Use Nsight Systems for launch order and kernel duration. Attempt one bounded
Nsight Compute basic collection per lane for occupancy/register/memory
diagnostics. If hardware-counter access is unavailable, record the exact tool
failure rather than changing permissions or substituting estimates.

Profiler runs are diagnostic and excluded from timing statistics. No code may
be changed in response to profiler results during G4.

## Evidence and stopping point

Freeze the protocol, control provenance, exact adapted source, build recipe,
binaries, compiler logs, validity outputs, raw randomized observations,
analysis JSON, machine and profiler records, and SHA-256 manifest.

G4 ends after interpreting the schedule effect, scaling, process overhead,
workspace tradeoff, order/thermal diagnostics, and profiler evidence. It does
not repair G3, add autograd, optimize either schedule, admit a backend, or begin
arbitrary-order GPU work.
