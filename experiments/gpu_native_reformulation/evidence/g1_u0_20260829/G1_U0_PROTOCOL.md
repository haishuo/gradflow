# G1 U0 Freeze Protocol

Status: frozen before the first counted U0 run.

Date: 2026-08-29 (UTC)

## Purpose

G1 asks only how fast the first deliberately GPU-native U0 formulation runs and
whether it remains numerically finite. It does **not** ask whether U0 is an
accurate replacement for Jiang--Shu WENO. Oracle comparison is prohibited
until the U0 source, executable, inputs, outputs, environment record, and G1
timings have been frozen and hashed.

## Frozen U0 contract

`u0_unique_f32_component_shared_density_local_lf_forward_euler_v1`

U0 uses unique periodic cells, FP32 fast arithmetic, componentwise split-flux
WENO-5, one density-derived nonlinear-weight pair shared by all five conserved
components, a six-cell face-local Lax--Friedrichs speed, face ownership, and
Forward Euler at CFL 0.1. The state stays device-resident during the counted
steps. These choices are intentionally accuracy-reckless.

## Build and machine

- compiler: CUDA 13 `nvcc`
- target: `sm_120`
- flags: recorded verbatim by `native_u0/build.sh`
- GPU: NVIDIA GeForce RTX 5070 Ti
- thread block: 256 threads, fixed before measurement
- no block-size or kernel-layout tuning is permitted in G1

The compiler output and machine inventory are part of the frozen evidence.

## Pre-counted checks

Compilation defects and runner defects may be fixed before the counted run.
The executable must complete a non-oracle `N=8`, one-step smoke test, report
finite output, and emit a full-state file of the expected byte length. No
comparison with any GradFlow, MATLAB, or Fortran result is allowed at this
stage.

## Counted matrix

All timings use five warmups and 30 measured repetitions.

| Role | Grid | Steps | Frozen full state |
|---|---:|---:|---|
| small damage specimen | 32^3 | 1 | initial and final |
| accumulated-damage specimen | 32^3 | 10 | final |
| primary frontier point | 128^3 | 1 | no |
| sustained frontier point | 128^3 | 10 | no |

Reported timing is CUDA-event device-resident time. Host initialization,
host-to-device input transfer, and device-to-host result transfer are excluded
and must not be presented as start-to-finish latency. The runner nevertheless
copies back and checks the final state after each measured case.

U0 uses one Forward Euler stage, while the qualified GradFlow/DVEB ceiling uses
three-stage SSP-RK3. Therefore G1 timings are frontier measurements, **not** a
valid speedup claim against that ceiling.

## Freeze and transition to G2

The evidence directory must contain:

- exact U0 source and build script;
- compiled executable;
- compiler log and machine/environment record;
- JSON output from every counted case;
- the two `N=32` final states and exact initial state;
- SHA-256 checksums covering every frozen artifact.

After those hashes are written, U0 is immutable. G2 may then compare the
frozen outputs against independently qualified GradFlow evidence. Any later
change becomes a separately named candidate and cannot overwrite U0.
