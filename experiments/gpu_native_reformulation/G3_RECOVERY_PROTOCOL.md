# G3 Correctness-Recovery Protocol

Status: frozen after G2 and before the first recovery-candidate result.

## Purpose

G3 restores one numerical property at a time to the immutable U0 frontier.
Candidates are cumulative. They are diagnostic contracts, not GradFlow
backends, and they cannot pass by relaxing an existing oracle tolerance.

## First recovery block

This first bounded block implements the two predeclared componentwise steps:

| Candidate | Change relative to predecessor |
|---|---|
| R1 | Remove CUDA fast-math contraction/approximation; request precise FP32 division and square root. |
| R2 | Compute distinct JS weights from each positive and negative split-flux component instead of sharing density-derived weights. |
| R3 | Restore face-local Roe characteristic projection and back-projection while retaining U0's scalar face-local LF speed. |
| R4 | Restore per-line characteristic-family LF maxima and Shu's 1.1 enlargement. |
| R5 | Restore three-stage SSP-RK3 while keeping unique cells and the face-once schedule. |
| R6 | Restore Shu's difference-form nonlinear correction, epsilon scaling, central flux, and operation structure. |

R1 otherwise retains every U0 numerical and scheduling choice. R2 is built
with R1's strict arithmetic and otherwise retains U0's componentwise
reconstruction, face-local LF speed, unique cells, and Forward Euler update.
R3 retains strict arithmetic, unique cells, the scalar local LF policy, face
ownership, and Forward Euler, but computes independent WENO weights for each
Roe characteristic family and projects the numerical face flux back to the
conserved basis.
R4 adds one line reduction per axis and stage, supplies separate minus,
center, and plus characteristic speeds, and retains the face-owned flux array
and Forward Euler update.
R5 computes one CFL timestep per complete step and performs the qualified
three-stage Shu--Osher update with fresh line speeds and face fluxes at every
stage.
R6 retains unique periodic cells because the omitted duplicate endpoint is
algebraically redundant. It restores the ancestral difference-form WENO
correction and central fourth-order flux so that the numerical contract, not
the CPU storage convention, is tested.

## Measurements

For each candidate:

- freeze source, executable, compiler diagnostics, one-step `N=32` output,
  and 30-sample resident timings at `N=32` and `N=128`;
- compare the `N=32` output to the same float64 characteristic WENO-5
  Forward-Euler oracle and qualified SSP-RK3 oracle used in G2;
- report error relative to the oracle update, update magnitude and direction,
  conservation, positivity, register count, spilling, and incremental latency;
- preserve U0 as the baseline and never overwrite its evidence.

The counted timing convention remains five warmups and 30 repetitions. The
GPU is checked between candidate batches after the unexpected host restart.

## Interpretation

R1 diagnoses floating-point execution semantics. R2 diagnoses shared sensing
within a componentwise scheme. R3 diagnoses characteristic reconstruction
without simultaneously restoring the ancestral line-global split. R4 measures
that split independently of time integration. R5 restores the temporal
contract. R6 targets the complete spatial and temporal contract. No R1--R6
timing is comparable as a qualified solver speedup because all still use
Forward Euler and altered spatial mathematics.
