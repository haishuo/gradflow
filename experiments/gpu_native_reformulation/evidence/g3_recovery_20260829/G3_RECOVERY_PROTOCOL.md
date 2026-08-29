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

R1 otherwise retains every U0 numerical and scheduling choice. R2 is built
with R1's strict arithmetic and otherwise retains U0's componentwise
reconstruction, face-local LF speed, unique cells, and Forward Euler update.

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
within a componentwise scheme. Neither can establish whether characteristic
projection is necessary; that belongs to R3. No R1/R2 timing is comparable as
a qualified solver speedup because both still use Forward Euler and changed
spatial mathematics.
