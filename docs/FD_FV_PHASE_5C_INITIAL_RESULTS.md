# FD/FV nonlinear Phase-5C initial performance result

Status: **frozen gate failed; timing preserved but affected complete/cold CUDA
cells are ineligible pending prospective resolution**.

Measurement source commit:
`7b0a989a8cad1b02ebd5a67446e2336c4a25675a`.

The prospectively frozen protocol is commit `1bc340c`. The immutable aggregate
is `experiments/fd_fv_nonlinear/results/phase_5c_20260828/benchmark.json`,
SHA-256
`e5b80ab950e20b50023f08426b5c2dea1f550aa77969dc3f119cdc5ff474ef3b`.

## Completed campaign

All 132 isolated workers completed:

- 60 complete-solve workers: three independent replicates of both methods,
  CPU/CUDA, and five sizes;
- 48 resident-step workers: the baseline matrix plus prospective crossover
  replication; and
- 24 process-launch-to-host-answer cold pilots.

Every step cell passed. Twelve CUDA complete-solve workers and eight CUDA cold
pilots failed the frozen full-solve conservation bound at `N=81` and `N=162`.
They remain ineligible in the initial aggregate even though their numerical
error, finiteness, eager/compiled parity, device, and dtype checks passed.

## Failure characterization

The frozen full-solve bound was

```text
64*eps*dx*sum(abs(initial)) + 2e-15 = 9.1054e-15.
```

Observed CUDA mass changes were approximately:

| N | Steps | Eager | Compiled |
|---:|---:|---:|---:|
| 81 | 531 | 1.23e-14 | 1.47e-14 |
| 162 | 1,685 | 3.95e-14 | 4.67e-14 |

The values are nearly identical for FD and FV. A separate timing-free
diagnostic transferred the final arrays to CPU and recomputed mass with both
ordinary and compensated-style Python summation; it reproduced the same drift.
This is state drift, not a CUDA reduction artifact.

Normalized drift is approximately `2.78e-17` per SSP-RK3 step and scales almost
exactly with 531 versus 1,685 steps. The agreement between FD and FV points to
roundoff accumulated by repeated time-integration updates rather than a loss of
semidiscrete conservation in one spatial formulation.

The original bound was suitable as a tight single-step/short-solve sanity gate
but did not contain an explicit step-count term. Phase 5C does not retroactively
change it. A prospective resolution must test semidiscrete conservation and
adopt a mechanically derived accumulated-roundoff bound before reclassifying
any timing.

## Provisional evidence not yet promoted

The resident-step matrix passed and its crossover replication is internally
intact. The initial aggregate reports compiled resident CUDA crossover brackets
of `N=2,048` for FD and `N=32,768` for FV. Those bracket decisions verified
within the initial schema, but this document does not promote a final Phase-5C
conclusion while the complete-solve gate remains unresolved.

Similarly, eligible CPU accuracy-target selections favored FV because its
higher observed accuracy required fewer cells. CUDA target selections at finer
errors are intentionally missing where the corresponding complete solves were
ineligible. Timing is preserved, not silently discarded or selectively quoted.

No implementation, threshold, or measured sample was changed after observing
this failure. No AOT artifact, optimization, mixed precision, DVEB change, or
new performance campaign was started.
