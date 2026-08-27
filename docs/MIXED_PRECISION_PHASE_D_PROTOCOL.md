# Phase-D mixed-precision WENO-JS protocol

Protocol freeze date: 2026-08-27 UTC.

## Research question

For generated finite-difference WENO-JS, which coherent calculations can be
performed in IEEE binary32 while retaining a binary64 state and satisfying a
declared, problem-wide accuracy contract?

This phase tests a bounded version of that question. It does not assume that
one precision assignment is universal, and it does not treat speed as evidence
of acceptability until the numerical gate passes.

## Prior evidence and claim boundary

Mixed precision is established in scientific computing and CFD. In the most
direct WENO precedent found for this protocol, Field, Gottlieb, Grant,
Isherwood, and Khanna evaluated nonlinear WENO weights in double precision
inside an otherwise quadruple-precision application and reported a 3.3-times
GPU speedup. Their result was application- and region-dependent and warned
about strong shocks. It does not establish that binary32 weights are safe in a
binary64 WENO solver.

Other GPU WENO work has compared complete single- and double-precision
executions. The Phase-C review did not find a published exhaustive assignment
of binary32/binary64 to the mathematical blocks below across generated WENO-JS
orders 5--15. This is a literature-review finding, not proof of absence.

Primary starting points:

- <https://doi.org/10.1007/s42967-021-00129-2>
- <https://doi.org/10.1016/j.compfluid.2017.11.012>

Novelty and publishability remain unclaimed pending execution, a refreshed
systematic search, and external review.

## Frozen scope

Tier 1 exhaustively evaluates scalar, periodic, global-Lax--Friedrichs
finite-difference WENO-JS at orders 5, 7, 9, 11, 13, and 15. The input state,
the persistent Runge--Kutta state, and the returned RHS remain binary64.

Six internal calculation blocks independently select binary32 or binary64:

1. `flux_split`: physical-flux evaluation, wave-speed evaluation when needed,
   and positive/negative LF splitting;
2. `candidates`: candidate-polynomial evaluation;
3. `indicators`: Jiang--Shu smoothness-indicator evaluation;
4. `weights`: scaled nonlinear-weight formation and normalization;
5. `combination`: weighted candidate combination at each face; and
6. `divergence`: conservative face-flux difference and division by spacing.

All `2^6 = 64` assignments are evaluated at every qualified order, for 384
order/assignment pairs. Conversion to a lower precision is explicit and never
changes device. Conversion back to binary64 cannot restore lost information.

Tier 1 intentionally excludes characteristic projection, Euler flux algebra,
positivity, boundary closures, and low-precision persistent state storage.
Those require a separately frozen Tier-2 full-solver protocol. A Tier-1 pass
must never be described as proving a safe mixed-precision CFD solver.

## Frozen numerical cases

Every assignment is compared against the identical all-binary64 generated
implementation, using the same binary64 inputs and coefficients:

- smooth multimode linear advection at `N = 48, 96, 192`;
- a smooth critical-point family, `sin(2*pi*x)^3`, at `N = 64, 128, 256`;
- near-constant waves `1 + A*sin(2*pi*x)` for `A = 1e-4, 1e-6, 1e-7` at
  `N = 256`, normalized to the derivative signal rather than the background;
- a discontinuous periodic square-wave flux derivative at `N = 257`;
- a seeded random Burgers state at `N = 257` for conservation and parity; and
- 40 fixed SSP-RK3 steps of smooth linear advection at `N = 128`, CFL `0.2`,
  with the persistent stage state retained in binary64.

The near-constant family is deliberate. A split that appears safe only when
all signals are large relative to binary32 spacing is not generally safe.

## Recorded metrics and classifications

For each case, the record retains maximum and RMS differences from the
all-binary64 oracle, normalized by a declared physical signal scale. It also
retains analytic error where available, observed convergence, finiteness,
constant-state preservation, and conservation residual.

Assignments receive three descriptive parity classifications; these are not
claims of universal solution accuracy:

| Class | Maximum normalized error | RMS normalized error |
|---|---:|---:|
| `tight` | `1e-5` | `1e-6` |
| `engineering` | `5e-4` | `1e-4` |
| `coarse` | `1e-2` | `2e-3` |

To receive any class, every value must be finite, constant-state RHS must be
no larger than `5e-5`, and the binary64 sum of the returned conservative RHS
must be no larger than
`64 * eps32 * sum(abs(rhs))`. A class is assigned only when all frozen cases
meet that class's two parity bounds. The record also exposes every raw metric
so later applications can apply a different tolerance without rerunning the
search.

These thresholds characterize deviation from the binary64 implementation.
They do not replace comparison with an independent mathematical solution.
Smooth-case analytic errors and rates are therefore recorded alongside
parity.

## Performance protocol

Accuracy search precedes timing. After classification, the all-binary64 and
all-binary32-internal endpoints plus the distinct nondominated passing
assignments are eligible for timing. Timing is device-resident and separates:

- eager execution;
- warm `torch.compile(..., fullgraph=True)` execution; and
- first-call compile latency.

Compilation latency is recorded but is not included in warm execution time.
Transfers are excluded from device-resident timing and must be reported as a
separate endpoint in any later full-solver campaign. CUDA events are used for
GPU execution. Median and interquartile range are recorded after warm-up; no
single-run number can support a performance conclusion.

The exhaustive numerical search is hardware-independent in intent. Timing and
the fastest passing assignment are hardware-specific. Results from the GeForce
RTX 5070 Ti must not be generalized to A100/H100 binary64 behavior.

## Reproducibility and refusal rules

- The protocol commit must predate the result record.
- The runner refuses to overwrite an existing result directory.
- Records include source revision, dirty state, Python/PyTorch/CUDA versions,
  device identity, policy enumeration, thresholds, seeds, and command line.
- A verifier recomputes hashes, counts all 384 pairs, checks classification
  from raw metrics, and rejects missing or non-finite records.
- Timing failures, compiler failures, and out-of-memory results are data and
  must not be silently dropped.
- Binary16, bfloat16, TF32, autocast, and reduced-precision matrix modes are
  outside this phase.

## Stop boundary

Phase D may conclude the Tier-1 scalar search and identify candidate splits
for Tier 2. It may not recommend a production default until the selected
splits pass characteristic Euler tests including smooth flow, critical points,
Sod, Shu--Osher, a stronger shock/positivity case, long-time integration,
gradient agreement, and full-solver performance.
