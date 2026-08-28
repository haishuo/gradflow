# FD/FV nonlinear Phase-5CR conservation-resolution protocol

Status: frozen after the initial Phase-5C failure and before resolution or
timing reclassification.

Freeze date: 2026-08-28 UTC.

## Purpose

The initial Phase-5C campaign is immutable. It found that CUDA complete solves
at `N=81` and `N=162` exceeded a full-solve conservation bound that did not
contain a time-step count. This resolution asks whether the failure represents:

1. nonconservative spatial RHS evaluation;
2. a reduction/measurement artifact;
3. actual floating-point drift accumulated by repeated SSP-RK3 updates; or
4. a materially incorrect numerical solution.

Phase 5CR collects no performance timing and changes no implementation. It may
reclassify preserved Phase-5C timings only if an independently justified
accumulated-roundoff contract passes. It never edits the initial record or raw
worker files.

## Immutable predecessor

`experiments/fd_fv_nonlinear/verify_phase5c_initial.py` must pass. The resolution
retains hashes for the initial aggregate, manifest, all raw records, the Phase
5C protocol, production numerical sources, and this protocol.

The initial failed bound and eligibility decisions remain historically correct.

## Mechanistic checks

For FD and FV, binary64, CPU and CUDA, and `N=(81,162)`:

1. evaluate the semidiscrete RHS on the exact initial projection and require

   ```text
   abs(dx*sum(rhs))
   <= 64*eps*dx*sum(abs(rhs)) + 2e-15;
   ```

2. execute one eager and one compiled SSP-RK3 step with the frozen `dt` and
   require mass change under the original single-update bound

   ```text
   B_single = 64*eps*dx*sum(abs(initial)) + 2e-15;
   ```

3. transfer final arrays to CPU and recompute mass using both tensor summation
   and `math.fsum`, requiring agreement within `2e-16` absolute; and

4. retain the initial complete-solve drift divided by step count for every
   method/device/mode/size. Similarity across FD/FV is causal evidence but is
   recorded rather than used to weaken a gate.

CPU/CUDA, eager/compiled, exact-oracle error, finiteness, dtype, device, and
host-visible-answer requirements remain unchanged.

## Prospective accumulated-roundoff contract

The original bound allocates the binary64 roundoff term once. A solve performs
`steps` repeated SSP-RK3 state updates. The mechanically accumulated full-solve
bound is frozen as

```text
B_accumulated = steps*(B_single - 2e-15) + 2e-15.
```

This applies the already frozen `64*eps*dx*sum(abs(initial))` allowance once per
step while retaining the fixed `2e-15` measurement slack once, not once per
step. It is selected from operation count, not fitted to the observed drift.

Every complete and cold record must satisfy both:

```text
mass_change <= B_accumulated
mass_change/steps <= B_single.
```

The second condition prevents a large solve from passing solely because it has
many steps. The record stores utilization ratios against both bounds.

## Timing reclassification rule

An initially ineligible complete or cold cell becomes
`eligible_under_phase_5cr` only when:

- every original nonconservation gate passed;
- its recorded mass change passes both prospective bounds;
- the fresh semidiscrete and one-step checks pass for its formulation/device
  stratum; and
- the immutable raw/aggregate hashes verify.

No duration, sample, statistic, error, or memory value is changed. Aggregates
and achieved-error target selections are recomputed from copies annotated with
the resolution eligibility. Original `eligible` fields remain untouched.

If any affected cell fails, it remains ineligible and Phase 5C remains failed.

## Resolution artifacts and stop

Write
`experiments/fd_fv_nonlinear/results/phase_5cr_20260828/resolution.json` and a
SHA-256 manifest. The independent verifier recomputes mechanistic gates, bounds,
reclassification, aggregate timing selections, and predecessor identities.

Stop after the resolution record, final Phase-5C interpretation, full configured
tests, coherent local commits, and clean tree. Do not rerun timing, optimize,
change the numerical method, add compensated updates, add AOT, begin Phase 5D,
or push without explicit authorization.
