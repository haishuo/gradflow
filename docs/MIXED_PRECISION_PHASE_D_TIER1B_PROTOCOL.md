# Phase-D Tier-1b weight-normalization refinement protocol

Protocol freeze date: 2026-08-27 UTC.

Tier 1a completed before this refinement was defined; see
`MIXED_PRECISION_PHASE_D_TIER1A_RESULTS.md` and commit `85d48bf`.

## Motivation

Tier 1a grouped nonlinear-weight numerator formation and normalization in one
precision block. That grouped binary32 operation failed chiefly on small
signals over a unit background. Prior WENO-specific mixed-precision work
explicitly retained a high-precision weight sum, so the grouped failure cannot
determine whether binary32 numerator formation with binary64 normalization is
safe.

Tier 1b changes exactly that experimental resolution. It is an
evidence-driven refinement, not a post-hoc alteration of the Tier-1a record.

## Frozen change

The six Tier-1a blocks become seven:

1. `flux_split`;
2. `candidates`;
3. `indicators`;
4. `weight_formation`, ending with unnormalized nonlinear weights;
5. `weight_normalization`, including the sum and division;
6. `combination`; and
7. `divergence`.

Every binary32/binary64 assignment is evaluated: `2^7 = 128` policies for
each of orders 5, 7, 9, 11, 13, and 15, or 768 records. All numerical cases,
seeds, thresholds, safety rules, state precision, and classification logic are
unchanged from the Tier-1a frozen protocol.

The important targeted assignment is binary32 `weight_formation` with
binary64 `weight_normalization`. It receives no special tolerance and is not
assumed to pass.

## Performance eligibility

After verification, timing may include:

- the all-binary64 control;
- the all-binary32-internal endpoint;
- the Tier-1a indicator-only candidate;
- numerator-only and numerator-plus-indicator candidates if they pass; and
- any other numerically passing assignment that is not strictly dominated in
  both number of demoted blocks and recorded error.

Warm compiled device-resident timing on CUDA is required before claiming a GPU
speed benefit. CPU-only execution can establish the numerical classifications
but not the RTX 5070 Ti performance result.

## Unchanged boundary

Tier 1b remains a scalar qualification. It cannot bless an Euler/full-solver
default. Binary16, bfloat16, TF32, autocast, low-precision persistent state,
and gradient qualification remain outside this search.
