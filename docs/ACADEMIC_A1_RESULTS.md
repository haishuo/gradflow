# GradFlow Academic A1 results

Status: **A1 complete; A2 is now the active academic gate**.

Date: 2026-08-30 (UTC)

## Decision

A1 closes without discovering a missing correctness experiment that must
precede performance work. The exact first-paper subject, claim vocabulary,
formulation contracts, numerical limits, prior-art boundary, and explicit
exclusions are now frozen.

The first paper remains:

> An empirical systems-and-numerics characterization of exact-generated
> Jiang--Shu finite-difference WENO in maintainable ordinary PyTorch from
> orders 5 through 15, relative to mathematically matched execution baselines.

It is not a new WENO formula, a “first PyTorch WENO” claim, a universal FD/FV
claim, or a general aerospace CFD product.

## New A1 findings

1. Exact construction remains mathematically sound through order 15, while
   floating-point difficulty grows sharply. The raw-monomial full moment
   condition rises from about `5.3e1` at order 5 to `3.25e12` at order 15;
   exact numerator/denominator complexity reaches 145/148 bits.
2. Higher order obtains smaller error on coarser grids but reaches its sampled
   roundoff floor earlier. Float64 onset moved from beyond `N=8192` at order 5
   to `N=256` at orders 13 and 15; float32 onset moved from `N=512` to `N=64`.
3. The canonical scalar epsilon `1e-29` was below the material-change boundary
   relative to `1e-40` throughout the frozen amplitude range. Larger epsilons
   changed scale-dependent cases, but the sweep provides no reason to change
   the default or transfer a scalar policy to characteristic Euler.
4. All new executions remained finite and conservative. No canonical source,
   epsilon, precision policy, or qualified-order set changed.

## Paper-structure decision

The main paper will center the exact-generated ordinary-PyTorch order sweep.
Mixed precision and face ownership are bounded representation findings. The
native CUDA result is a fixed WENO-5 ceiling/control. FD/FV and G0--G6 remain
supporting studies or appendices rather than independent headline claims.
DVEB remains optional.

## Remaining gates

- **A2:** formulation-matched arbitrary-order performance matrix;
- **A3:** one independently gradient-checked sensitivity or inverse use; and
- **A4:** second-machine replication, external audit, licensing decision,
  clean-room artifact, and citable release candidate.

No further numerical-method detour is authorized before A2.

## Artifacts

- claim matrix: `docs/ACADEMIC_A1_CLAIM_MATRIX.md`;
- numerical limits: `docs/ACADEMIC_A1_NUMERICAL_LIMITS.md`;
- prior-art comparison: `docs/ACADEMIC_A1_PRIOR_ART_COMPARISON.md`;
- new raw record:
  `experiments/academic_a1/evidence/a1_20260830/numerical_limits.json`;
- derived claim/source record:
  `experiments/academic_a1/evidence/a1_20260830/consolidation.json`; and
- frozen protocols: `docs/ACADEMIC_A1_PROTOCOL.md` and
  `docs/ACADEMIC_A1_PROTOCOL_CLARIFICATION.md`.
