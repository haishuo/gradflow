# GradFlow Academic A1 claim and numerical-limit protocol

Status: **frozen before A1 implementation or new numerical-limit execution**.

Date: 2026-08-30 (UTC)

## Purpose

A1 freezes the bounded first-paper question, formulation contracts, claim
taxonomy, order-dependent numerical limits, and close-prior-art comparison
before the core performance matrix begins.

The first paper asks:

> How accurately, differentiably, and efficiently can one exact-generated
> Jiang--Shu finite-difference WENO implementation execute as maintainable
> ordinary PyTorch from orders 5 through 15, relative to mathematically
> matched CPU, compiler-generated, and native-GPU baselines?

A1 does not run a performance comparison, select a new epsilon, begin a new
mixed-precision search, extend native CUDA or DVEB, add a WENO family, or
expand the commercial solver.

## Existing immutable evidence

A1 consolidates rather than repeats the qualified records for:

- exact-generated scalar WENO-JS orders 5, 7, 9, 11, 13, and 15;
- generated Roe-characteristic 3-D Euler at the same orders;
- critical-point convergence and smooth convergence;
- scalar and characteristic mixed-precision qualification;
- periodic/transmissive Euler boundary and shock qualification;
- the Phase-C systematic prior-art review;
- the endpoint-explicit FD/FV study;
- the fixed WENO-5 deployment bakeoff and G0--G6 native study; and
- the ordinary-PyTorch face-ownership screen.

Every imported fact must name its source record and SHA-256 where one is
available. A1 may not rewrite an older artifact to simplify the synthesis.

## Frozen formulation contracts

### Scalar numerical-limit subject

- finite-difference WENO-JS of design order `p=2r-1`;
- orders `{5,7,9,11,13,15}`;
- exact-rationally generated candidate coefficients, optimal weights,
  smoothness matrices, and LDLT sum-of-squares factors;
- unique periodic nodes on `[0,1)`;
- scalar positive linear advection, `f(u)=u`, with explicit `alpha=1`;
- native float32 or float64 arithmetic as declared;
- smoothness indicators multiplied by 12;
- nonlinear power two; and
- canonical scalar epsilon `1e-29` except in the explicit epsilon sweep.

### Characteristic paper subject

The existing qualified Euler contract remains separate: duplicated periodic
endpoints, ideal-gas gamma `1.4`, face-frozen Roe projection, per-line
characteristic global LF speeds enlarged by `1.1`, epsilon `1e-6`, the same
12-scaled generated scalar reconstruction, and SSP-RK3. Scalar epsilon results
must not be transferred to this contract.

## New numerical-limit measurements

All new execution is deterministic, eager CPU, and single-threaded. No timing
is interpreted.

### N1. Coefficient and construction conditioning diagnostics

For each qualified order, record:

- substencil width;
- minimum and maximum exact optimal weight and their ratio;
- maximum candidate and full-stencil coefficient L1 norm;
- maximum numerator and denominator bit length across the exact payload;
- 2-norm condition number of every monomial cell-average moment matrix used
  for candidate and full-stencil reconstruction; and
- 2-norm condition number of each smoothness matrix restricted to the
  Euclidean subspace orthogonal to the constant vector.

Condition numbers are binary64, basis-dependent diagnostics—not proofs of
intrinsic WENO stability. Exact polynomial reproduction and positivity remain
the mathematical correctness evidence.

### N2. Roundoff-floor sweep

For both float32 and float64, evaluate the canonical scalar RHS for

```text
u(x) = sin(2*pi*x) + 0.15*cos(6*pi*x)
N    = {32,64,128,256,512,1024,2048,4096,8192}
```

against the analytic `-du/dx`. Input values are evaluated in float64 at the
exact grid locations and then cast to the execution dtype. Errors are reduced
in float64. Record L1, L2, Linf, finiteness, conservation, the sampled minimum
L2 error, and the first sampled post-minimum point exceeding the preceding L2
error by 5%. The sampled minimum is an observed floor on this problem and
machine, not a universal lower bound.

Conservation must satisfy
`abs(sum(rhs)) <= 32*eps(dtype)*sum(abs(rhs))`. A nonfinite output or failed
conservation gate is a declared numerical failure and remains in the record.

### N3. Epsilon sensitivity sweep

In float64 at `N=128`, evaluate the smooth family above and the critical family
`u(x)=sin(2*pi*x)^3` at amplitudes `{1,1e-3,1e-6}` and epsilons
`{1e-40,1e-29,1e-20,1e-12,1e-6}`. Record normalized errors, conservation, and
the RHS difference from the `1e-40` execution at the same amplitude.

For descriptive classification, an epsilon produces a material change when
either its error is outside `[0.5,2]` times the `1e-40` error or its normalized
RHS difference exceeds `1e-8`. This threshold identifies scale dependence; it
is not a pass/fail accuracy criterion and cannot select a new default.

## Claim taxonomy

Every final statement receives exactly one status:

- `established`: supported by mathematical proof or an independent oracle;
- `observed`: directly measured under a frozen contract;
- `inferred`: a bounded explanation consistent with observations but not
  causally isolated;
- `untested`: outside current evidence; or
- `prohibited`: contradicted by prior art or beyond the evidence.

Negative results and hardware/software limitations remain visible. “First,”
“only,” “universal,” “production CFD,” “real time,” and aerospace-readiness
claims are prohibited.

## Prior-art comparison freeze

A1 derives one property-by-property table from the Phase-C records for
OpenSBLI, PyWENO/PyClaw, HOPE, JAX-Fluids, and JAX-Shock. Unknown remains
unknown. The comparison is current to the Phase-C review date and is not an
external novelty determination. That audit remains an A4 release gate.

The FD/FV campaign and G-series are supporting studies, not separate headline
claims in the first paper. This choice limits the replication surface while
retaining their bounded findings as context and appendices.

## Required outputs

- `docs/ACADEMIC_A1_CLAIM_MATRIX.md`;
- `docs/ACADEMIC_A1_NUMERICAL_LIMITS.md`;
- `docs/ACADEMIC_A1_PRIOR_ART_COMPARISON.md`;
- `docs/ACADEMIC_A1_RESULTS.md`;
- machine-readable new measurements and a source-record index under
  `experiments/academic_a1/evidence/`;
- an offline semantic/checksum verifier and regression test; and
- updated academic scope/roadmap status.

## Stop condition

A1 closes when all outputs exist, every imported and new record verifies,
the claim and numerical-limit tables agree with their machine-readable
sources, no unresolved missing correctness experiment remains before A2,
tests pass, coherent local commits exist, and the working tree is clean.

Do not push without explicit authorization. After A1, proceed directly to the
separately frozen A2 arbitrary-order performance matrix.
