# Academic U4-B OpenSBLI qualification results

Status: **closed — `matched_operator_adapted_qualified`**.

Date: 2026-08-30 (UTC)

U4-B tested correctness only. OpenSBLI's automatically printed execution
timings remain in raw logs as unavoidable program output, but they are not
interpreted, compared, or admitted as performance evidence.

## Decision

The pinned OpenSBLI revision can express and execute the frozen GradFlow scalar
FD-WENO-JS5 operator after a bounded generalization patch. All four frozen
gates pass. The adapted OpenSBLI lane is therefore eligible for a separately
designed U4-C performance qualification; it is not yet an admitted timing
baseline.

## What remained native OpenSBLI

OpenSBLI generated and executed its own:

- Jiang--Shu smoothness indicators;
- nonlinear weights and candidate reconstruction;
- characteristic flux transformation and native LLF split;
- conservative flux divergence;
- periodic halo exchange;
- generated C/OPS kernels; and
- sequential OPS runtime path.

The one-wave physics object supplies only the scalar eigensystem
`lambda=L=R=[1]`. OpenSBLI's native LLF therefore evaluates `alpha=1` at every
face, which is mathematically identical to the frozen global-LF policy for
constant-speed positive advection.

## Bounded adaptation

The retained patch:

1. preserves a supplied custom eigensystem instead of losing it to an
   uninitialized Euler local;
2. skips Euler-specific global reductions on an LLF path;
3. creates inverse-hoisting temporaries only when their expressions occur;
4. avoids declaring gamma for a non-Euler expression; and
5. exposes the existing WENO-JS epsilon as a constructor argument without
   changing its formula.

No WENO coefficient, smoothness indicator, nonlinear weight, reconstruction,
split, divergence, boundary exchange, or OPS kernel was reimplemented in the
adapter. A separately hashed generated-source hook exports the first native
residual immediately after its evaluation and exits before the first RK
update.

OpenSBLI declares SymPy 1.1, whose parser predates Python 3.11's AST and
standard-library layout. The adapter provides import/AST compatibility shims
only; generation ran with SymPy 1.1, not GradFlow's newer SymPy.

## Correctness results

At `N=64`, maximum absolute differences from the canonical GradFlow float64
RHS were:

| case | maximum absolute difference | tolerance |
|---|---:|---:|
| `u_a` | `2.7000623958883807e-13` | `2e-12` |
| `u_b` | `2.842170943040401e-14` | `2e-12` |
| constant `0.37` | `0` | `2e-12` |

All arrays were finite. The OpenSBLI residual sum was exactly zero for both
nonconstant pointwise cases; every OpenSBLI and GradFlow case passed the frozen
roundoff-scaled conservation bound.

For `u=sin(2*pi*x)`, the OpenSBLI L2 errors and successive rates were:

| N | L2 error | rate |
|---:|---:|---:|
| 40 | `5.812108063410716e-05` | — |
| 80 | `1.6997017025971051e-06` | `5.095708046052657` |
| 160 | `5.094929837658295e-08` | `5.060075486425139` |
| 320 | `1.5542499208619103e-09` | `5.034771868626690` |

Every measured rate exceeds the precommitted `4.8` floor. Across these four
grids, OpenSBLI and GradFlow RHS arrays differed by at most
`3.552713678800501e-13`.

## Scope of the result

This establishes an independently maintained, mathematically matched external
operator for the scalar order-5 subject. It does not establish OpenSBLI CUDA
agreement, performance competitiveness, arbitrary-order equivalence beyond the
U4-A source audit, Euler equivalence, or a paper-ready external benchmark.

U4-C may now freeze a timing constitution. Before any CUDA timing is admitted,
the generated CUDA lane must pass CPU/CUDA agreement under a declared policy.

## Evidence

Frozen evidence lives in `experiments/academic_u4b/evidence/u4b_20260830/`.
It contains the exact arrays, qualification record, generated-source hashes,
commands, logs, and SHA-256 manifest. Run
`python experiments/academic_u4b/verify_u4b.py` for an offline checksum and
semantic verification.
