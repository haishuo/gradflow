# Academic U4-A external-baseline compatibility result

Status: **complete; no external benchmark run**.

Subsequent outcome: U4-B qualified the bounded adapted operator; see
`ACADEMIC_U4B_RESULTS.md`. This document retains the earlier U4-A decision
state and does not retroactively treat stock OpenSBLI as matched.

Date: 2026-08-30 (UTC)

## Decision

OpenSBLI is the first external system selected for a bounded U4-B
qualification attempt. It is not yet an admitted performance lane. Stock
OpenSBLI does not execute the frozen GradFlow endpoint without adaptation, so
no OpenSBLI speedup or slowdown may currently appear in the paper.

PyWENO is retained as an independent symbolic/emitted-kernel cross-check.
JAX-Fluids and HOPE are retained as application-level prior art and candidates
for a future matched-error time-to-solution study. Neither is a direct
finite-difference operator baseline under the U4-A constitution.

## Why OpenSBLI remains promising

The inspected official OpenSBLI revision is
`e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d`. Its native source constructs
arbitrary odd-order finite-difference WENO-JS candidate polynomials, optimal
weights, smoothness indicators, left/right reconstructions, characteristic LF
splitting, and OPS CPU/GPU code.

A temporary exact-rational cross-check found that, after reversing OpenSBLI's
candidate enumeration, its offsets, candidate coefficients, and optimal
weights equal GradFlow's for orders 5, 7, 9, 11, 13, and 15. The generated
smoothness matrices also matched exactly for the completed orders 5, 7, and
9. The higher-order smoothness check was not completed in the bounded audit;
source inspection shows the same Jiang--Shu derivative-integral construction,
but that observation is not mislabeled as an executed equality check.

This establishes shared reconstruction mathematics. It does not establish an
equal executable.

## Why stock OpenSBLI is not admitted

Four differences are currently decisive:

1. OpenSBLI's WENO-JS epsilon is hard-coded to `1e-6`. GradFlow's scalar
   contract uses smoothness indicators multiplied by 12 and epsilon `1e-29`.
   With standard unscaled indicators, the mathematically equivalent external
   epsilon is `1e-29/12`.
2. The stock applications do not expose the frozen scalar positive-advection
   semidiscrete RHS endpoint. They are complete CFD applications with their
   own initialization, boundary, integration, and I/O contracts.
3. The shipped OpenSBLI performance benchmark is either TENO6 or fourth-order
   central differencing with WENO5 used as a filter. It is not direct
   WENO-JS5 applied to the frozen scalar flux.
4. OpenSBLI's Euler choices do not silently match GradFlow's duplicated
   endpoint, face-frozen Roe, per-line global LF speeds enlarged by 1.1, and
   12-scaled epsilon convention.

Accordingly, timing an existing OpenSBLI example would answer a different
question.

## Permitted U4-B adapter

The selected attempt is a small OpenSBLI application that uses OpenSBLI's own
WENO generator, LF machinery, divergence, periodic halo handling, and OPS
backend. The adapter may expose epsilon, specify scalar positive advection,
inject the frozen state, and isolate one RHS. It must retain a complete patch
and hash.

If that requires replacing OpenSBLI's flux splitting, reconstruction,
divergence, or OPS execution with newly written GradFlow-like code, OpenSBLI
will be downgraded to `building_block_only`. We will not manufacture an
“external” baseline by writing the baseline ourselves.

Before any timing, the adapted order-5 RHS must pass pointwise float64 parity,
constant-state, conservation, and smooth-convergence gates. Higher orders
enter only after order 5 qualifies.

## Other candidates

| System | U4-A class | Direct timing decision |
|---|---|---|
| OpenSBLI | `matched_operator_candidate` | Selected for adaptation and correctness qualification; not yet admitted |
| PyWENO | `building_block_only` | No direct system timing; may cross-check coefficients or emitted kernels |
| JAX-Fluids | `application_context_only` | Reserve for matched-error application comparison |
| HOPE | `application_context_only` | Reserve for finite-volume/application comparison |
| Native CUDA / DVEB / generated C++ | internal controls | Useful matched controls, but not independent external systems |

PyWENO's public runtime reconstructs one-dimensional cell-average data, while
its generator emits reconstruction building blocks. It does not supply the
frozen flux split, periodic divergence, PDE RHS, and resident GPU endpoint.

JAX-Fluids is principally a finite-volume Godunov CFD system. HOPE is a
two-dimensional finite-volume shallow-water system. Both are important
evidence that accelerator-native differentiable WENO software exists; neither
may be relabeled as the same FD-WENO-JS operator for a favorable ratio.

## Scientific consequence

U4-A closes the protocol ambiguity but not the manuscript's external-baseline
gap. The accurate statement remains:

> GradFlow has mathematically matched internal native and compiler controls,
> and OpenSBLI has been selected as the first independent external
> qualification candidate; no independent external performance result has yet
> been measured.

The next possible step is U4-B correctness qualification of the minimal
OpenSBLI adapter. Comparative timing remains downstream of that result.
