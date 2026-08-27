# Phase-D Tier-2 characteristic-Euler mixed-precision protocol

Protocol freeze date: 2026-08-27 UTC.

Tier 1a, Tier 1b, and the scalar RTX performance campaign are complete and
immutable. Tier 2 asks whether the scalar-qualified precision seam remains
correct when embedded inside GradFlow's already-qualified Roe-characteristic
Euler formulation.

## Admitted policies

Exactly four scalar policies enter Tier 2:

1. all internal WENO-JS blocks binary64;
2. binary32 smoothness indicators only;
3. binary32 unnormalized nonlinear-weight formation only; and
4. binary32 indicators plus binary32 unnormalized weight formation.

No failed scalar policy may be reconsidered here. Weight normalization,
candidate reconstruction, and face combination remain binary64.

Every Euler-specific calculation remains binary64: conservative state and RK
storage, primitive and physical-flux algebra, global LF reductions, Roe
averages and eigensystems, characteristic projection and back-transform,
conservative divergence, CFL calculation, and SSP-RK3 accumulation. Tier 2
does not search these operations or silently use autocast.

The existing default Euler functions remain bit-for-bit unchanged when no
precision policy is supplied. An explicit experimental policy changes only
the shared `WENOJS.reconstruct_stencils` computation.

## Orders and inherited classes

Smooth local gates cover all qualified orders 5, 7, 9, 11, 13, and 15.
Time integration, shocks, gradients, compilation, and later performance use
representative orders 5, 11, and 15, matching the frozen Phase-B shock gate.

At order 5, indicator-only and combined demotion inherit the Tier-1b
`engineering` class; weight-formation-only is `tight`. At orders 7--15 all
three candidates inherit `tight`. Tier 2 may retain or reject an inherited
class but cannot improve it.

## Local Euler gates

Every admitted policy is compared with the all-binary64 Euler path using the
same binary64 state.

For every order:

- the Phase-B periodic entropy-wave RHS is evaluated at 37 points under both
  periodic and transmissive boundaries;
- near-equilibrium entropy waves use density
  `1 + A*sin(2*pi*x)`, velocity `0.7`, pressure `1`, `N=257`, and
  `A = 1e-4, 1e-6, 1e-7`, under both boundaries;
- uniform states must retain maximum absolute RHS at most `2e-12`;
- periodic and transmissive boundary-flux conservation ratios must remain at
  most 64; and
- the output must remain finite and binary64 on the input device.

Near-equilibrium errors are normalized componentwise by the known conserved
derivative signal, never by the order-one background. Other RHS errors use the
maximum absolute all-binary64 component as scale.

An inherited `tight` candidate requires maximum normalized parity at most
`1e-5` and RMS normalized parity at most `1e-6` across every local case. An
inherited `engineering` candidate requires `5e-4` and `1e-4`. These are the
unchanged scalar class thresholds.

Representative orders also evaluate a 3-D duplicated-periodic entropy wave
and one 3-D periodic-vortex SSP-RK3 step on at least 17 unique cells per axis.
The same class thresholds apply componentwise.

## Repeated-step smooth gate

For representative orders, a periodic one-dimensional entropy wave is
advanced for one complete advection period on 64 physical points, CFL `0.1`,
with exact final-step shortening. Persistent state and RK arithmetic remain
binary64.

Compared with the all-binary64 terminal state:

- a `tight` policy requires componentwise normalized L1 at most `1e-4` and
  normalized Linf at most `2e-3`;
- an `engineering` policy requires normalized L1 at most `5e-4` and
  normalized Linf at most `1e-2`.

Every stage must retain finite positive density and pressure. The mixed
analytic L1 error may not exceed `1.05 *` the all-binary64 analytic error plus
`64 * eps32` times the component signal scale.

## Frozen shock gate

Orders 5, 11, and 15 run the Phase-B Sod problem at 800 points to time 0.2 and
the Phase-B Shu--Osher problem at 800 points to time 1.8, using transmissive
boundaries, CFL `0.1`, exact final-step shortening, and stage-by-stage physical
admissibility checks.

The all-binary64 terminal arrays and independent Phase-A oracles already
committed by Phase B are immutable controls. Every mixed candidate must:

- complete every step with finite positive density and pressure;
- continue to satisfy the original Phase-A finest-grid L1, Sod wave-location,
  Shu--Osher correlation, and total-variation thresholds; and
- agree with the committed all-binary64 Phase-B primitive state componentwise
  within its inherited terminal bounds: normalized L1 `1e-4` and Linf `2e-3`
  for `tight`, or L1 `5e-4` and Linf `1e-2` for `engineering`.

Normalization uses each binary64 primitive component's range, with a minimum
scale of one only for components whose range is smaller than one. This
terminal comparison is deliberately less strict than a single-RHS comparison
because shock position and thousands of nonlinear steps amplify roundoff.

No clipping, positivity limiter, adaptive epsilon, WENO substitution,
output-dependent retry, or relaxed independent-oracle threshold is permitted.

## Differentiation, device, and compiler gates

For each representative order and admitted policy:

- the existing boundary-sensitive one-step objective must produce finite,
  nonzero gradients;
- its directional derivative must continue to agree with centered finite
  differences within relative `2e-5` or absolute `2e-7`;
- the mixed gradient must agree with the all-binary64 gradient with normalized
  L2 at most `5e-4` and normalized Linf at most `2e-3`;
- CPU/CUDA eager RHS agreement on the 37-point entropy wave must have maximum
  normalized difference at most `5e-4`;
- fixed-shape CUDA and CPU `torch.compile(fullgraph=True)` must capture one
  graph with zero breaks; and
- compiled/eager maximum normalized difference must be at most `5e-5`, with
  RMS normalized difference at most `1e-5`.

Static inspection must confirm no hidden host/device transfer, NumPy
conversion, scalar extraction, custom operator, handwritten CUDA, or
handwritten Triton in the Euler numerical loop. Explicit dtype-only conversion
inside the already-audited WENO precision helper is permitted.

## Evidence and stop boundary

The committed runner must precede execution and refuse a dirty source tree or
existing output directory. The record contains policy identities, source and
dependency hashes, per-case raw metrics, stage minima, step counts, environment
identity, compiler behavior, failures, and SHA-256 checksums. An independent
verifier recomputes every decision from raw metrics.

Only policies passing every Tier-2 correctness gate become eligible for a
separately frozen full-Euler performance campaign. Tier 2 does not measure
performance, choose a production default, explore Euler-specific demotions,
modify DVEB, or make a publication claim.
