# G3 R6 Comprehensive Qualification Protocol

Status: frozen before R6Q compilation or qualification results.

Date: 2026-08-29 (UTC)

## Question

Does the recovered R6 face-once CUDA schedule preserve the qualified Shu
characteristic FD-WENO-5 forward calculation beyond the single periodic-vortex
specimen that guided the R1--R6 ladder?

This closes G3 only as a forward numerical qualification. It does not perform
the randomized performance comparison reserved for G4 and does not admit the
candidate into `src/gradflow/`.

## R6Q interface-only extension

The qualification executable may add only:

- raw component-major unique-cell FP32 input;
- raw output;
- an RHS-only mode using the same alpha and face kernels;
- support for the already-qualified six-interval minimum; and
- metadata identifying mode and contract.

The R6 spatial and SSP-RK3 arithmetic must not change. On the exact frozen
`N=32` R6 input, R6Q must reproduce the frozen one-step R6 output bit-for-bit.
Failure of this identity gate invalidates all later R6Q evidence.

## Oracles and bounds

- Primary parity oracle: qualified GradFlow characteristic FD JS-WENO-5 in
  float32, using the exact same input values.
- Diagnostic oracle: the same GradFlow formulation in float64 after exact
  promotion of the frozen float32 input.
- Full-step parity: `rtol=0`, `atol=2e-5`, inherited unchanged from E4.
- RHS parity: maximum absolute error at most `5e-5` and RMS error divided by
  oracle RMS at most `2e-5`.

No tolerance may change after results exist.

## Frozen matrix

### Full-step parity

For both the periodic vortex and the existing physically admissible periodic
non-vortex perturbation:

```text
(N, steps) = (6,1), (6,10), (32,1)
```

### Smooth spatial convergence

Use the characteristic entropy wave with constant pressure and velocity on
`N={12,18,27,40}` cubic unique grids. Compare the R6Q RHS with both the
qualified float32 RHS and the analytic Euler RHS. All analytic RMS errors must
decrease, and at least one observed rate must be at least 3.0, matching the
existing conservative arbitrary-order qualification criterion for WENO-5.

### Critical-point characterization

Use `rho = 1 + 0.1 sin(2 pi x / 10)^3` with the same constant pressure and
velocity on `N={12,18,27,40}`. Record global RMS error and the point error at
the aligned critical point. Candidate/oracle RHS parity must pass. The critical
rate is characterized, not required to be fifth order; no post-result claim
may erase known Jiang--Shu critical-point degradation.

### Periodic discontinuity stress

At `N=32`, run one and ten steps for:

1. a periodic dual-interface Sod state; and
2. a periodic dual-interface Shu--Osher-type shock/entropy-wave state.

These are periodic stress specimens, not substitutes for the canonical
outflow-boundary Sod and Shu--Osher benchmarks. Full-state FP32 parity,
finiteness, positive density, and positive pressure are required.

### Conservation

For every full-step case, componentwise drift must be recorded. The absolute
drift bound is

```text
128 * steps * eps_float32 * sum(abs(initial_component))
```

for each conserved component. This is a roundoff budget, not permission for a
systematic conservation defect.

### Directional sensitivity

On the `N=6` non-vortex state, use a fixed smooth conservative perturbation and
central finite differences with `h=1e-3`. Compare the R6Q and float32-oracle
one-step directional responses. Relative RMS discrepancy must not exceed 2%.

This diagnoses forward sensitivity only. R6Q exposes no reverse-mode or
autograd ABI; successful finite differences do not qualify it as a
differentiable GradFlow backend.

## Evidence and decision

Freeze the protocol, source, build recipe, executable, compiler log, full JSON
record, compressed comparison arrays, environment record, and SHA-256
manifest. Report every metric and failed case.

G3 forward qualification passes only if identity, step parity, RHS parity,
smooth convergence, discontinuity admissibility/parity, conservation, and
directional sensitivity all pass. The differentiability limitation remains an
explicit backend-admission blocker rather than being converted into a failed
numerical gate.
