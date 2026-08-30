# Academic U4-B OpenSBLI adapter and correctness protocol

Status: **frozen before adapter implementation or execution**.

Outcome: the frozen protocol passed; see `ACADEMIC_U4B_RESULTS.md`.

Date: 2026-08-30 (UTC)

## Purpose

U4-B asks whether the independently maintained OpenSBLI framework can express
and execute the frozen GradFlow scalar FD-WENO-JS5 semidiscrete operator using
OpenSBLI's own numerical and generated OPS machinery.

This phase is correctness-only. Runtime, compilation duration, generated-code
size, and installation effort are not performance results. No external lane
may enter a timing table during U4-B.

## Frozen upstream

- repository: `https://github.com/opensbli/opensbli`;
- commit: `e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d`;
- tree: `0ff053443f6b243b2bd42475f98122306151427d`; and
- execution backend for qualification: sequential CPU OPS generated from the
  same OpenSBLI description intended for later heterogeneous translation.

CPU execution is sufficient to qualify the mathematical adapter. A future
CUDA lane must separately pass CPU/CUDA agreement before timing.

## Frozen mathematical subject

- one-dimensional positive scalar advection `u_t + u_x = 0`;
- unique periodic nodes `x_j=j/N`, `j=0,...,N-1`;
- finite-difference WENO-JS5 reconstruction;
- global LF speed `alpha=1`;
- nonlinear power two;
- GradFlow smoothness scale 12 and epsilon `1e-29`, represented in OpenSBLI's
  standard unscaled smoothness convention by epsilon `1e-29/12`;
- one semidiscrete RHS `du/dt`, with no RK update; and
- IEEE float64.

For this constant-speed problem, a native local-LF evaluation that returns
exactly one at every face is mathematically identical to the frozen global
`alpha=1`. The implementation must disclose which path it uses.

## Adapter boundary

The adapter may contain a one-equation physics/eigensystem description,
problem initialization, periodic boundary declaration, result export, and
minor generalization patches needed to let existing OpenSBLI classes accept a
scalar system.

The retained patch may only:

1. expose WENO-JS epsilon as a constructor value without changing its weight
   formula;
2. make the existing custom-physics hook retain the supplied eigensystem;
3. avoid constructing Euler-specific temporary inverses or global reductions
   when the selected scalar/LLF expression does not contain them;
4. define the scalar 1x1 identity characteristic system with eigenvalue one;
5. expose the generated semidiscrete residual before RK integration; and
6. add deterministic state input/output and qualification instrumentation.

It may not provide hard-coded WENO coefficients, reimplement smoothness
indicators, nonlinear weights, reconstruction, LF splitting, flux divergence,
periodic halo exchange, or the generated execution loop. Those must remain
traceable to OpenSBLI source and OPS output.

If the permitted changes are insufficient, U4-B stops with
`building_block_only`. We do not expand the adapter until it becomes a new
solver.

## Frozen cases and gates

All comparisons are `rtol=0` against GradFlow's canonical float64 order-5
RHS.

### Q1. Nonconstant pointwise cases

At `N=64`, execute:

```text
u_a(x) = 0.4 + sin(2*pi*x) + 0.1*cos(6*pi*x)
u_b(x) = sin(6*pi*x) + 0.15*cos(8*pi*x)
```

Each maximum absolute RHS difference must be at most `2e-12`. Both external
and canonical arrays must be finite and have identical ordering and shape.

### Q2. Constant-state preservation

At `N=64`, `u(x)=0.37` must produce maximum absolute RHS at most `2e-12`.

### Q3. Conservation

For each nonconstant case, require

```text
abs(sum(rhs)) <= 32*eps(float64)*sum(abs(rhs)).
```

The bound is evaluated independently for OpenSBLI and GradFlow.

### Q4. Smooth convergence

For `u(x)=sin(2*pi*x)` at `N={40,80,160,320}`, compare each external RHS
against the analytic `-2*pi*cos(2*pi*x)`. Every array must be finite and
conservative, and every successive L2 rate must exceed `4.8`.

The external and GradFlow L2 errors are both retained. Convergence does not
replace the Q1 pointwise gate.

## Evidence requirements

Retain:

- the exact upstream revision and inspected file hashes;
- the complete adapter patch and its SHA-256;
- the standalone adapter/application source authored for the experiment;
- generation, translation, build, and execution commands;
- generated-source hashes sufficient to identify what ran;
- compiler, Python, SymPy, OPS, and host identity;
- all input and output arrays or losslessly reproducible generators;
- a machine-readable qualification record; and
- an offline checksum/semantic verifier.

External GPL source is not vendored into GradFlow. The local adapter and patch
are research evidence; their presence does not resolve GradFlow's project
license or redistribution questions.

## Decisions

- All gates pass with only permitted adaptation:
  `matched_operator_adapted_qualified`; U4-C timing protocol may be designed.
- Native execution works but a numerical gate fails:
  `correctness_excluded`; retain the failure and do not time it.
- Expressing the operator requires prohibited replacement:
  `building_block_only`; no direct external-system timing.
- Toolchain or environment failure after the adapter is shown admissible:
  `qualification_blocked`; report the exact external dependency rather than a
  numerical conclusion.

## Stop condition

U4-B closes when exactly one decision above is supported by hashed evidence,
the verifier and relevant tests pass, coherent local commits exist, and the
worktree is clean.

Do not push without explicit authorization.
