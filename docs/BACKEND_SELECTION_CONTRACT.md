# Backend selection contract

This contract is governed by `ENGINEERING_CHARTER.md`. Placement is a
performance decision below the correctness gate: no backend win can authorize
different mathematics, invalid output, or an unqualified approximation.
DVEB participation additionally follows the independent-project rule in
`DVEB_RELATIONSHIP.md`.

## User-facing principle

GradFlow's scientific API describes the problem, not the implementation
schedule. The intended surface remains:

```python
solver = gradflow.Solver(
    equations="navier-stokes",
    dimension=3,
    weno=("JS", 11),
    flux_split="global_lf",
    boundaries=...,
    dtype=torch.float32,
)
result = solver.run(initial_state, final_time=...)
```

The compiler/runtime may select CPU SIMD, CUDA, ordinary eager PyTorch,
compiled/AOT PyTorch, or another qualified backend without exposing that
machinery. An explicit backend override remains legal for experiments and
expert use even when it is slower.

This document is a contract for future generalization, not a claim that the
general surface above exists today. A narrow 3-D Euler JS-WENO-5 `Solver`
slice now exists; it rejects the shown Navier--Stokes/JS-11 request. See
`SOLVER_VERTICAL_SLICE.md`.

## Semantics precede placement

A backend is eligible only when it implements the complete problem signature:

- equations and equation parameters;
- spatial dimension;
- finite-difference WENO family and order;
- componentwise or characteristic reconstruction;
- flux splitting and alpha policy;
- boundary treatment and endpoint convention;
- dtype and numerical tolerances;
- grid shape and layout;
- time integrator and requested work amount;
- input and required output residency; and
- differentiability requirements.

"Approximately similar WENO" is not eligible. Placement never changes the
mathematics to obtain a timing win.

## Automatic selection

Automatic selection consumes measured, versioned evidence rather than a
device heuristic. A qualification record binds:

```text
(problem signature, hardware signature, software versions,
 endpoint, calibrated domain) -> eligible targets and cost model
```

The calibrated domain includes at least grid shape, timestep/work stratum,
dtype, input/output residency, and memory feasibility. A selector may choose
only inside a tested or explicitly interpolable envelope. It must not silently
extrapolate.

When no qualification covers a request, GradFlow must do one of the following
in order:

1. use a semantically matched, correctness-qualified fallback;
2. perform separately reported calibration if the user/environment permits;
3. honor an explicit user target; or
4. fail with a precise unsupported-placement message.

Falling back is preferable to inventing confidence. Selection provenance and
the chosen backend must be inspectable through diagnostics, but ordinary users
need not see it during a successful run.

## Explicit targets

An eventual expert override may name targets such as:

```python
solver.run(state, final_time=..., backend="cpu_simd", backend_options={"threads": 4})
solver.run(state, final_time=..., backend="cuda")
solver.run(state, final_time=..., backend="pytorch-aot")
```

Overrides bypass the performance selector, not semantic validation, memory
safety, or correctness checks. GradFlow should warn when a target lies outside
its performance qualification, while still permitting a legal request.

## Timing and preparation

GradFlow distinguishes:

- deployment preparation: generated C++/CUDA compilation, AOTInductor
  packaging, and machine calibration;
- fresh invocation: from process/application request through required output
  materialization; and
- resident execution: after implementation and state placement through the
  terminal device synchronization.

Ahead-of-time preparation is excluded from prepared invocation latency but is
never described as free. Any compilation triggered by an ordinary measured
run is charged to that run. Backend decisions must use the endpoint matching
the user's workload; resident kernel timing alone cannot decide a
CPU-originating, host-output request.

## Differentiability

The canonical ordinary-PyTorch formulation is the differentiable scientific
source of truth. A native generated backend is not automatically
differentiable merely because its forward result agrees. Until a backend has
a separately verified autograd contract, requests requiring gradients must
remain on a differentiable PyTorch path. Native DVEB deployment is currently
qualified only for forward execution.

## Current evidence and allowed DVEB role

The final DVEB WENO requalification at GradFlow branch
`codex/dveb-final-requalification` establishes:

- full-state float32 agreement within `7.153e-7` for the matched 3-D Shu Euler
  JS-WENO-5 workload;
- generated CUDA within 1.65% of an independent matched native ceiling over
  the declared N=96/128, one/ten-step fresh-process points; and
- a WENO-specific, machine-specific held-out selector pass for N in
  `{8,16,32,48,64}` and steps in `{1,10}`.

Therefore DVEB's generated implementation is a qualified optional native
forward backend for this exact formulation. Portable ABI v1 now exposes the
internal native functions to `Solver.run(initial_state, ...)` through
caller-owned CPU float32 buffers. It is versioned, hash-qualified, and has
passed arbitrary-state CPU/CUDA/PyTorch parity. It does not provide autograd or
device-pointer input.

Portable device ABI v2 is now separately qualified as an explicit resident
CUDA endpoint for this exact fixed program. It accepts caller-owned CUDA
buffers and a caller stream through a reusable context. The E4 addendum found
it materially faster than the tested PyTorch resident lanes, but it has not
been calibrated for `auto`; this evidence must not be turned into an implicit
selection rule yet.

Automatic DVEB placement is enabled only when an installation explicitly
supplies both the ABI artifact and a verified model. The tested model refuses
outside its bounded N=7--72 range; GradFlow catches that placement-only refusal
and uses the correctness-qualified PyTorch fallback. The existing model was
measured through the fresh-process executable endpoint. It is sufficient to
gate bounded selector mechanics, but it is not an ABI-specific in-process
latency calibration; a future performance decision must measure that endpoint
directly.

DVEB's generic automatic selector remains **NO-GO** at DVEB commit `2f1f3ab`.
That result and the bounded WENO pass answer different questions and must both
remain visible.

## Implemented vertical slice

The first code slice is deliberately narrow:

1. expose the existing direct-PyTorch 3-D Euler JS-WENO-5 formulation through
   an internal typed problem description;
2. validate periodic duplicated-endpoint shape, spacing, dtype, and positivity;
3. provide eager PyTorch as the correctness fallback;
4. allow the hash/version-qualified DVEB ABI only for its exact forward
   eligibility envelope;
5. expose selection diagnostics and explicit target override; and
6. reproduce the committed parity tests through the new surface.

The slice does not pretend to implement arbitrary equations, JS-11, JS-15,
general boundaries, or characteristic policy generation. Its fixed-step
direct PyTorch path has a bounded autograd gate; broader backward claims still
require independent mathematical tests.
