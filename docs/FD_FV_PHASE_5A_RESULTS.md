# FD/FV nonlinear Phase-5A result

Status: **passed; nonlinear mathematics and independent oracle frozen without
production implementation or timing**.

Freeze date: 2026-08-28 UTC.

The immutable records are under
`experiments/fd_fv_nonlinear/results/phase_5a_20260828/`. Their manifest covers
`contract.json` and `oracle_cases.json`; the independent verifier regenerates
all three files byte-for-byte.

## Frozen nonlinear seed

The next matched FD/FV boundary is one-dimensional inviscid Burgers flow on
`[0,1)` with unique periodic storage:

```text
u_t + (u^2/2)_x = 0,
u(x,0) = 0.5 + 0.2*sin(2*pi*(x-0.07)),
T = 0.1.
```

The analytic first shock time is `0.7957747154594768`; the final time is only
`0.12566370614359174` of that value. The minimum characteristic-map Jacobian
through the experiment is `0.8743362938564083`, so this phase tests a unique
smooth nonlinear solution and makes no shock claim.

The matched pair retains different, mathematically correct discrete states:

- FD stores nodal values and reconstructs globally LF-split physical flux;
- FV stores physical cell averages, reconstructs face states, and then applies
  the two-state Rusanov flux.

Both use WENO-JS5, the existing scale-12/epsilon-`1e-29` weight policy,
SSP-RK3, and the same frozen physical LF bound `alpha=0.7`. Because flux
splitting and state reconstruction no longer commute, this is the first
genuinely nonlinear matched-component comparison rather than another linear
translation test.

## Independent oracle result

The standard-library oracle imports no PyTorch, NumPy, GradFlow, or WENO code.
It obtains FD point values by deterministic inversion of the characteristic
map. It obtains FV cell averages from an exact conservation-law primitive in
characteristic coordinates.

Across frozen `N=8` and `N=17` initial and terminal cases:

- maximum characteristic residual: `2.220446049250313e-16`;
- maximum primitive-versus-4,096-panel Simpson difference:
  `1.27675647831893e-15`;
- maximum 2,048-to-4,096-panel Simpson change:
  `2.220446049250313e-16`;
- maximum FV periodic-mean error: `1.1102230246251565e-16`; and
- maximum FV-average versus center-sample difference:
  `0.00544959428972458`.

The last value is deliberately nonzero evidence that the FV state cannot be
initialized or judged by silently substituting center samples.

## Infrastructure correction

`docs/EXECUTION_INFRASTRUCTURE_ADMISSION.md` now separates physical host
inventory from process-local visibility. New records may distinguish an
admitted device, a visible but failed admission, a device hidden from the
current process despite confirmed host presence, confirmed host absence, and
unknown/probe-failure states.

Accordingly, a sandbox-local negative CUDA probe on Forge is now classified
as `process_hidden_host_present`, not “Forge has no GPU.” Historical records
remain immutable; their linked CUDA supplements retain the definitive device
evidence.

## What Phase 5A establishes—and does not

Phase 5A establishes an auditable nonlinear continuous problem, exact
method-appropriate projections, execution-infrastructure vocabulary, and the
prospective Phase-5B correctness gate. It does not establish that the
production FD or FV implementation solves Burgers correctly, has fifth-order
whole-domain behavior at JS critical points, compiles, differentiates, avoids
transfers, or performs well.

No production Burgers solver, timing sample, optimization, multidimensional
extension, Euler extension, order sweep, or publication claim was added.
Phase 5B may now implement only this frozen seed and must pass its correctness
gate before a Phase-5C performance protocol is written.
