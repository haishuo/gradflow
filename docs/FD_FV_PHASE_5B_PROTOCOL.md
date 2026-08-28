# FD/FV nonlinear Phase-5B implementation and qualification protocol

Status: frozen before production Burgers implementation or qualification.

Freeze date: 2026-08-28 UTC.

## Purpose and claim boundary

Phase 5B implements exactly the smooth pre-shock Burgers seed frozen in
FD/FV nonlinear Phase 5A and determines whether both matched WENO-JS5 paths are
correct, conservative, differentiable, device-resident, and compilable. It
collects no performance measurements.

Correctness > performance > convenience remains binding. Thresholds in this
document are frozen before observing the production implementation. A failed
gate remains a failed result; changing mathematics or acceptance bands requires
a prospective named resolution protocol.

Phase 5B does not add nonlinear shocks, multidimensional Burgers, Euler, a
best-practical formulation, dynamic wave-speed estimation, mixed precision,
arbitrary-order FV, automatic selection, DVEB work, optimization, or a general
PDE framework.

## Required predecessor

The committed Phase-5A verifier must pass byte-for-byte before qualification.
The qualification record retains SHA-256 identities for its contract and
oracle-case records, the Phase-5A protocol, and the independent oracle source.

The governing continuous problem is

```text
u_t + (u^2/2)_x = 0,
u(x,0) = 0.5 + 0.2*sin(2*pi*(x-0.07)),
x in [0,1), periodic, T=0.1.
```

The exact first shock time is approximately `0.7957747154594768`; all Phase-5B
solution and derivative claims are strictly pre-shock.

## Production implementation surface

The only new canonical module is `src/gradflow/burgers.py`, with this narrow
scientific surface:

```python
flux = gradflow.burgers_flux(state)
fd_rhs = gradflow.burgers_fd_weno5_rhs(points, dx, alpha, axis=-1)
fv_rhs = gradflow.burgers_fv_weno5_rhs(averages, dx, alpha, axis=-1)
```

The FD function delegates reconstruction to the qualified generated
`WENOJS(5)` classical split-physical-flux path. The FV function delegates to
the qualified physical-cell-average reconstruction and Rusanov path. Neither
may duplicate WENO algebra, convert point values to averages, convert averages
to points, estimate `alpha`, move devices, extract scalars, or select a backend.

Inputs must already be float32 or float64 tensors on the desired device. Output
shape, dtype, and device match the state. `dx` and `alpha` retain the existing
validated scalar contracts. Phase 5B passes the frozen `alpha=0.7` explicitly
at every SSP-RK3 stage. The public functions are reusable for periodic scalar
Burgers states; the phase-specific initial condition and time step remain in
the experiment layer.

The frozen formulation identities are:

- `fd_classical_js5_burgers_global_lf_periodic_v1`;
- `fv_dimensional_js5_burgers_global_lf_periodic_v1`.

## Oracle and projection gate

For both `N=8` and `N=17`, recreate the Phase-5A initial and terminal point and
cell-average projections and require byte-equivalent hexadecimal binary64
values to the frozen record. The independent standard-library oracle must
remain free of PyTorch, NumPy, GradFlow, and WENO imports.

At `N=37`, evaluate the exact initial spatial derivative independently:

- FD: `-u(x_i)*u_x(x_i)`;
- FV: `-(f(u(x_{i+1}))-f(u(x_i)))/dx`, the exact derivative of each physical
  cell average.

Production outputs must be finite and retain their declared point/average
semantics. This gate does not require FD and FV arrays to agree with each other.

## Constant and conservation gates

For a constant binary64 state `u=0.4` at `N=37`, each RHS must have maximum
absolute value no larger than `5e-13`. One SSP-RK3 step with `dt=1e-3` must
remain constant within the same tolerance.

For every complete solve, mass change must satisfy

```text
abs(dx*sum(final-initial))
<= 64*eps*dx*sum(abs(initial)) + 2e-15.
```

The same bound applies to a deterministic nonconstant RHS telescoping probe.

## Spatial convergence and critical-point characterization

At `t=0`, use `N=(24,36,54,81)`, binary64, `dx=1/N`, and `alpha=0.7`. Compare
each semidiscrete RHS to its method-appropriate exact derivative.

Record L1 and L2 errors on the whole grid. Also record a noncritical L1 norm
whose associated coordinate is `x_i` for FD and the cell center for FV, and
whose periodic distance from each analytic critical point

```text
x = 0.07 + 0.25, 0.07 + 0.75  (mod 1)
```

is at least `0.1`. This fixed physical mask does not move toward a critical
point under refinement.

Acceptance requires:

- whole-grid L1 and L2 errors decrease at every refinement;
- noncritical L1 errors decrease at every refinement;
- the final consecutive whole-grid L1 rate is at least `3.0`; and
- the final consecutive noncritical L1 rate is at least `4.3`.

The method remains nominally WENO-JS5. Whole-grid rates are not mislabeled as
uniform fifth order because smooth periodic nonconstant data necessarily have
critical points and prior GradFlow evidence recorded material JS order loss.
All rates and critical/noncritical errors remain in the result even when the
gate passes.

## Complete-solve convergence

For the same `N=(24,36,54,81)`, initialize each method from the independent
projection and use

```text
nominal_dt = 0.2*dx^(5/3)/0.7
steps      = ceil(0.1/nominal_dt)
dt         = 0.1/steps.
```

Advance with SSP-RK3 and fixed `alpha=0.7`. Compare at exactly `T=0.1` with
the method-appropriate characteristic oracle. For each method:

- binary64 L1 and L2 errors are finite and decrease at every refinement;
- the final consecutive L1 rate is at least `3.0`;
- the largest-grid L1 and L2 errors are each no larger than `2e-5`; and
- every mass check passes the frozen conservation bound.

No solve duration is read or stored.

## Differentiation gate

At `N=19`, define a smooth deterministic binary64 state and perturbation. For
each formulation, compare the JVP of three fixed SSP-RK3 steps (`dt=1e-3`,
fixed `alpha=0.7`) with a centered finite difference using perturbation step
`1e-6`.

Both vectors must be finite. Their maximum absolute difference must be no
larger than `3e-6`, and their relative L2 difference (with denominator clamped
only against machine underflow) must be no larger than `3e-5`. This qualifies
a smooth fixed-alpha discrete map; it does not claim differentiability through
a max-speed branch or a shock.

## CPU, CUDA, compiler, and residency gates

At deterministic `N=37` binary64 state, both RHS functions and one SSP-RK3 step
must execute eagerly on CPU. Compile each fixed-shape callable with
`torch.compile(fullgraph=True, dynamic=False)`. `torch._dynamo.explain` must
report one graph and zero graph breaks. Compiled/eager maximum absolute
difference must not exceed `2e-11`; outputs must be finite and retain CPU
shape, dtype, and device.

Forge's RTX 5070 Ti is confirmed host inventory. Phase 5B therefore requires a
fresh device-visible CUDA run rather than treating sandbox isolation as absent
hardware. For each RHS and step:

- CPU/CUDA eager maximum absolute difference is at most `2e-11`;
- CUDA compiled/eager difference is at most `2e-11`;
- one graph and zero breaks are required; and
- every output remains finite binary64 on the input CUDA device.

Compilation must occur, but compilation duration is neither measured nor
stored. MPS is `host_confirmed_absent` on Forge and is not simulated.

## No-transfer gate

Static inspection of `src/gradflow/burgers.py` rejects `.cpu()`, `.cuda()`,
`.to()`, `.item()`, `.numpy()`, and imports of NumPy, custom operators, C++,
CUDA, or Triton.

Profile one eager binary64 RHS per formulation on CPU and CUDA. Reject
`aten::_to_copy`, `aten::copy_`, H2D, D2H, memcpy, or any profiler event with a
copy/movement name. An `aten::to` dispatch is admissible only when its CPU and
device memory usage fields are all zero. Profiler synchronization is outside
the numerical call. This is event evidence, not timing.

## Infrastructure statuses

The record follows `docs/EXECUTION_INFRASTRUCTURE_ADMISSION.md`. Host inventory,
process visibility, and numerical admission are separate fields. Overall
Phase-5B acceptance on Forge requires CUDA status `admitted`; a hidden process
must be rerun in an explicitly authorized device-visible context. A visible
failure is `visible_admission_failed`, never “unavailable.”

## Immutable record and verifier

The qualification runner writes one deterministic JSON record containing:

- clean source commit, protocol commit, source and predecessor hashes;
- environment and explicit host/process device observations;
- projection/oracle parity;
- constant, conservation, spatial, complete-solve, differentiation, compiler,
  CPU/CUDA, and movement evidence;
- every tolerance and gate decision;
- `performance_measurements_collected=false`; and
- explicit exclusions and failure names.

The canonical output is
`experiments/fd_fv_nonlinear/results/phase_5b_20260828/qualification.json` with
a SHA-256 manifest. The writer refuses overwrite. An independent verifier
recomputes all decisions and checks source, predecessor, record, and manifest
identity without rerunning the solves.

## Stop condition

Stop after the Phase-5A predecessor verifies, the implementation and immutable
Phase-5B record pass, the full configured CPU/CUDA test suite passes, coherent
local commits exist, and the working tree is clean. Do not collect timing,
write a Phase-5C performance protocol after seeing runtime, add a nonlinear
shock, change WENO order, optimize representations, modify DVEB, claim an FD/FV
winner, claim publication readiness, or push without explicit authorization.
