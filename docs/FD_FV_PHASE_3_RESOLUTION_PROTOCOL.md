# FD/FV Phase-3R resolution protocol

Status: frozen before the Phase-3R diagnostic implementation and run.

Freeze date: 2026-08-27 UTC.

## Purpose and relationship to Phase 3

Phase 3R resolves two questions exposed by the immutable failed Phase-3 run:

1. did the `4.692164` negative-advection rate indicate a failure of the
   fifth-order reconstruction, or did the frozen mixed Fourier experiment
   combine ordinary smooth accuracy with classical WENO-JS critical-point
   behavior; and
2. did the recorded `aten::to` profiler label represent data movement, or was
   it a no-copy operator dispatch?

This is a new prospective result series. It does not overwrite, amend, or
reclassify the original Phase-3 record, whose two gates remain failed. It does
not change the canonical implementation or mathematical formulation. Phase 3R
collects correctness and execution-semantics evidence only; no latency,
throughput, memory-performance, FD/FV comparison, or compiler optimization is
permitted.

Correctness > performance > convenience remains binding.

## Frozen source identity

The candidate is the unchanged
`fv_dimensional_js5_global_lf_periodic_v1` source introduced at commit
`1d920ea97ed7abec9e4e451b377343cf72316f4c`. Phase 3R must record and verify
the SHA-256 of `src/gradflow/fv_weno5.py`. The original Phase-2 and Phase-3
records and their manifests must verify before Phase 3R can pass.

No numerical source edit is admissible inside this resolution study. If an
implementation repair becomes necessary, Phase 3R fails and a separately
frozen implementation series is required.

## Noncritical design-order experiment

The design-order gate uses analytic physical cell averages of

```text
u(x) = exp(x),  x in [0,1].
```

Its derivative is nonzero everywhere. The tensor is still processed by the
canonical periodic implementation, but errors are evaluated only on faces or
cells whose complete WENO-5 stencil lies inside `[0,1]`; periodic wrap values
are excluded. This is a local reconstruction/semidiscrete consistency test,
not a new boundary condition.

For `N=(32,48,72,108)` in float64:

- compare left face reconstruction on face indices `2 .. N-3` with
  `exp((i+1)dx)`;
- compare right face reconstruction on face indices `1 .. N-4` with the same
  exact values;
- compare the positive-speed RHS on cell indices `3 .. N-3` and the
  negative-speed RHS on indices `2 .. N-4` with
  `-c*(exp((i+1)dx)-exp(i*dx))/dx`, for `c in {1,-1}`.

Each of the four L2 error sequences must decrease at every refinement. The
last two consecutive rates of every sequence must each be at least `4.7`.
This gate retains a fifth-order threshold; it does not lower the observed
Phase-3 threshold to admit the candidate.

## Critical-point characterization

Two records are mandatory and are not pooled with the noncritical gate:

1. reproduce the original mixed-Fourier positive/negative semidiscrete errors
   for `N=(32,48,72,108)` and verify them against the immutable Phase-3 record;
2. reconstruct the aligned maximum at `x=1/4` of
   `u(x)=sin(2*pi*x)` from exact physical cell averages, using
   `N=(32,64,128,256)` so the critical point is a face for every grid.

For the aligned case, left- and right-biased absolute errors must be finite and
decrease at every refinement. All errors and consecutive rates are recorded.
No fifth-order lower bound is imposed on this diagnostic: classical WENO-JS is
known to have critical-point sensitivity, and the purpose is to disclose the
observed behavior rather than redefine it as ordinary smooth-region order.

Phase 3R fails if the original mixed-Fourier evidence cannot be reproduced, or
if the aligned critical-point errors are nonfinite or nondecreasing.

## Data-movement evidence

The original `aten::to` event-name rule remains part of the failed Phase-3
record. Phase 3R uses direct movement evidence prospectively.

Static AST inspection covers both `src/gradflow/fv_weno5.py` and its numerical
dependency `src/gradflow/weno_js.py`:

- `.cpu()`, `.cuda()`, `.item()`, and `.numpy()` are forbidden in the
  numerical sources;
- a `.to()` call that names or supplies a device is forbidden;
- dtype-only `.to(dtype=...)` sites are recorded because the shared generated
  implementation supports explicit mixed-precision policies, but they do not
  establish movement for the native float64 configuration.

The fixed float64 RHS is profiled after input creation. The gate fails on any
`aten::_to_copy`, `aten::copy_`, or event whose lowercase name contains
`memcpy`, `host to device`, `device to host`, `h2d`, or `d2h`. The count and
memory fields of every `aten::to` event are recorded even though `aten::to`
alone is not treated as proof of a copy. Input/output dtype and device must be
identical.

When CUDA is visible, the input is allocated on CUDA before profiling, device
synchronization brackets the probe, CPU and CUDA profiler activities are
enabled, and the same rules apply. CUDA unavailability is recorded as
`untested_unavailable`, never simulated. MPS availability is recorded but MPS
profiling is not claimed in this Linux protocol.

## Record and acceptance

The immutable JSON record must contain source/protocol/original-record hashes,
environment identity, all errors and rates, AST findings, profiler keys and
movement-event decisions, device/dtype residency, explicit gate decisions,
and `performance_measurements_collected=false`. A separate verifier checks the
record and `SHA256SUMS`.

Phase 3R passes only when:

- the original Phase-2 and Phase-3 immutable records verify;
- the candidate numerical source hash is unchanged;
- all four noncritical fifth-order sequences pass;
- both critical-point records satisfy their characterization rules;
- CPU movement evidence passes and CUDA evidence passes when CUDA is visible;
- the existing test suite passes; and
- the source tree used for the run is clean.

A passing Phase 3R result qualifies this exact scalar FV seed under the
combined Phase-2, original Phase-3, and prospective Phase-3R evidence. It does
not turn the original failed gates into passes and does not qualify any other
FV formulation, boundary, PDE, precision, or device.

## Stop condition

Stop after the record, verifier, interpretation, tests, coherent local commits,
and clean working tree. Do not begin Phase 4 timing, alter the canonical
numerical source, optimize representations, extend to Euler or arbitrary
order, change DVEB, or push Phase-3R commits without new explicit
authorization.
