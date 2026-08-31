# Academic U4-F batched-line regime-map protocol

Status: **prospective protocol frozen before U4-F implementation, qualification,
or timing**.

Date: 2026-08-31 (UTC)

## Question

U4-E rejected PyTorch/TorchInductor as the fastest admitted backend for one
scalar one-dimensional binary64 WENO-JS5 line at `N=8192`. U4-F asks the next
bounded question:

> Does ordinary compiled PyTorch recover resident forward-execution
> competitiveness against automatically scheduled DVEB when the same
> independently reconstructed WENO line is repeated as a contiguous batch?

This is a prospective test of one proposed amortization regime, not a rescue
claim. The batch axis represents independent lines, ensembles, or pencils. It
is not called a two- or three-dimensional PDE and does not add transverse
fluxes.

## Fixed implementations and provenance

- GradFlow source and this protocol commit;
- the exact DVEB Trunk 005 ABI-v1 library already qualified in U4-E, SHA-256
  `9ff9172b1ac712b8bc97ca9523fd114b2637e5d7825259371ba9850459168443`;
- the U4-E handoff bundle, SHA-256
  `2342f66416b1b120efd42e0e4ca8838f32cef4c62a13bf43042fb12ef7354ae0`;
- GradFlow's direct `weno5_rhs` compiled with
  `torch.compile(fullgraph=True, dynamic=False)`; and
- the immutable U4-E OpenSBLI/OPS result at batch one as an external anchor,
  not as a participant in the new batched timing decisions.

DVEB and OpenSBLI remain read-only. The new DVEB adapter may allocate and pad
batched caller-owned arrays, transfer them, call the public ABI, time the
declared endpoint, and export qualification data. It may contain no WENO
mathematics and may not force a schedule. New machine-readable lane names are
`dveb_native` and `pytorch_inductor`; the historical ambiguous key `gradflow`
is prohibited.

## Frozen mathematics and state family

Every row solves the U4-E scalar contract:

- `u_t + u_x = 0` on `N=8192` unique periodic nodes;
- finite-difference WENO-JS5 in the Gottlieb-equivalent correction algebra;
- global Lax--Friedrichs `alpha=1`;
- 12-scaled smoothness indicators and epsilon `1e-29` inside squared weights;
- one semidiscrete RHS, no time update; and
- IEEE binary64.

For batch row `b`, with `x_j=j/N`, the deterministic input is

```text
0.4
+ sin(2*pi*(37*x_j + (b mod 127)/127))
+ 0.1*cos(2*pi*(91*x_j + 3*(b mod 127)/127)).
```

Batch one therefore reproduces the exact U4-E input bytes. The batch axis is
contiguous outside the spatial axis, shape `(B, 8192)`. Each row has its own
periodic halo and no row may read another row.

The prospectively frozen batch counts are:

```text
B = 1, 4, 16, 64, 256, 1024
```

This spans 8,192 through 8,388,608 independent spatial evaluations while
holding `dx`, per-line floating-point amplification, order, and mathematics
fixed. No batch point may be removed because its result is inconvenient.

## F1: implementation and correctness admission

Before comparative timing, every `(B, backend, device)` lane must:

1. return a full finite array of shape `(B,8192)`;
2. match an eager CPU execution of the frozen direct formulation under
   `maximum_normalized <= 5e-11` and `RMS_normalized <= 5e-12`;
3. satisfy conservation separately on every row under
   `32 * epsilon_machine * sum(abs(rhs_row))`;
4. agree between CPU and CUDA under the same normalized bounds;
5. record exact input and output SHA-256 digests;
6. report one TorchInductor graph and zero graph breaks for PyTorch; and
7. record DVEB's automatic CPU loop, CUDA block, reuse policy, launch count,
   scratch bytes, elements written, and synchronization field.

The DVEB query is authoritative. The adapter sets `n=8192`, `nb=B`, ghost
width three, and one CPU thread, but does not set a forced loop, block size, or
reuse policy. Device execution must use a caller-owned nondefault CUDA stream
and report no internal synchronization. A failed cell remains in the record
and its timing is prohibited.

The eager CPU array is a local execution authority, not a newly independent
mathematical oracle. Independence comes from the already qualified Gottlieb,
OpenSBLI, and U4-E lineage at batch one; U4-F tests preservation under the
leading batch dimension.

## F2: resident forward regime map

Only cells admitted by F1 are timed. For every batch and device:

```text
lanes per block                    = 2
independent workers per lane       = 6
warmups per worker                 = 5
retained observations per worker  = 20
lane-order seed                    = 20260831 + B + device offset
bootstrap resamples               = 20,000
material effect threshold         = 5 percent
thermal stop                      = 80 C
CPU threads                       = 1
```

The parent randomizes lane order inside each worker block. CPU timing uses a
monotonic clock. CUDA timing uses events on the lane's nondefault stream around
only the resident numerical schedule. Inputs and outputs remain allocated and
resident. Context creation, process startup, compilation, allocation, halo
filling, transfers, and synchronization needed only to read timing events are
outside resident samples. No outlier is removed.

For paired worker-median ratio `PyTorch/DVEB`, PyTorch wins only when the median
is below `0.95` and the bootstrap 95% upper endpoint is below one. DVEB wins
only when the median exceeds `1.05` and the lower endpoint exceeds one.
Otherwise the cell is unresolved. The regime map reports all medians,
dispersion, confidence intervals, schedule metadata, throughput, and peak
memory. It does not interpolate an unobserved crossover.

## F3: bounded interpretation

The primary outcomes are:

1. whether any admitted batch produces a resolved PyTorch resident win;
2. whether the ranking changes or remains stable across the frozen surface;
3. how each backend's time per reconstructed point changes with batch; and
4. which automatically selected DVEB schedules accompany any transitions.

U4-F excludes transfer-inclusive and fresh-launch endpoints because U4-E
already established their fixed-overhead ordering and the present question is
resident amortization. It excludes eager timing, AOT packaging, backward
passes, higher WENO order, true multidimensional fluxes, systems, float32,
multiple CPU threads, datacenter GPUs, and automatic GradFlow backend
selection.

OpenSBLI is not naively looped `B` times and is not rewritten into a new 2-D
application merely to fill the table. Either choice would introduce a new
external schedule rather than test the frozen implementation. Its U4-E batch-
one result remains the independent external anchor; U4-F's new resolved
decisions are only DVEB versus PyTorch.

## Evidence and stop condition

Retain the protocol commit, exact artifact identities, adapter/worker hashes,
commands, deterministic state/output hashes, every qualification diagnostic,
all raw samples, randomized orders, graph records, DVEB queries, telemetry,
exclusions, analysis, and a SHA-256 manifest with an offline verifier. Large
qualification arrays may remain ephemeral because the deterministic generator,
full-array digests, and full-array diagnostics are retained; no sampled-only
correctness test is permitted.

Close U4-F only after all frozen cells complete or retain explicit failures,
the offline verifier and relevant regression tests pass, coherent local
commits exist, and the GradFlow worktree is clean. Do not update the paper
export or push either repository without explicit authorization.
