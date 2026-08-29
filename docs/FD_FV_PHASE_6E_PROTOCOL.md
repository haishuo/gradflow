# FD/FV Euler Phase-6E reproducibility and prepared-execution protocol

Status: frozen before any Phase-6E numerical run, package build, or timing.

Freeze date: 2026-08-29 UTC.

## Purpose

Phase 6E addresses the correctness and deployment boundaries exposed by Phase
6D. It asks three deliberately separate questions:

1. Are fresh-process float64 CUDA shock solutions numerically reproducible even
   when their terminal byte hashes differ?
2. Can packaged AOTInductor remove runtime compilation while preserving the
   exact adaptive CFL and SSP-RK3 mathematics?
3. Can ordinary PyTorch express and package the entire adaptive loop without
   hidden host/device synchronization inside that loop?

The third question is not assumed to have a positive answer. A packaged
one-step module called from Python remains `host_controlled_aot`; it is not
called device-autonomous. A full-loop endpoint earns
`device_autonomous_aot` only after explicit export, numerical, residency, and
movement gates pass.

Correctness > performance > convenience remains governing law.

## Inherited authority and exclusions

Phases 6A--6D continue to govern the Euler formulations, projections, shock
oracles, WENO-JS5 coefficients, global characteristic matrix-LF policy,
transmissive boundaries, SSP-RK3 method, CFL `0.1`, float64 reference policy,
hardware vocabulary, and result provenance.

Phase 6E does not change a numerical formula, CFL constant, stopping time,
boundary, precision, or oracle. It does not add fixed-step substitution,
mixed precision, WENO-Z, another order, another dimension, Navier--Stokes,
custom CUDA/Triton/C++, CUDA graphs, DVEB changes, data-center hardware, or
production API changes. It does not optimize after observing a result.

The study is restricted to `N=800` Sod and Shu--Osher, both FD and FV, on the
admitted Forge RTX 5070 Ti and Ryzen 7600X environment.

## Required admission

Before any Phase-6E execution:

1. the independent Phase-6D verifier passes;
2. the tree is clean at a committed protocol revision;
3. Forge CUDA is visible and freshly passes the inherited float64 compiled
   stage-parity probe;
4. the Phase-6A projections and shock thresholds retain their committed hashes;
5. production numerical source hashes match Phase 6D; and
6. the canonical output directory does not exist.

Every failed worker, package build, export attempt, or movement probe remains
part of the record.

## Lane A: retained-array CUDA reproducibility qualification

Run one isolated CPU-eager numerical authority and five isolated CUDA-compiled
replicates for every `(problem, method)` pair:

```text
problem = (sod, shu_osher)
method  = (fd, fv)
cells   = 800
dtype   = float64
```

This produces 24 workers. The CPU authorities are qualification references,
not newly selected performance endpoints. Every worker retains its terminal
conserved array as `.npy`; JSON records its exact array hash, steps, physical
time, state minima, oracle errors, feature metrics, execution identity, and
file hash. Arrays are copied to the host only after the complete solve.

For each CUDA output, independently compare it with its CPU authority and with
every other CUDA replicate. Record:

- exact byte-hash equality and equal-element count;
- maximum and mean absolute difference;
- RMS difference;
- normalized L1 and L2 difference using
  `max(norm(reference), tiny)`; and
- the component/index and values at maximum absolute difference.

Exact byte identity remains a diagnostic. Phase 6D is not retroactively
reclassified. Phase 6E numerical reproducibility uses a prospective
accumulated-roundoff envelope derived before arrays are observed. For a
comparison with `s` steps, machine epsilon `eps`, and reference scale
`M=max(1, ||reference||inf)`, require:

```text
maximum absolute difference <= 128 * eps * max(1, s) * M
normalized L1 difference    <= 128 * eps * max(1, s)
normalized L2 difference    <= 128 * eps * max(1, s)
```

The factor accounts conservatively for two independent reduction/execution
histories and follows the project's existing step-accumulated roundoff logic.
It is not chosen from the Phase-6E outputs. Every output must also pass the
unchanged physical/oracle gates and use the same step count as its authority.

Lane A passes only if all CPU authorities and CUDA workers pass, all retained
files verify, and every required comparison lies inside the envelope.

## Lane B: packaged host-controlled AOT qualification

Only after Lane A passes, build one fixed-shape AOTInductor package for each
problem/method pair. Package construction is preparation and is never charged
to a prepared runtime endpoint, but build duration, package size, source
commit, command, environment, and SHA-256 are recorded.

The packaged module performs exactly one adaptive advance:

1. compute the unchanged method-specific global CFL timestep on device;
2. clamp it to the device-resident remaining physical time;
3. execute the unchanged SSP-RK3 stages; and
4. return the next state, used timestep, and on-device stage admissibility
   diagnostics.

Python may read the returned scalar timestep and diagnostics to control the
outer loop. That transfer is declared and counted. The endpoint is therefore
`host_controlled_aot`, not device-autonomous.

Qualification requires:

- export and package construction succeed without custom operators;
- one-step AOT/eager maximum absolute difference is at most `5e-11`;
- the complete terminal array passes the Lane-A accumulated-roundoff envelope
  against the corresponding CPU authority and CUDA reference set;
- step count, final time, positivity, and inherited oracle gates pass;
- state input and output are CUDA float64 with shape `(3,800)`; and
- profiler/static inspection identifies no state transfer within the packaged
  numerical module. The declared scalar loop control is reported separately.

Failure blocks Lane-B performance; it does not trigger implementation tuning
inside Phase 6E.

## Lane C: full-loop device-autonomous AOT qualification

Only after Lane A passes, attempt an ordinary-PyTorch full adaptive solver
using structured tensor control flow (`torch.while_loop`) with state, time,
step count, accumulated minima, and failure status carried as tensors. The
body uses exactly the same CFL and SSP-RK3 operations as Lanes A and B.

The loop has a semantic guard of one million steps, matching the inherited
solver. It must stop at the same physical final time and may not replace the
adaptive loop with a fixed-step or host-unrolled computation.

The candidate earns `device_autonomous_aot` only if:

- full-loop export and AOT packaging succeed;
- complete-solve output passes the Lane-A accumulated-roundoff envelope,
  inherited oracle gates, positivity, final-time, and step-count gates;
- input and output state remain CUDA-resident until the declared final
  materialization;
- profiler and generated-wrapper inspection find no D2H copy, scalar
  extraction, graph break, or host synchronization inside the loop; and
- package loading performs no runtime compilation.

If PyTorch/AOTInductor cannot represent or lower this loop, or lowers it to a
host-synchronized loop, record the exact boundary as `unsupported` and do not
time or relabel it. Lane B may still proceed independently.

## Lane D: prepared process-entry performance

Timing begins only for lanes that passed their qualification. The primary
endpoint is a user-visible prepared invocation:

```text
process entry -> imports -> package load -> input construction/transfer
-> complete adaptive solve -> final host materialization -> serialization
-> process exit
```

For each admitted AOT lane, run three fresh isolated replicates for both
problems and methods. Pair them by replicate with the already committed
Phase-6D selected CPU and CUDA-JIT workers. Reuse is permitted only after their
checksums and Phase-6D verifier pass; their raw durations are not rewritten.

A prepared AOT win against an endpoint is confirmed only if all three paired
ratios are below `1/1.05`. A ratio inside the 5% band or any ineligible worker
is unresolved. Also report median ratios descriptively, package-load time,
loop time, final-transfer time, peak RSS, peak CUDA allocation/reservation,
step count, and declared loop-control transfers.

Package build time is reported but excluded only from this explicitly named
prepared endpoint. It must not be omitted from a cold-build interpretation.
Warm or resident measurements may be reported diagnostically but cannot
replace the process-entry result.

## Records and independent verification

The canonical directory is:

```text
experiments/fd_fv_euler/results/phase_6e_20260829/
```

It contains the aggregate, every raw JSON worker record, retained terminal
arrays, package manifests or failure records, and `SHA256SUMS`. Large prepared
binary packages may be stored outside git under a hash-locked artifact
directory; their manifest and restoration/build commands remain committed.

An independent verifier recomputes file hashes, raw statistics, array
comparisons, accumulated bounds, eligibility, package identities, and timing
decisions without rerunning numerical or timed work.

## Stop and claim boundary

Stop after Lane A and every conditionally admitted lane, immutable records,
independent verification, bounded interpretation, complete configured tests,
coherent local commits, and a clean tree.

Do not weaken the new envelope after observing arrays, retroactively confirm
Phase 6D, optimize a failed AOT path, add native code, modify DVEB, begin Phase
6F/7, rent hardware, claim universal determinism, or claim publication
readiness. Do not push without explicit authorization.
