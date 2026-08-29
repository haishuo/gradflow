# FD/FV Euler Phase-6F prepared-deployment protocol

Status: frozen before any Phase-6F cache preparation, qualification, profiling,
or timing.

Freeze date: 2026-08-29 UTC.

## Purpose

Phase 6F follows the deployment boundary exposed by Phase 6E. It asks:

1. Can the generic TorchInductor runtime helpers observed in Phase 6E be
   prepared once, hash-locked, and reused without compilation or cache mutation
   during a fresh scientific invocation?
2. What execution boundary does the packaged `torch.while_loop` actually
   implement on the admitted PyTorch stack?
3. If a prepared runtime passes qualification, how do fresh process-entry
   CUDA JIT, host-controlled AOT, and packaged tensor-loop AOT compare for the
   unchanged float64 Euler shock problems?

This phase distinguishes package construction, runtime-image preparation, and
scientific invocation. Excluding preparation is valid only for an explicitly
named prepared-deployment endpoint. All preparation cost remains reported.

Correctness > performance > convenience remains governing law.

## Inherited authority and exclusions

Phases 6A--6E continue to govern the Euler formulations, characteristic
projection, WENO-JS5 coefficients, global characteristic matrix-LF policy,
transmissive boundaries, SSP-RK3 method, adaptive CFL `0.1`, float64 policy,
shock oracles, accumulated-roundoff envelope, and admitted Forge hardware.

Phase 6F changes no numerical source, equation, coefficient, boundary,
precision, step policy, stopping condition, or oracle. It does not introduce
custom CUDA, Triton, C++, DVEB changes, CUDA graphs, fixed-step substitution,
mixed precision, another WENO order, another dimension, or production API
work. It is not an optimization search.

The numerical study remains restricted to:

```text
problem = (sod, shu_osher)
method  = (fd, fv)
cells   = 800
dtype   = float64
device  = Forge RTX 5070 Ti
```

## Required admission

Before Phase-6F execution:

1. all three independent Phase-6E verifiers pass;
2. the tree is clean at a committed Phase-6F protocol revision;
3. CUDA is visible and the inherited float64 stage-parity probe passes;
4. all Phase-6E packages, authorities, arrays, and manifests match their
   committed SHA-256 identities;
5. production numerical source hashes match Phase 6E; and
6. no canonical Phase-6F output or external artifact directory exists.

Every failed preparation, qualification, profiling, or timing worker remains
part of the record. A failed gate is recorded rather than tuned around.

## Lane A: prepared runtime-cache construction

Starting from an empty isolated `TORCHINDUCTOR_CACHE_DIR`, load and execute a
known-good Phase-6E AOT package once. This is deployment preparation, not a
scientific invocation. Record:

- the exact package and SHA-256 used to prepare the cache;
- process-entry wall time and package-load/first-call durations;
- every cache file path, size, mode, and SHA-256;
- compiler subprocesses observed during preparation;
- Python, PyTorch, CUDA, driver, compiler, OS, CPU, and GPU identity; and
- the preparation command and source commit.

Archive the complete prepared cache outside git as a deterministic `.tar.gz`
and commit its manifest, archive SHA-256, size, construction instructions, and
restoration instructions. The artifact is specific to this recorded software
and hardware environment. Phase 6F makes no portability or redistributability
claim for it.

## Lane B: prepared-cache runtime qualification

For every host-controlled and tensor-loop Phase-6E package, restore a private
copy of the hash-locked prepared cache before starting the worker. Cache
restoration is deployment preparation and occurs before the invocation clock.

Trace process creation and hash the cache before and after each worker. A
package lane passes the zero-runtime-compilation gate only if:

- no C, C++, CUDA, assembler, linker, or build-system subprocess is executed;
- no cache file is created, removed, or modified;
- package and cache identities match the frozen manifests;
- the complete terminal array passes the inherited CPU-authority
  accumulated-roundoff envelope, oracle, positivity, final-time, and exact
  step-count gates; and
- input and terminal state remain CUDA float64, with host materialization only
  at the declared output boundary.

The trace itself is qualification instrumentation and is not timed as a
performance endpoint. If any package requires a new helper for its shape or
control-flow path, the prepared image is incomplete and that lane is rejected.

## Lane C: tensor-loop lowering characterization

The packaged full-loop implementation is prospectively named
`packaged_tensor_loop_host_synchronized`. It may not be called
device-autonomous unless new evidence overturns the Phase-6E observation.

Preserve and hash the relevant installed TorchInductor source and generated
AOT wrapper. Static inspection records whether:

- TorchInductor documents `torch.while_loop` as host-side code generation;
- the generated wrapper contains a host `while` loop; and
- the loop condition is extracted through `aoti_torch_item_bool` or an
  equivalent scalar bridge.

For one complete solve per problem/method pair, profile the package call and
record counts for `aten::_local_scalar_dense`, device-to-host copies, and CUDA
synchronization evidence. Relate event counts to the recorded adaptive step
count where possible. Profiling is characterization, not performance timing.

On the admitted PyTorch 2.13 stack, local source inspection before protocol
freeze found the explicit statement that `torch.while_loop` is code-generated
as a host-side loop, and a generated Phase-6E wrapper called
`aoti_torch_item_bool` for the loop condition. The phase tests and preserves
that implementation fact; it does not search for an undocumented toggle.

## Lane D: conditional prepared process-entry bakeoff

Lane D runs only after Lane B passes. It compares these accurately named
endpoints:

1. `cuda_jit_process_entry`: ordinary compiled PyTorch, including the
   problem-specific WENO compilation in every fresh invocation;
2. `host_controlled_aot_prepared`: the packaged one-advance module with the
   declared Python adaptive loop and scalar transfers; and
3. `tensor_loop_aot_prepared_host_synchronized`: the packaged tensor-control
   loop with its characterized host condition bridge.

The JIT lane receives the same generic prepared helper cache but no previously
compiled WENO graph. This prevents generic runtime-helper setup from being
charged to JIT alone while retaining the intended JIT compilation cost.

Each endpoint receives three fresh, isolated workers for every problem/method
pair: 36 CUDA workers total. Every worker retains its terminal array. Timing is
measured by a parent process over the user-visible invocation:

```text
process entry -> imports -> JIT or package load -> input construction/transfer
-> complete adaptive solve -> final host materialization -> result serialization
-> process exit
```

Cache restoration, AOT package construction, and runtime-cache preparation
are outside this explicitly prepared invocation. Their costs and break-even
interpretation are reported separately and never silently discarded.

Reuse the three selected Phase-6D CPU workers per problem/method only after
their exact identities and the Phase-6D verifier pass. Those endpoints remain
`cpu_eager` for Sod and `cpu_compiled` for Shu--Osher. Their recorded durations
are not rewritten.

Pair repetitions by index. A CUDA endpoint is a confirmed win over a comparator
only if all three paired duration ratios are below `1/1.05`. Any ratio in the
5% practical-equivalence band, or any ineligible worker, makes the comparison
unresolved. Report every raw duration, median ratio, full-solve step count,
retained-array comparison, peak RSS, peak CUDA allocation/reservation, package
load duration, solve duration, and final-transfer/serialization duration.

Warm in-process and device-resident timings may not substitute for this
process-entry endpoint. Phase-6E qualification durations may not be reused as
performance observations.

## Records and independent verification

Canonical committed directories are:

```text
experiments/fd_fv_euler/results/phase_6f_qualification_20260829/
experiments/fd_fv_euler/results/phase_6f_performance_20260829/
```

The second directory exists only if Lane D is admitted. Large cache/package
artifacts remain outside git under:

```text
/mnt/artifacts/gradflow/fd_fv_phase6f_20260829/
```

Committed manifests contain exact identities and restoration commands. An
independent verifier recomputes hashes, cache before/after equality, traced
subprocess decisions, numerical comparisons, eligibility, raw statistics, and
timing decisions without rerunning numerical or timed work.

## Stop and claim boundary

Stop after all conditionally admitted lanes, immutable records, independent
verification, bounded interpretation, the complete configured test suite,
coherent local commits, and a clean tree.

Do not weaken a gate after observing results, repair or optimize a losing lane,
relocate work outside the timed boundary, relabel a host-synchronized loop,
change DVEB or production numerics, begin Phase 6G/7, claim deployment
portability, claim universal performance, or claim publication readiness. Do
not push without explicit authorization.
