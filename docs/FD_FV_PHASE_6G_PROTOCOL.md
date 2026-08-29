# FD/FV Euler Phase-6G compiler-independent AOT protocol

Status: frozen before any Phase-6G loader qualification, process trace, or
performance measurement.

Freeze date: 2026-08-29 UTC.

## Purpose

Phase 6G follows the exact boundary exposed by Phase 6F. It asks:

1. Are the three runtime `g++` metadata queries caused by the documented
   Python AOT-package loading layer rather than the prepared binary package?
2. Can PyTorch's underlying AOTI package loader execute the unchanged package
   with zero compiler/build subprocesses and zero cache mutation?
3. If that internal, version-locked loader passes, what is its honest fresh
   process-entry performance relative to CUDA JIT and the frozen CPU
   comparators?

Phase 6G does not retroactively weaken Phase 6F. Its gate still requires zero
compiler executable invocation. It tests a different loader boundary frozen
in advance.

Correctness > performance > convenience remains governing law.

## Causal hypothesis frozen before execution

On PyTorch `2.13.0+cu130`, local source inspection found that:

- `torch._inductor.aoti_load_package` delegates to the Python
  `torch._inductor.package.load_package` layer;
- that package module imports `CppBuilder` and Inductor configuration; and
- the actual package is ultimately opened by
  `torch._C._aoti.AOTIModelPackageLoader`.

The prospective hypothesis is that direct construction of the underlying AOTI
loader and use of its `boxed_run` method may avoid Python compiler discovery.
This is not assumed to succeed.

The direct loader is an internal PyTorch API. If successful, it is labeled
`internal_aoti_loader`; it is not called stable, public, portable, or ready for
the GradFlow user API. Its exact PyTorch source/binary identity must be pinned.

## Inherited authority and exclusions

Phases 6A--6F continue to govern the Euler formulations, WENO-JS5
coefficients, characteristic projection, global characteristic matrix-LF
policy, transmissive boundaries, SSP-RK3 method, adaptive CFL `0.1`, float64
policy, shock oracles, accumulated-roundoff envelope, prepared cache, AOT
packages, and admitted Forge hardware.

Phase 6G changes no numerical source, equation, coefficient, boundary,
precision, timestep, stopping condition, or oracle. It introduces no custom
CUDA, Triton, C++, Cython, DVEB change, CUDA graph, fixed-step substitution,
mixed precision, new WENO order, or production API. The fixed output arities
and order passed to `boxed_run` come from the already-qualified exported
package contracts; no numerical behavior is reimplemented.

The numerical study remains:

```text
problem = (sod, shu_osher)
method  = (fd, fv)
cells   = 800
dtype   = float64
device  = Forge RTX 5070 Ti
```

## Required admission

Before Phase-6G execution:

1. the Phase-6F independent verifier passes;
2. the tree is clean at a committed Phase-6G protocol and harness revision;
3. CUDA and the inherited float64 parity gate pass;
4. the prepared-cache archive, all eight packages, CPU authorities, and
   Phase-6F records match their frozen SHA-256 identities;
5. the installed PyTorch Python source and binary identities match the
   Phase-6F environment; and
6. no canonical Phase-6G output directory exists.

Every failed worker or trace remains in the record. No loader implementation
is edited or tuned after results are observed.

## Lane A: internal-loader qualification

For each host-controlled and tensor-loop package, restore a private copy of
the Phase-6F prepared cache and run one complete solve through:

```text
torch._C._aoti.AOTIModelPackageLoader(...).boxed_run(flat_tensor_inputs)
```

Trace all process creation and retain the terminal array. Qualification
requires:

- no invocation or attempted lookup of `g++`, `gcc`, `clang`, `cc`, `nvcc`,
  assembler, linker, CMake, Make, or Ninja;
- no child Python compilation/helper process attributable to Inductor;
- no cache file creation, removal, mode change, or content change;
- package, cache, PyTorch, CUDA, and source identities match;
- terminal state passes the inherited CPU-authority accumulated-roundoff,
  oracle, positivity, final-time, and exact step-count gates;
- input and terminal state are CUDA float64 with shape `(3,800)`; and
- host materialization occurs only at the already-declared control/output
  boundaries.

The host-controlled AOT lane retains its per-step Python scalar transfers. The
tensor-loop lane remains host-synchronized inside the generated wrapper. This
phase removes neither behavior and makes no device-autonomy claim.

The endpoint earns `compiler_free_internal_aoti` only if all eight packages
pass. A partial pass is scientifically recorded but does not admit timing.

## Lane B: causal public-loader control

Phase 6F already contains eight prospective traces showing that the documented
public loader performs three successful compiler metadata queries per process.
Those immutable traces remain the primary public-loader control and are not
rerun merely to obtain another observation.

Phase 6G independently verifies their hashes and parses successful compiler
commands separately from failed executable-search attempts. It records:

- three successful metadata queries per Phase-6F package invocation;
- zero runtime compilation commands after cache preparation; and
- byte-identical caches before and after those invocations.

This causal reuse is declared before execution and is not performance reuse.

## Lane C: conditional process-entry performance

Lane C runs only if all eight internal-loader qualifications pass. Compare:

1. `cuda_jit_process_entry`: ordinary `torch.compile`, with problem-specific
   WENO compilation charged to each fresh invocation;
2. `host_controlled_aot_internal_loader_prepared`; and
3. `tensor_loop_aot_internal_loader_prepared_host_synchronized`.

Each endpoint receives three fresh isolated workers for every problem/method
pair: 36 new CUDA workers. Every worker starts with an independently restored
copy of the same generic prepared cache and retains its terminal array. The JIT
cache contains no problem-specific compiled WENO graph.

Timing is measured by the parent process over:

```text
process entry -> imports -> JIT or package load -> input construction/transfer
-> complete adaptive solve -> final host materialization -> result serialization
-> process exit
```

Prepared-cache restoration, original AOT package construction, and cache
construction remain deployment preparation outside this explicitly prepared
invocation. Their costs are reported separately and included in break-even
analysis.

Reuse the three selected Phase-6D CPU workers only after exact verification:
Sod uses `cpu_eager`; Shu--Osher uses `cpu_compiled`. Their durations are not
rewritten.

Pair repetitions by index. A numerator is a confirmed win only when all three
ratios are below `1/1.05`; all three ratios inside `[1/1.05, 1.05]` establish
practical equivalence; all three above `1.05` establish a denominator win.
Mixed or ineligible results are unresolved. Report raw durations, median
ratios, package-load and solve durations, final materialization duration,
memory, step counts, and retained-array comparisons.

No warm or resident timing may replace this process-entry endpoint. Phase-6F
qualification diagnostics are not performance observations.

## Records and independent verification

Canonical committed directories are:

```text
experiments/fd_fv_euler/results/phase_6g_qualification_20260829/
experiments/fd_fv_euler/results/phase_6g_performance_20260829/
```

The performance directory exists only if Lane C is admitted. Large prepared
artifacts remain outside git and are referenced by their Phase-6F identities.

Independent verifiers recompute checksum sets, process-trace classifications,
cache equality, numerical eligibility, raw statistics, pairing, decisions,
and break-even arithmetic without rerunning numerical or timed work.

## Stop and claim boundary

Stop after every conditionally admitted lane, immutable records, independent
verification, bounded interpretation, complete configured tests, coherent
local commits, and a clean tree.

Do not weaken a gate after observing results, patch installed PyTorch, hide or
rename a subprocess, introduce a wrapper executable, move work outside the
timed interval, relabel the internal loader as public, relabel the tensor loop
as device-autonomous, change numerical or DVEB code, begin Phase 6H/7, or claim
publication readiness. Do not push without explicit authorization.
