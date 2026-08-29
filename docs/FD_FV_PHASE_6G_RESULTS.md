# FD/FV Euler Phase-6G compiler-independent AOT result

Status: **verified positive deployment and process-entry result**.

Execution date: 2026-08-29 UTC.

## Decision

Phase 6G identified and qualified a genuinely compiler-free prepared AOT
invocation on the admitted PyTorch `2.13.0+cu130` stack.

Direct use of PyTorch's internal
`torch._C._aoti.AOTIModelPackageLoader` bypassed the Python Inductor packaging
layer that caused Phase 6F's runtime compiler-discovery probes. Across all
eight FD/FV, Sod/Shu--Osher, host/tensor-loop packages:

- no compiler, assembler, linker, or build executable was attempted;
- no Inductor child Python helper process was launched;
- the prepared cache remained byte-for-byte unchanged;
- all terminal arrays passed the inherited float64 numerical gates; and
- package loading took approximately `8.6--10.4 ms` under process tracing.

That qualification admitted the frozen 36-worker process-entry bakeoff. Both
prepared AOT endpoints decisively beat fresh CUDA JIT and the frozen selected
CPU endpoint in every problem/method pair.

The loader is an internal, version-locked PyTorch API. This is a valid research
deployment result, not yet a stable public GradFlow interface.

## Why Phase 6F and Phase 6G differ

The Phase-6F public loader traces remain unchanged. Every public
`torch._inductor.aoti_load_package` invocation:

- attempted 36 compiler paths;
- reached `/usr/bin/g++` three times for `--version`, `-v`, and `--version`;
- issued zero runtime compilation commands after cache preparation; and
- left the prepared cache unchanged.

Local source shows that the public function delegates through the Python
`torch._inductor.package` layer before constructing the same underlying AOTI
loader. The internal Phase-6G path constructs that loader directly and invokes
`boxed_run` with the exported package's frozen tensor contract.

Every Phase-6G trace contained only three process executions: `strace`, the
worker Python process, and `ldconfig`. No compiler lookup or child Python
helper appeared. This supports the preregistered causal hypothesis that the
remaining Phase-6F probes belonged to the Python loading layer, not to the AOT
package's numerical execution.

## Numerical qualification

All eight internal-loader solutions used the same step counts and remained
inside the Phase-6E accumulated-roundoff bounds:

| Endpoint | Problem | Method | Load (ms) | Solve (s) | Max abs vs CPU | Bound | Steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| host AOT | Sod | FD | 8.74 | 1.273 | 4.340e-13 | 2.486e-10 | 3,499 |
| host AOT | Sod | FV | 9.23 | 1.346 | 8.079e-13 | 2.486e-10 | 3,499 |
| host AOT | Shu--Osher | FD | 8.56 | 2.404 | 3.469e-9 | 8.312e-9 | 6,892 |
| host AOT | Shu--Osher | FV | 8.84 | 2.529 | 5.208e-11 | 8.317e-9 | 6,894 |
| tensor-loop AOT | Sod | FD | 9.81 | 1.162 | 8.150e-13 | 2.486e-10 | 3,499 |
| tensor-loop AOT | Sod | FV | 10.35 | 1.205 | 8.079e-13 | 2.486e-10 | 3,499 |
| tensor-loop AOT | Shu--Osher | FD | 9.76 | 2.202 | 3.469e-9 | 8.312e-9 | 6,892 |
| tensor-loop AOT | Shu--Osher | FV | 10.39 | 2.342 | 5.208e-11 | 8.317e-9 | 6,894 |

These traced qualification durations are not used as benchmark observations.

## Fresh process-entry bakeoff

Each new CUDA endpoint has three isolated repetitions. Durations include
process entry, imports, JIT compilation or package load, input construction and
transfer, the complete adaptive solve, terminal host materialization,
serialization, and process exit. Prepared-cache restoration is explicitly
deployment preparation outside this endpoint.

Median complete process-entry durations:

| Problem | Method | Frozen CPU (s) | Fresh CUDA JIT (s) | Host AOT (s) | Tensor-loop AOT (s) | JIT/tensor speedup | CPU/tensor speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sod | FD | 17.175 | 13.647 | 2.413 | 2.293 | 5.95x | 7.49x |
| Sod | FV | 17.192 | 13.686 | 2.495 | 2.324 | 5.89x | 7.40x |
| Shu--Osher | FD | 21.290 | 15.795 | 3.533 | 3.330 | 4.74x | 6.39x |
| Shu--Osher | FV | 22.585 | 15.717 | 3.648 | 3.462 | 4.54x | 6.52x |

Every prepared-AOT/JIT and prepared-AOT/CPU comparison is a confirmed AOT win
under the prospectively frozen rule: all three paired ratios clear the 5%
practical-significance threshold.

Fresh CUDA JIT also beats the selected CPU endpoint in all four cases, but it
pays approximately `11--12 s` more per invocation than prepared AOT because
problem-specific compilation is inside the endpoint.

## Host-controlled versus tensor-loop AOT

The tensor-loop package has the lower median process-entry duration in all
four cases:

- Sod FD: `0.950` tensor/host median ratio, but mixed paired thresholds make
  the result unresolved;
- Sod FV: `0.931`, confirmed tensor-loop win;
- Shu--Osher FD: `0.942`, confirmed tensor-loop win; and
- Shu--Osher FV: `0.949`, confirmed tensor-loop win.

This does not make the tensor loop device-autonomous. Phase 6F proved that its
generated wrapper performs one D2H condition extraction per loop test. Phase
6G shows that, at `N=800` on this stack, consolidating the complete solve into
one packaged call usually outweighs that remaining host condition bridge.

## Preparation and break-even

The one-time host package builds took approximately `13.3--13.4 s`; tensor-loop
package builds took `16.5--16.6 s`. The shared generic runtime-cache preparation
took `7.710 s`.

Against rebuilding fresh JIT on every invocation:

- host-controlled AOT amortizes its package build in 2 invocations for all
  four cases;
- tensor-loop AOT amortizes its package build in 2 invocations for all four;
  and
- conservatively charging the entire shared cache preparation to each single
  package gives 3 invocations for the two Sod tensor-loop cases and 2 for all
  others.

The cache preparation is actually shared among packages, so the conservative
per-package calculation overcharges it in a multi-solver deployment. These
counts do not include engineering, storage, validation, or redistribution
costs.

## What this establishes

Within the frozen scope, Phase 6G establishes:

1. PyTorch AOT packages can execute the unchanged FD/FV WENO-JS5 Euler shock
   solvers without a compiler or runtime-generated artifact during invocation.
2. The documented high-level loader's remaining compiler probes are avoidable
   loading-layer overhead on this PyTorch version, not a requirement of the
   packaged numerical kernels.
3. Prepared process-entry AOT is `4.54--5.95x` faster than fresh JIT and
   `6.39--7.49x` faster than the selected CPU endpoints for these four
   `N=800`, float64 cases on Forge.
4. Preparation amortizes extremely quickly when a solver configuration is run
   repeatedly.

It does not establish that the internal API is stable, that the package is
portable to another PyTorch/CUDA/hardware stack, that tensor control flow is
device-autonomous, that AOT is optimal at every grid size, or that FD is
universally faster than FV.

The result materially strengthens the case for an ahead-of-time deployment
path in GradFlow, provided the internal-loader dependency is isolated behind a
version-qualified adapter and eventually replaced by a stable public runtime
surface or upstream-supported equivalent.

## Artifacts and verification

Qualification records:

```text
experiments/fd_fv_euler/results/phase_6g_qualification_20260829/
```

Performance records:

```text
experiments/fd_fv_euler/results/phase_6g_performance_20260829/
```

Important hashes:

| Record | SHA-256 |
| --- | --- |
| qualification aggregate | `6555c99564625cc441ce369e6ed3f7348ead940100dd38e0d93718f617979e15` |
| qualification `SHA256SUMS` | `ea7c9d3005ab6b0687a4a927e01b60bc3af9a100a620e9d6a3adda9a8ad1211a` |
| performance aggregate | `da1bf77dc17d6904931f9af98aaa710852f40519f7fa13ad1817fa58f245a6a1` |
| performance `SHA256SUMS` | `0dda91524343076af707afe3e521c419956c7dd1f34d886472c592f5b76a5309` |

Independent verification:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6g_qualification.py
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6g_performance.py
```

Both verifiers pass. They independently recompute record checksums, process
classifications, cache equality, numerical eligibility, retained-array
identities, timing pairings, decisions, and break-even arithmetic.

The complete configured Forge regression surface also passed. The ordinary
suite reported `343 passed, 12 skipped, 14 warnings`; the 12 explicit external
DVEB fixtures then reported `12 passed` against the preserved official
artifacts. All 355 configured tests therefore passed, with only the 14 expected
upstream PyTorch deprecation warnings.

No production numerical source, DVEB source, precision policy, formulation,
or custom backend code changed. No portability, universal-performance, or
publication-readiness claim is made.
