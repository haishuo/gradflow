# FD/FV Euler Phase-6F prepared-deployment result

Status: **verified bounded negative deployment result; performance not
admitted**.

Execution date: 2026-08-29 UTC.

## Decision

Phase 6F successfully constructed and restored a hash-locked TorchInductor
runtime cache, and every prepared package remained numerically correct without
changing that cache. The prepared process still invoked the configured C++
compiler executable three times during every package load to query compiler
metadata.

Those invocations were:

```text
/usr/bin/g++ --version
/usr/bin/g++ -v
/usr/bin/g++ --version
```

No runtime invocation issued a C++ compilation command, created a source or
shared object, or changed a cache byte. Nevertheless, the frozen protocol
required **no compiler subprocess**, not merely no compilation. All eight
packages therefore failed the prepared-runtime gate and the conditional
36-worker performance bakeoff was not run.

This distinction matters. Phase 6F demonstrates that prewarming moves the six
generic helper compilations out of the scientific invocation on this stack. It
does not yet demonstrate an invocation independent of a compiler executable.

## Prepared runtime artifact

Preparation began from an empty isolated cache and used the Phase-6E
host-controlled Sod/FD package. The preparation process took `7.710 s` from
launch to exit. Package loading took `6.038 s`; the first CUDA advance took
`0.00499 s`.

The preparation trace records six actual C++ compilation commands in addition
to compiler metadata queries. The resulting cache contains 18 files: six C++
sources, six shared objects, and six lock files. Its contents, modes, sizes,
and SHA-256 identities are committed in the cache manifest.

The deterministic archive is stored outside git:

```text
/mnt/artifacts/gradflow/fd_fv_phase6f_20260829/prepared_runtime_cache.tar.gz
SHA-256: 668d60833943ad44f6548ac86cdd8aa124a5edc31edd881bb067c15506cc20c6
size:    8,540 bytes
```

Restore it into a new runtime cache with:

```bash
mkdir PREPARED_CACHE
tar -xzf /mnt/artifacts/gradflow/fd_fv_phase6f_20260829/prepared_runtime_cache.tar.gz \
  -C PREPARED_CACHE
TORCHINDUCTOR_CACHE_DIR=PREPARED_CACHE <prepared invocation>
```

This cache is qualified only for the recorded Forge software/hardware
environment. No portability or public-redistribution claim is made.

## Prepared package qualification

All eight host-controlled and tensor-loop package solves passed the inherited
float64 numerical gates. Each solve used a private restored cache, and the
cache manifest was identical before and after execution.

| Endpoint | Problem | Method | Load (s) | Solve (s) | Max abs vs CPU | Bound | Steps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| host AOT | Sod | FD | 1.909 | 1.312 | 4.340e-13 | 2.486e-10 | 3,499 |
| host AOT | Sod | FV | 1.947 | 1.386 | 8.079e-13 | 2.486e-10 | 3,499 |
| host AOT | Shu--Osher | FD | 1.911 | 2.482 | 3.469e-9 | 8.312e-9 | 6,892 |
| host AOT | Shu--Osher | FV | 1.932 | 2.604 | 5.208e-11 | 8.317e-9 | 6,894 |
| tensor-loop AOT | Sod | FD | 1.913 | 1.161 | 8.150e-13 | 2.486e-10 | 3,499 |
| tensor-loop AOT | Sod | FV | 1.912 | 1.203 | 8.079e-13 | 2.486e-10 | 3,499 |
| tensor-loop AOT | Shu--Osher | FD | 1.912 | 2.202 | 3.469e-9 | 8.312e-9 | 6,892 |
| tensor-loop AOT | Shu--Osher | FV | 1.920 | 2.345 | 5.208e-11 | 8.317e-9 | 6,894 |

These durations are qualification diagnostics. They were collected under
`strace`, are single observations, and are not repurposed as performance
results.

Every trace contained 36 attempted `g++` path executions because executable
lookup tried multiple path entries; exactly three reached `/usr/bin/g++`.
Those three successful invocations queried version metadata only. This still
fails the prospectively frozen no-compiler-subprocess criterion.

## Tensor-loop causal characterization

Static and dynamic evidence agree that the packaged tensor loop is
host-synchronized on PyTorch `2.13.0+cu130`:

- installed TorchInductor source at line 4259 says that `while_loop` is
  code-generated as a host-side loop;
- the generated wrapper contains `while (1)` at line 3957; and
- the condition crosses the scalar bridge through `aoti_torch_item_bool` at
  line 3994.

Complete-solve profiles found one condition-related host transfer per loop
test, plus profiler/runtime overhead:

| Problem | Method | Steps | `_local_scalar_dense` | D2H copies | Excess over steps |
| --- | --- | ---: | ---: | ---: | ---: |
| Sod | FD | 3,499 | 3,508 | 3,508 | 9 |
| Sod | FV | 3,499 | 3,508 | 3,508 | 9 |
| Shu--Osher | FD | 6,892 | 6,901 | 6,901 | 9 |
| Shu--Osher | FV | 6,894 | 6,903 | 6,903 | 9 |

The endpoint remains
`tensor_loop_aot_prepared_host_synchronized`. Packaging structured control
flow does not make its loop device-autonomous on this compiler version.

## Why no bakeoff appears

Lane D was conditional on all eight prepared packages passing Lane B. They
failed the no-compiler-subprocess requirement, so:

- no Phase-6F performance directory exists;
- no 36-worker CUDA timing matrix was started;
- no Phase-6D CPU number was compared with these qualification durations; and
- no AOT, JIT, FD, or FV performance winner is declared by Phase 6F.

This preserves the original user-visible question. A favorable internal solve
duration cannot substitute for a complete admitted process-entry measurement.

## Artifacts and verification

Committed qualification records:

```text
experiments/fd_fv_euler/results/phase_6f_qualification_20260829/
```

Important hashes:

| Record | SHA-256 |
| --- | --- |
| `qualification.json` | `351413cf7f23a2eff149847a3a9935799f30555a93e28951523e0dcb06da84e5` |
| `SHA256SUMS` | `aa492e469fd539f72221e5ac17cd011bab9bbc6310b502305ef490944a2c694d` |
| prepared cache archive | `668d60833943ad44f6548ac86cdd8aa124a5edc31edd881bb067c15506cc20c6` |

Independent verification:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6f_qualification.py
```

The verifier checks the complete committed checksum set, reconstructs the
archive manifest from tar members, independently parses every process trace,
recomputes eligibility, verifies terminal-array identities and numerical gate
decisions, and checks the static/dynamic loop evidence.

The complete configured Forge regression surface also passed. The ordinary
suite reported `343 passed, 12 skipped, 14 warnings`; the 12 skips are the
explicit external DVEB fixtures. Running those fixtures against the preserved
official portable ABI-v1 manifest, placement model, direct executable, and
device ABI-v2 manifest reported `12 passed`. Thus all 355 configured tests
passed, with the 14 expected upstream PyTorch deprecation warnings.

## Scientific interpretation and next boundary

Phase 6F narrows the deployment problem into two separable costs:

1. **Generic helper compilation is preparable.** Its six compilation commands
   disappeared from every restored-cache invocation, and package load fell
   descriptively from `6.038 s` during preparation to `1.909--1.947 s` during
   traced qualification.
2. **Compiler discovery remains runtime work.** PyTorch still starts `g++`
   three times to identify it, even when no artifact is built.

A future prospectively frozen phase may investigate a truly compiler-free
runtime image or a supported way to persist compiler metadata. Alternatively,
it may define and justify a separate “zero runtime compilation” endpoint that
permits compiler-version interrogation while timing its cost. That definition
must be frozen before new performance data; Phase 6F is not retroactively
reclassified.

Separately, genuine device-side adaptive loop control requires a compiler
lowering different from the current host-side `torch.while_loop` wrapper or a
mathematically requalified execution design. Phase 6F does not implement one.

No production numerical source, DVEB source, precision policy, formulation,
or custom backend code changed. No optimization, universal deployment claim,
or publication claim was made.
