# FD/FV Euler Phase-6E reproducibility and prepared-execution result

Status: **verified bounded result; no AOT performance lane admitted**.

Execution date: 2026-08-29 UTC.

## Decision

Phase 6E resolves the Phase-6D CUDA terminal-hash question positively at the
numerical level, but neither candidate AOT execution boundary passes the
prospectively frozen deployment gate.

- Fresh-process float64 CUDA shock outputs are reproducible inside the
  preregistered step-accumulated roundoff envelope.
- Fixed-shape host-controlled and full-loop AOTInductor packages can be built
  from the unchanged ordinary-PyTorch FD/FV Euler source.
- Every package produced a numerically eligible complete shock solution.
- A pristine AOT package load nevertheless compiled six generic TorchInductor
  C++ runtime helper probes.
- The full tensor-control-flow package retained device-to-host scalar
  synchronization inside its lowered loop.

Because zero runtime compilation and genuine device autonomy were explicit
qualification requirements, no Phase-6E AOT lane reached performance timing.
The project does not time a failed configuration and then present the number
as if it answered the intended question.

## Lane A: CUDA numerical reproducibility

All 24 workers passed: one CPU-eager authority and five fresh CUDA-compiled
replicates for each of Sod/Shu--Osher and FD/FV. All 60 required CPU/CUDA and
CUDA/CUDA retained-array comparisons passed.

| Comparison | Problem | Method | Exact pairs | Pairs | Worst max abs | Frozen abs bound |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CUDA/CUDA | Sod | FD | 6 | 10 | 4.052e-13 | 2.486e-10 |
| CUDA/CUDA | Sod | FV | 6 | 10 | 4.221e-13 | 2.486e-10 |
| CUDA/CUDA | Shu--Osher | FD | 10 | 10 | 0 | 8.312e-9 |
| CUDA/CUDA | Shu--Osher | FV | 10 | 10 | 0 | 8.317e-9 |
| CPU/CUDA | Sod | FD | 0 | 5 | 8.168e-13 | 2.486e-10 |
| CPU/CUDA | Sod | FV | 0 | 5 | 8.255e-13 | 2.486e-10 |
| CPU/CUDA | Shu--Osher | FD | 0 | 5 | 3.743e-9 | 8.312e-9 |
| CPU/CUDA | Shu--Osher | FV | 0 | 5 | 2.540e-10 | 8.317e-9 |

Every pair also passed the normalized L1/L2 bounds and had the same adaptive
step count. The two Sod CUDA variants explain the Phase-6D hash instability;
their numerical separation is approximately `4e-13`, not a materially
different shock solution. Both Shu--Osher methods were bitwise identical
across all five new CUDA processes.

This does not retroactively change Phase 6D, whose exact-hash gate correctly
remains failed. It establishes a new, prospectively bounded Phase-6E
reproducibility result using retained arrays rather than hashes alone.

## Host-controlled packaged AOT

All four one-advance packages built successfully. Package export took
`0.79--0.83 s`, package compilation took `12.39--12.50 s`, and package sizes
were `2.64--2.75 MB`. Build time is preparation data, not runtime timing.

One-step AOT/eager maximum differences were at most `7.11e-15`. All four
complete AOT solves passed their CPU-authority roundoff envelope, inherited
shock oracle, positivity, final-time, and exact step-count gates:

| Problem | Method | Complete max abs vs CPU | Bound | Steps |
| --- | --- | ---: | ---: | ---: |
| Sod | FD | 4.340e-13 | 2.486e-10 | 3,499 |
| Sod | FV | 8.079e-13 | 2.486e-10 | 3,499 |
| Shu--Osher | FD | 3.469e-9 | 8.312e-9 | 6,892 |
| Shu--Osher | FV | 5.208e-11 | 8.317e-9 | 6,894 |

The package contains prebuilt CUDA cubins and a wrapper shared object. Under a
fresh runtime cache, however, `aoti_load_package` created and compiled six
small generic C++ helper sources/shared objects before the first call. The
same helper identities appeared for all packages. They are not generated WENO
CUDA kernels, but they are still runtime compilation in a pristine process.
The frozen gate therefore rejects the zero-runtime-compilation claim.

The outer adaptive loop also deliberately reads the timestep and stage
admissibility diagnostics on the host. This lane was always labeled
`host_controlled_aot`; it makes no device-autonomy claim.

## Full-loop packaged AOT

The initial device-loop attempt is preserved. All four exports failed because
the harness passed one aliased scalar tensor as both initial density and
pressure minima, which violates `torch.while_loop`'s input contract. The
prospective amendment authorized one legality correction—two distinct scalar
tensors—and no optimization.

After that correction, all four full-loop modules exported, packaged, and
completed. Export took `2.89--2.91 s`, package compilation took
`13.51--13.57 s`, and package sizes were `2.86--3.00 MB`. All four outputs
passed exactly the same numerical, oracle, positivity, final-time, and
step-count gates shown above.

The result is mathematically useful but not device-autonomous. Profiling every
complete packaged loop found both:

- `aten::_local_scalar_dense`; and
- `Memcpy DtoH (Device -> Pinned)`.

These events occurred inside the profiled package call, before the declared
final result materialization. AOTInductor preserved the structured-loop
semantics but lowered the loop condition through the host on this software
stack. Package loading also compiled the same six runtime C++ helpers.

Consequently:

```text
ordinary PyTorch can express/export/package the adaptive loop: yes
the packaged loop is numerically correct:                    yes
the packaged loop is device-autonomous here:                 no
the pristine package has zero runtime compilation here:      no
```

## Why no timing result appears

Lane D was conditional. It admitted only an AOT lane that had already passed
its numerical, no-runtime-compilation, and execution-boundary gates. Both AOT
lanes failed at least one of those gates, so:

- no AOT process-entry timing workers were run;
- no favorable qualification duration is repurposed as a benchmark;
- no comparison with Phase-6D CPU or JIT timing is made; and
- `performance_measurements_collected` remains false in every Phase-6E record.

This is a correctness-first negative result, not a missing benchmark.

## Artifacts and verification

Committed records:

- Lane A: `experiments/fd_fv_euler/results/phase_6e_20260829/`
- initial AOT attempt: `experiments/fd_fv_euler/results/phase_6e_aot_20260829/`
- corrected device attempt:
  `experiments/fd_fv_euler/results/phase_6e_device_r1_20260829/`

Aggregate and manifest hashes:

| Record | Aggregate SHA-256 | `SHA256SUMS` SHA-256 |
| --- | --- | --- |
| Lane A | `657f4595a3b3ee3e714f14793d17c789550cfa81b131746beb973b8550fa368e` | `15bb1ee0b9e8f69543331a522173047b840c3f4417b74616d40a122300dfde6d` |
| Initial AOT | `1453456f64afa88f14d3eb409396e58b1e628076aa5190ab8788a4d2728389e3` | `b968410ffd56c2a35eb8391b6c7b7057dcdc21e5f663b385a47f1908d9670731` |
| Device r1 | `0def89da16e13288fcc0bff0bc2849a2065dea4ce2f03bc871b4699765cf555c` | `c3795b772daf5f286a1f0ae2190bcc30082b78afaac3de2bb3b22d358ecdec4c` |

The eight successful packages are stored outside git under:

```text
/mnt/artifacts/gradflow/fd_fv_phase6e_20260829/
/mnt/artifacts/gradflow/fd_fv_phase6e_device_r1_20260829/
```

Their exact paths, sizes, SHA-256 identities, build commands, versions, and
build durations are committed in the build records. Independent verification:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6e_repro.py
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6e_aot.py
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6e_device_r1.py
```

## Scientific interpretation and next boundary

Phase 6E separates three issues that had previously been conflated:

1. CUDA byte-hash variation did not imply a materially different answer; with
   retained arrays, the observed variation is bounded accumulated float64
   roundoff.
2. AOT packaging successfully moves the WENO-generated CUDA kernels and model
   wrapper to preparation time, but a pristine PyTorch runtime still performs
   generic helper compilation on first package load.
3. Structured tensor control flow packages the adaptive algorithm but does not
   guarantee device-side loop control; backend lowering determines whether
   scalar synchronization remains.

A future phase may prospectively test a deployable preprepared runtime image
that includes the generic helper cache, because that is a different boundary
from a pristine package load. It may also investigate compiler-supported
device loop control or a mathematically qualified alternative. Neither result
is assumed, and neither may reuse Phase-6E qualification durations as timing.

No production source, DVEB source, precision policy, formulation, native code,
or custom kernel was changed. No optimization or publication claim was made.
