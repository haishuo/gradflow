# Academic U4-C C1 CUDA qualification results

Status: **passed — `cuda_correctness_qualified`**.

Date: 2026-08-30 (UTC)

This is the prospective correctness gate from the committed U4-C protocol. No
comparative performance result was inspected before this gate passed.

## Result

The adapted, pinned OpenSBLI WENO-JS5 operator compiled with OPS CUDA 13.0.88
for native `sm_120` and executed on the NVIDIA GeForce RTX 5070 Ti. Its full
float64 residual passed every frozen finiteness, pointwise, constant-state, and
conservation check.

| case | CUDA vs OPS sequential | CUDA vs GradFlow | tolerance |
|---|---:|---:|---:|
| `state_a` | `7.815970093361102e-14` | `2.1316282072803006e-13` | `2e-12` |
| `state_b` | `2.842170943040401e-14` | `2.842170943040401e-14` | `2e-12` |
| constant `0.37` | `0` | `0` | `2e-12` |

The CUDA residual sums were `8.881784197001252e-16`,
`3.552713678800501e-15`, and exactly zero respectively, all below the frozen
roundoff-scaled conservation bounds. The constant residual was exactly zero.

## Interpretation

This admits the external OpenSBLI CUDA lane to U4-C performance measurement.
It does not show that either implementation is faster, and it does not extend
the U4-B scalar order-5 equivalence to Euler systems, higher orders, or full
applications.

## Evidence

The frozen record is in
`experiments/academic_u4c/evidence/u4c_c1_20260830/`. It includes all three
CPU, CUDA, and GradFlow arrays, build and execution logs, generated-source
hashes, commands, environment metadata, and a SHA-256 manifest. Run
`python experiments/academic_u4c/verify_cuda_qualification.py` for offline
verification.
