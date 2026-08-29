# Frozen G1 U0 Evidence

This directory freezes the first counted result for the deliberately unsafe U0
GPU-native formulation. It was completed before U0 was compared with any
numerical oracle.

## Identity

- contract: `u0_unique_f32_component_shared_density_local_lf_forward_euler_v1`
- source parent: recorded in `source_parent_commit.txt`
- compiler and flags: `build.sh`, `nvcc_version.txt`, and `compiler.log`
- machine: `nvidia_smi_q.txt`, `lscpu.txt`, and `uname.txt`
- exact executable: `gradflow_u0`

The copied source, build script, protocol, and executable are the immutable U0
identity. Later candidates must use a new name rather than replace these files.

## Creation commands

From the GradFlow repository root, the executable was rebuilt and recorded
with:

```bash
script -q -e -c experiments/gpu_native_reformulation/native_u0/build.sh \
  experiments/gpu_native_reformulation/evidence/g1_u0_20260829/compiler.log
```

The counted cases were run from the copied frozen executable using the
following arguments:

```text
--size 32  --steps 1  --warmups 5 --repetitions 30 --output-initial u0_n32_initial.f32 --output-state u0_n32_s1_final.f32
--size 32  --steps 10 --warmups 5 --repetitions 30 --output-state u0_n32_s10_final.f32
--size 128 --steps 1  --warmups 5 --repetitions 30
--size 128 --steps 10 --warmups 5 --repetitions 30
```

The first two output paths were expanded into this directory. Standard output
from each case was frozen in the correspondingly named JSON file.

## G1 observations (no correctness claim)

| Grid | Steps | Median resident time | Median per step | Finite |
|---:|---:|---:|---:|:---:|
| 32^3 | 1 | 0.016384 ms | 0.016384 ms | yes |
| 32^3 | 10 | 0.165968 ms | 0.016597 ms | yes |
| 128^3 | 1 | 0.515168 ms | 0.515168 ms | yes |
| 128^3 | 10 | 5.206560 ms | 0.520656 ms | yes |

The `N=128` sustained observation advances about 4.03 billion cells/s and
constructs about 12.08 billion directional faces/s. This is device-resident
CUDA-event timing, not start-to-finish application latency.

The compiler reports 72 registers/thread for the face kernel, 40 for the CFL
and update kernels, and no spills. Each step has four launches: block-reduced
CFL, CFL finish, face construction, and conservative update.

These numbers cannot be described as a speedup over qualified GradFlow or DVEB:
U0 deliberately changes precision, characteristic reconstruction, LF policy,
endpoint storage, nonlinear-weight reuse, and time integration. G2 determines
what numerical damage purchased this frontier.

## Verification

Run this command inside this directory:

```bash
sha256sum --check SHA256SUMS
```

The raw `.f32` files are component-major arrays of five `float32` fields on a
unique periodic `32 x 32 x 32` grid. Each file is exactly 655,360 bytes.
