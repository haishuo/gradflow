# Academic U4-E E1 qualification results

Status: **all six lanes admitted; comparative timing not yet run**.

Date: 2026-08-31 (UTC)

The immutable DVEB Trunk 005 handoff passed bundle and member verification,
public-header compilation as C11 and C++17, ABI-v1 loading, and the frozen
scalar WENO-JS5 correctness gate. The unchanged OpenSBLI/OPS and
PyTorch/TorchInductor lanes were requalified from the same `N=8192` input
before any U4-E timing.

Here “GradFlow” in the frozen evidence key means the PyTorch/TorchInductor
backend, not the encompassing GradFlow system. Human-facing names use the
backend-specific term below; the hashed evidence keys remain unchanged.

| lane | maximum normalized error | RMS normalized error | conservative |
|---|---:|---:|---|
| DVEB CPU | `1.5701e-15` | `2.4533e-17` | yes |
| DVEB CUDA | `2.5121e-14` | `3.6152e-15` | yes |
| OpenSBLI CPU | `1.1305e-12` | `2.4567e-14` | yes |
| OpenSBLI CUDA | `8.4156e-13` | `2.1027e-14` | yes |
| PyTorch/TorchInductor CPU | `6.2803e-15` | `2.6611e-15` | yes |
| PyTorch/TorchInductor CUDA | `2.5121e-14` | `4.2653e-15` | yes |

Every lane is below the frozen maximum `5e-11` and RMS `5e-12` bounds.  DVEB
CPU/CUDA agreement also passes those bounds.

DVEB auto selected direct CPU lowering plus materialization and CUDA block 32
plus materialization.  Both use 65,584 bytes of scratch and two numerical
stages.  The CUDA ABI run reported no internal synchronization and ran on the
adapter's caller-owned nondefault stream.  No policy was forced.

These process durations are qualification and preparation observations only.
They are explicitly prohibited from supporting a performance claim.  The E2
randomized resident campaign remains to be run.

Evidence is retained under
`experiments/academic_u4e/evidence/u4e_e1_20260831/`.  Run
`python3 experiments/academic_u4e/verify_qualification.py` for offline
verification; DVEB, CUDA, and the temporary native artifacts are not required.
