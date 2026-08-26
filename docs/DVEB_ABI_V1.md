# DVEB portable arbitrary-state ABI v1

## Outcome

DVEB portable ABI v1 closes the benchmark-initializer gap. A compiled portable
pipeline now exports a C-linkage shared library accepting caller-owned state,
and GradFlow can use it for arbitrary-state forward execution without a
temporary file or subprocess.

The interface is versioned by both symbol name and structure handshake:

```text
dveb_portable_abi_version
dveb_portable_query_v1
dveb_portable_run_v1
```

The generated `dveb-artifact-v2` manifest records SHA-256 identities for the
compatibility executable, ABI library, public header, scientific program, and
mathematical module. Library and header paths are relative to the manifest so
the ABI sidecar set can be relocated together.

## State and execution contract

For the current Shu Euler artifact, input and output are contiguous CPU
float32 tensors in component-major `(5, nz+1, ny+1, nx+1)` order with all
periodic endpoints duplicated. Exact input/output aliasing is legal. The
caller owns both buffers.

Each call requests a positive fixed step count and one target:

- exact CPU worker count;
- CUDA, including the required H2D and D2H copies; or
- a verified bounded automatic-placement model.

The result identifies the selected target, element count, native execution
time, total ABI time including copies, peak memory, and finite-output status.
Stable status codes distinguish ABI, argument, count, placement, and execution
failures. No C++ object or exception crosses the boundary.

GradFlow additionally enforces the compiled artifact's current eligibility:
3-D Euler characteristic JS-WENO-5, cubic grids, spacing `10/N`, CFL 0.1,
duplicated periodic endpoints, float32, CPU-resident contiguous input, positive
fixed steps, and no gradient requirement. Explicit native targets refuse any
other request. Automatic selection uses direct PyTorch as the semantically
matched fallback.

## Gate on Forge

The gate ran on 2026-08-26 with a Ryzen 5 7600X, RTX 5070 Ti, CUDA 13.0, and
PyTorch 2.13.0+cu130.

- DVEB full suite with the ABI enabled: 71 passed.
- Original trunk-001 functional/refusal gate: PASS.
- Complete GradFlow suite with real CUDA and ABI: 36 passed.
- ABI-specific GradFlow gate: 7 passed, including the unchanged direct CPU and
  CUDA portable runner.
- Public header: valid C11 and C++17.
- Error status, arbitrary input consumption, exact alias, bounded automatic
  placement, and out-of-range refusal: passed.

Independent non-vortex parity results:

| Intervals | Steps | CPU ABI vs PyTorch | CUDA ABI vs PyTorch | CPU vs CUDA |
|---:|---:|---:|---:|---:|
| 6 | 1 | `4.768e-7` | `4.768e-7` | `2.384e-7` |
| 6 | 10 | `7.153e-7` | `8.345e-7` | `8.345e-7` |
| 32 | 1 | `4.768e-7` | `4.768e-7` | `4.768e-7` |

All are inside the predeclared absolute `2e-5` float32 bound. The committed
machine-readable record is
`experiments/shu_torch_ablation/results/dveb_abi_v1_gate_20260826.json`.

The same record includes one-shot overhead observations. The first CUDA ABI
call paid about 184 ms total, dominated by cold CUDA initialization and data
movement; subsequent calls in that process were much smaller. Those numbers
verify timing fields and expose startup behavior. They are not a calibrated
bakeoff, crossover result, or optimization campaign.

## Artifact identities

The local FMA build used for the gate recorded:

```text
ABI library  cfa939a5b492ed5711a432391d604ceda65ed55c6df7a4a77b6bfabdd7bd1b1c
ABI header   c14731d87423f95f9b19f216ddb7d4d2719e7196b6bd0d19205598ab23015c2a
program      c6e5bd916f951ff412eac99863a74f8c98e5e14b044097a7ad59fe26f704c381
math module  555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260
```

The rebuilt compatibility executable differs bytewise from the frozen
requalification executable only in an NVCC-generated temporary local symbol
name; its GNU build ID and executable behavior are unchanged. This known NVCC
build nondeterminism is why every generated artifact is hash-pinned rather
than assumed reproducible byte-for-byte.

## Limits and next decision

ABI v1 is synchronous, CPU-memory-only, float32, fixed-step, and not qualified
for concurrent calls. It does not return accumulated simulated time and does
not implement backward differentiation. GradFlow therefore records native
`simulated_time` as unavailable and keeps gradient requests on PyTorch.

The prior WENO placement model was calibrated through a fresh-process
executable, not the loaded-library endpoint. It may test bounded selection
mechanics, but a defensible in-process automatic performance policy needs a
separate frozen calibration protocol. No WENO-11 or WENO-15 implementation and
no new broad performance campaign began in this work.
