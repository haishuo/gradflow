# DVEB portable device-resident ABI v2

## Outcome

DVEB device ABI v2 closes the public resident-state gap left by portable ABI
v1. A generated shared library now exposes a reusable opaque CUDA context and
accepts caller-owned CUDA buffers directly:

```text
dveb_portable_device_abi_version
dveb_portable_device_create_v2
dveb_portable_device_run_v2
dveb_portable_device_destroy_v2
```

ABI v1 remains available and unchanged at its public header/symbol/behavior
boundary. It is still the CPU-resident input/output interface for explicit CPU,
CUDA-with-transfers, and bounded placement calls. The two interfaces are not
silently substituted for one another.

## Contract

For the current Shu Euler artifact, `create_v2` binds one cubic interval count
and one CUDA device. It allocates the full reusable native workspace once and
returns its exact element and byte counts. The context is not safe for
concurrent calls.

`run_v2` accepts contiguous component-major float32 buffers in
`(5, N+1, N+1, N+1)` order. Input and output belong to the caller. They may be
disjoint or exactly aliased. The call performs the complete global-CFL plus
requested SSP-RK3 steps on an optional caller stream, synchronizes that stream,
and returns wall and CUDA-event timing metadata. It performs no host-to-device
or device-to-host transfer and no device allocation.

GradFlow exposes this boundary as:

```python
artifact = gradflow.DvebArtifact.from_manifest("artifact.json")
with gradflow.DvebDeviceContext(artifact, intervals=128) as context:
    result = context.run(cuda_state, steps=10)
```

The adapter uses the current PyTorch stream and returns a CUDA tensor. An
optional caller-owned `out=` tensor avoids output allocation in the public
Python adapter. CPU, wrong-device, noncontiguous, non-float32, wrong-shape, and
autograd inputs are refused rather than converted.

## Correctness gate

On the Ryzen 5 7600X / RTX 5070 Ti host with CUDA 13.0 and PyTorch
2.13.0+cu130, full-array comparisons at `(N,steps)={(6,1),(6,10),(32,1)}`
had worst absolute error `1.430511475e-06` against the frozen `2e-5` bound.
A non-default PyTorch stream and exact input/output alias both passed. DVEB's
own full suite passed 77 tests with one optional selector-model skip, and its
functional/refusal gate passed.

## E4 requalification

The protocol was committed before timing. At each of ten points, each of four
resident lanes ran in six independent randomized workers with five warmups and
five counted calls. All 1,200 calls succeeded. DVEB won every point and all 60
blocks, with medians 2.53--7.36 times faster than packaged AOTInductor.

At `N=128`, one step, the public wall median was 9.638 ms and the internal
total was 9.626 ms. At ten steps they were 94.910 ms and 94.892 ms. Thus the
versioned, caller-owned ABI reached the native generated-CUDA floor without a
material large-grid boundary penalty.

The complete report is
`experiments/shu_torch_ablation/DVEB_DEVICE_ABI_E4_RESULTS.md`; hashed raw
records are under `results/dveb_device_e4_20260827/`. The immutable sidecar is
outside the repository at
`/mnt/artifacts/gradflow/dveb_device_e4_20260827/manifest.json` with SHA-256
`5404f5eb668b2be78b5aa8f0be82efba52556324c6bdfda35ef536db21cd8b77`.

## Artifact identities

The frozen E4 timing sidecar used these identities:

```text
device ABI library  4541677eb21c6d93a7f0c6694ff78006c707b1f6b79c5752c7b497a841ff199c
device ABI header   ad920101e3aa7ed4a41bf8ac86625e7c9149c58cd5ee218beb607750181ee2a4
v1 ABI library      fb41b855e31e2ca2a8a989798be838b20d8220848d92189afcf5d94dc18f6663
v1 ABI header       c14731d87423f95f9b19f216ddb7d4d2719e7196b6bd0d19205598ab23015c2a
program             c6e5bd916f951ff412eac99863a74f8c98e5e14b044097a7ad59fe26f704c381
math module         555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260
```

The combined v1-then-v2 GradFlow suite subsequently exposed a pre-existing v1
odd-step teardown bug: the final stage pointer replaced an allocation handle,
causing a duplicate free and leaving an error in the caller's CUDA runtime.
DVEB commit `1e7fec3` fixed ownership and added a regression assertion. It did
not change the v2 device numerical path. The official rebuilt v1/v2 library
hashes are respectively
`216961117e81990d47e3b1efa01d72d86e3d1c104f422eaec44e4dc277e6bf90`
and
`fdd56a78be2c9396c77eb945fb40f5a8eab0875474b1efe8b34fe9c9ec99e5e5`.
Post-fix N=128 sentinel medians were 9.612 ms for one step and 94.573 ms for
ten steps, within 0.4% of the frozen campaign medians.

## Limits

This is a synchronous, forward-only, float32 interface for one compiled 3-D
Euler characteristic JS-WENO-5 formulation. It does not implement gradients,
arbitrary order, noncubic grids, general boundary conditions, accumulated
simulation time, asynchronous completion, or concurrent use of one context.
It has not been evaluated on another GPU. It is not yet wired into GradFlow's
automatic backend selector; users must select it explicitly.
