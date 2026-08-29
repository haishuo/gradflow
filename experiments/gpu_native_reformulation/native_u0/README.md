# Native U0 candidate

This is the deliberately unsafe G1 candidate defined by
`docs/GPU_NATIVE_WENO_EXPLORATION.md`.

The exact counted U0 source and executable are frozen under
`evidence/g1_u0_20260829/`. The live source now also accepts compile-time
recovery levels R1--R6 through `build_recovery.sh`; those cumulative candidates
do not alter the frozen U0 identity.

It is not canonical GradFlow code and is not a qualified backend.  Its first
counted output must be frozen before comparison with the numerical oracle.

Contract:

```text
u0_unique_f32_component_shared_density_local_lf_forward_euler_v1
```

The candidate uses unique periodic nodes, float32 fast arithmetic,
componentwise split-flux WENO-5, one density-derived pair of nonlinear weights
per face shared by all five flux components, a six-point face-local LF speed,
one face evaluation per direction and cell, and a conservative Forward Euler
update.  State is resident during counted execution.

The candidate does not implement Roe characteristic projection, ancestral
line-wise characteristic LF speeds, duplicated endpoints, or SSP-RK3.  Its
latency is therefore an unsafe frontier rather than a speedup over the
qualified solver.

Build on Forge with `./build.sh`.  The binary emits one JSON record and can
write its final component-major unique-node state as raw float32:

```text
build/gradflow_u0 --size 128 --steps 1 --warmups 5 --repetitions 30 \
  --output-state u0_n128_s1.f32
```
