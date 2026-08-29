# G3 recovery evidence

This directory contains the exact cumulative recovery sources, build recipes,
executables, compiler logs, raw `N=32` states, 30-sample timing records, and
float64 oracle comparisons for R1 through R6.

The source sequence is:

- `recovery.cu`: R1/R2 source;
- `recovery_r3.cu` through `recovery_r6.cu`: source frozen at each later step;
- matching `build_recovery*.sh` snapshots and `r*_compiler.log` files.

The implementation was written in GradFlow after inspecting the qualified DVEB
ceiling's mathematical source. It is not emitted by DVEB and does not claim
independent authorship of the underlying Shu algebra. The inspected source
identities remain recorded in `docs/GPU_NATIVE_WENO_EXPLORATION.md`.

Timing JSON reports CUDA-event numerical-loop latency. Raw `.f32` files are
five component-major fields on a unique periodic `32^3` grid. Damage JSON was
computed by the committed `compare_recovery_candidate.py` using the qualified
GradFlow float64 oracle.

Verify this directory with:

```bash
sha256sum --check SHA256SUMS
```

The scientific interpretation and claim boundary are in
`experiments/gpu_native_reformulation/G3_RECOVERY_RESULTS.md`.
