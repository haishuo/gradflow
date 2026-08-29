# GPU-native WENO reformulation experiment

This directory is reserved for the noncanonical experiment defined in
`docs/GPU_NATIVE_WENO_EXPLORATION.md`.

The experiment deliberately separates two questions:

1. What execution and numerical graph is most comfortable on the GPU?
2. Starting from that frontier, which operations must be restored to recover
   independent correctness?

No candidate in this directory is part of the `gradflow` package.  No failing
candidate may be used as a backend, and no timing may influence admission
until the corresponding mathematical contract and oracle result are recorded.

## Phase order

1. **G0 — static audit.** Count duplicated face work, whole-grid
   intermediates, launch boundaries, reductions, and retained state in the
   qualified CUDA and ordinary-PyTorch paths. **Source audit complete; GPU
   profiling pending. See `G0_STATIC_AUDIT.md`.**
2. **G1 — reckless U0 frontier.** Implement the complete, explicitly declared
   GPU-comfortable formulation without consulting the numerical oracle during
   design or tuning.
3. **G2 — frontier and damage record.** Freeze U0 timing and output, then
   compare it with the independent oracle and qualified native control.
4. **G3 — correctness recovery and schedule controls.** Execute the
   preregistered R1--R6 ladder and exact-math face-reuse controls against
   independent GradFlow/Fortran evidence.
5. **G4 — qualified comparison.** Time only candidates that pass the complete
   gate, then determine whether the formulation result generalizes beyond
   WENO-5.

CUDA is not visible to the current task process.  G0 and protocol work may be
performed locally; G1 onward require the Forge-visible CUDA execution route
used by the prior E4 campaign.

## Starting evidence

The starting native specimen remains in the DVEB project and is read-only from
this experiment:

```text
/mnt/projects/dveb/tools/shu_euler3d_ceiling/
```

Its source must not be silently copied or modified here.  A future prototype
must record authorship, source identity, build flags, generated artifacts,
hardware, and its relationship to the independent oracle.
