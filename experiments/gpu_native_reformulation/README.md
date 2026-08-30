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
   design or tuning. **Complete and frozen.**
3. **G2 — frontier and damage record.** Freeze U0 timing and output, then
   compare it with the independent oracle and qualified native control.
   **Initial damage comparison complete.**
4. **G3 — correctness recovery and schedule controls.** Execute the
   preregistered R1--R6 ladder and exact-math face-reuse controls against
   independent GradFlow/Fortran evidence. **Complete. R6Q strongly
   corroborated the face-once forward calculation but did not pass every
   frozen qualification criterion and is not backend-admitted. See
   `G3_RECOVERY_RESULTS.md` and `G3_QUALIFICATION_RESULTS.md`.**
5. **G4 — qualified comparison.** Time only candidates that pass the complete
   gate, then determine whether the formulation result generalizes beyond
   WENO-5. Because R6Q did not pass every G3 criterion, G4 was explicitly
   narrowed to a **non-admission schedule-control performance study**.
   **Complete: the face-once schedule produced a sustained approximately 2x
   resident speedup at moderate and large grids, with approximately 2x peak
   workspace. See `G4_PERFORMANCE_RESULTS.md`.**
6. **G5 — shared-pencil memory recovery.** Replace the global face arrays with
   one-block-per-line shared-memory faces and a fused final directional update.
   **Complete: P1 was bitwise identical to R6Q and reduced declared peak memory
   by 37.43% at `128^3`, but took 2.62--2.70x R6Q time at the primary points
   and lost to cell-recompute throughout resident timing. The frozen candidate
   therefore failed its speed-memory Pareto gate. See
   `G5_SHARED_PENCIL_RESULTS.md`.**
7. **G6 — exact-math occupancy ablation.** Cross face block sizes 64/128/256
   with uncapped/112/96 register policies while holding R6Q mathematics and
   storage fixed. **Complete: all variants were bitwise identical, but no
   candidate met the two-point improvement rule. A 96-register cap raised
   theoretical occupancy only for 64/128-thread blocks, introduced spills,
   and lost at moderate/large scale; 112 registers retained 33.33% occupancy
   and performance parity. See `G6_OCCUPANCY_RESULTS.md`.**

CUDA is visible on Forge through the explicitly authorized native execution
route. G1 was measured on the local NVIDIA GeForce RTX 5070 Ti.

## Starting evidence

The starting native specimen remains in the DVEB project and is read-only from
this experiment:

```text
/mnt/projects/dveb/tools/shu_euler3d_ceiling/
```

Its source must not be silently copied or modified here.  A future prototype
must record authorship, source identity, build flags, generated artifacts,
hardware, and its relationship to the independent oracle.
