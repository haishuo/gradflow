# FD/FV Euler Phase 6

Phase 6A freezes independent point-value and cell-average Euler projections
before a production finite-volume Euler implementation exists. It reuses the
hash-identified exact Sod and high-resolution Shu--Osher authorities from the
earlier boundary/shock study.

The generator requires a clean committed tree and refuses an existing output
directory:

```bash
PYTHONPATH=. python experiments/fd_fv_euler/freeze_phase6a.py \
  --output experiments/fd_fv_euler/results/phase_6a_20260828
```

Phase 6A performs no GradFlow production solve and collects no performance
timing.

Verify the immutable committed record without running a production solver:

```bash
PYTHONPATH=. python experiments/fd_fv_euler/verify_phase6a.py
```

Phase 6B implements only the prospectively registered matched FV Euler JS5
formulation and applies the frozen correctness matrix to it and the existing
FD formulation. The qualification requires a clean committed tree and visible,
freshly admitted Forge CUDA:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/qualify_phase6b.py \
  --output experiments/fd_fv_euler/results/phase_6b_20260828
```

The preserved record contains untimed scalar metrics and all raw numerical
arrays needed for independent recomputation. Verify it without rerunning a
production solve:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6b.py
```

Phase 6B passed. See `docs/FD_FV_PHASE_6B_RESULTS.md`. It collected no
performance measurements and does not authorize an FD/FV speed conclusion.

Phase 6C prospectively freezes and executes the matched Euler performance
matrix. Reproduce the canonical campaign only on the admitted Forge hardware:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/run_phase6c.py \
  --output-dir experiments/fd_fv_euler/results/phase_6c_20260829
```

Verify its raw records and derived decisions without rerunning timing:

```bash
PYTHONPATH=src:. python experiments/fd_fv_euler/verify_phase6c.py
```

See `docs/FD_FV_PHASE_6C_RESULTS.md` for the bounded interpretation. The
result does not authorize a universal FD/FV or CPU/GPU claim.
