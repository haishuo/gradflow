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
