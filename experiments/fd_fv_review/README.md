# FD/FV Phase-1 review artifacts

This directory stores the machine-readable evidence for the review frozen in
`docs/FD_FV_PHASE_1_PROTOCOL.md`. It contains metadata, classifications,
evidence locations, claim decisions, and checksums. It does not redistribute
the reviewed papers.

Run the verifier after the Phase-1 records are committed:

```bash
python experiments/fd_fv_review/verify_phase_1.py
```

The frozen outputs are in `results/phase_1_20260827/`. The human-readable
conclusions and governing rules are in `docs/FD_FV_PHASE_1_RESULTS.md` and
`docs/FD_FV_EXPERIMENTAL_CONSTITUTION.md`. Phase 1 contains no finite-volume
implementation or benchmark result.
