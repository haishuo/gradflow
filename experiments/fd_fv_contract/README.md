# FD/FV mathematical-contract artifacts

This directory holds the independent standard-library derivation and frozen
oracle records for FD/FV Phase 2. It is not canonical solver code and must not
be imported by `src/gradflow`.

The governing preimplementation protocol is
`docs/FD_FV_PHASE_2_PROTOCOL.md`. Generate a candidate record in a temporary
location with `derive_phase_2.py`, and verify the committed record with:

```bash
python3 experiments/fd_fv_contract/verify_phase_2.py
```

Phase 2 contains no performance results.

The frozen result is documented in `docs/FD_FV_PHASE_2_RESULTS.md`; committed
machine-readable records live in `results/phase_2_20260827/`.
