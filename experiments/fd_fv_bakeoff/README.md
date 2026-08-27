# FD/FV Phase-4 scalar bakeoff

This directory contains the frozen Phase-4A multidimensional admission and the
Phase-4B isolated performance campaign. The governing protocol is
`docs/FD_FV_PHASE_4_PROTOCOL.md`.

No timing result is eligible unless the immutable Phase-4A record passes and
verifies. CUDA absence is recorded rather than replaced with an inferred GPU
result.

Phase 4A passed and is preserved under `results/phase_4a_20260827`. The admitted
CPU Phase-4B matrix is under `results/phase_4b_20260827`; run
`verify_phase_4a.py` and `verify_phase_4b.py` to verify both result series.
