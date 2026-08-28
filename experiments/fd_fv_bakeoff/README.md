# FD/FV Phase-4 scalar bakeoff

This directory contains the frozen Phase-4A multidimensional admission, the
Phase-4B isolated performance campaign, and the Phase-4R replication and
causal-characterization campaign. Their governing protocols are
`docs/FD_FV_PHASE_4_PROTOCOL.md` and
`docs/FD_FV_PHASE_4_REPLICATION_PROTOCOL.md`.

No timing result is eligible unless the immutable Phase-4A record passes and
verifies. CUDA absence is recorded rather than replaced with an inferred GPU
result.

Phase 4A passed and is preserved under `results/phase_4a_20260827`. The admitted
CPU Phase-4B matrix is under `results/phase_4b_20260827`; run
`verify_phase_4a.py` and `verify_phase_4b.py` to verify both result series.
Phase 4R is under `results/phase_4r_20260827`; `verify_phase_4r.py` recomputes
all statistics and the failed strong-replication decision. CUDA remained
unavailable and has no measurements in any of these records.
