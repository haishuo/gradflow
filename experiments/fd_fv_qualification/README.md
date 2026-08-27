# FD/FV Phase-3 qualification

This directory qualifies the canonical scalar periodic FV-WENO-JS5 PyTorch
seed against the independent Phase-2 contract. It records correctness,
convergence, conservation, differentiation, device, compiler, and transfer
evidence only.

The frozen protocol is `docs/FD_FV_PHASE_3_PROTOCOL.md`. Phase 3 contains no
performance benchmark.

The immutable first run is
`results/phase_3_20260827/qualification.json`. It passed nine gate areas and
failed the frozen smooth-spatial and profiler-event gates. Run
`verify_phase_3.py` to verify its source identities, decisions, and checksum.
The result is intentionally preserved as a failed qualification; see
`docs/FD_FV_PHASE_3_RESULTS.md` before interpreting either failure.

The separately frozen Phase-3R resolution is stored in
`results/phase_3r_20260827/resolution.json`. It preserves the failed first run,
separates noncritical fifth-order behavior from critical-point
characterization, and tests actual movement events. Verify it with
`verify_phase_3r.py`; interpret it through
`docs/FD_FV_PHASE_3_RESOLUTION_RESULTS.md`.
