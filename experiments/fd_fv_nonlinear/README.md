# FD/FV nonlinear Phase 5

This directory begins the nonlinear scalar boundary with a pure-standard-
library pre-shock Burgers oracle. Phase 5A contains specifications and frozen
oracle data only. It does not contain the production PyTorch Burgers solver or
performance measurements.

From the repository root:

```bash
python experiments/fd_fv_nonlinear/verify_phase_5a.py
```

The generator refuses to overwrite the committed record. Use an explicit new
output directory when testing deterministic regeneration.

Phase 5B adds the correctness-only production qualification runner. It requires
a clean committed source revision and, on Forge, an explicitly device-visible
CUDA process:

```bash
python experiments/fd_fv_nonlinear/qualify_phase_5b.py \
  --output-dir experiments/fd_fv_nonlinear/results/phase_5b_20260828
```

The runner performs no timing and refuses an existing output directory.

Verify the committed qualification without rerunning numerical work:

```bash
python experiments/fd_fv_nonlinear/verify_phase_5b.py
```

Phase 5C's isolated performance harness is governed by the prospectively frozen
`docs/FD_FV_PHASE_5C_PROTOCOL.md`. It requires a clean committed source and a
device-visible Forge process; do not run it from the default device-isolated
sandbox.

The immutable first campaign exceeded its original conservation bound after
hundreds of time steps. Phase 5CR therefore freezes a timing-free mechanistic
resolution in `docs/FD_FV_PHASE_5CR_PROTOCOL.md`. Its resolver verifies the
original record, diagnoses semidiscrete, one-step, and accumulated drift, and
may reclassify copies of the old timing records without changing any measured
sample:

```bash
python experiments/fd_fv_nonlinear/resolve_phase5c.py \
  --output-dir experiments/fd_fv_nonlinear/results/phase_5cr_20260828
```

Verify the committed resolution without collecting timing or requiring CUDA:

```bash
python experiments/fd_fv_nonlinear/verify_phase5cr.py
```

The final bounded interpretation is in
`docs/FD_FV_PHASE_5C_RESULTS.md`. The immutable initial failure remains in
`docs/FD_FV_PHASE_5C_INITIAL_RESULTS.md`.
