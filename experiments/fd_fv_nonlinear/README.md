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
