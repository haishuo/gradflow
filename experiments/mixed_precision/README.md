# Mixed-precision exhaustive search

This directory first executed the frozen scalar Tier-1a protocol in
`docs/MIXED_PRECISION_PHASE_D_PROTOCOL.md`. The current runner executes the
evidence-driven Tier-1b refinement in
`docs/MIXED_PRECISION_PHASE_D_TIER1B_PROTOCOL.md`, which separates nonlinear
weight formation from weight normalization.

The protocol is intentionally committed before the implementation and result
record. No result from this directory qualifies characteristic Euler or a
production precision default.

Run the complete frozen matrix from the repository root:

```bash
PYTHONPATH=src python experiments/mixed_precision/search.py \
  --output experiments/mixed_precision/results/phase_d_tier1_YYYYMMDD
python experiments/mixed_precision/verify.py \
  experiments/mixed_precision/results/phase_d_tier1_YYYYMMDD
```

`search.py` also accepts restricted `--orders` and `--masks` for development,
but the verifier deliberately refuses such partial records.

The CUDA performance follow-up uses `benchmark.py`, which isolates every
order/policy pair in a fresh process and cache as required by
`docs/MIXED_PRECISION_PHASE_D_PERFORMANCE_PROTOCOL.md`.

`euler_qualify.py` carries only the scalar-qualified policies into the frozen
Tier-2 characteristic-Euler correctness gate. Its shock cases reuse the
immutable Phase-A independent oracles and Phase-B float64 terminal arrays.
The frozen protocol and completed result are documented in
`docs/MIXED_PRECISION_PHASE_D_TIER2_PROTOCOL.md` and
`docs/MIXED_PRECISION_PHASE_D_TIER2_RESULTS.md`. Verify the complete result
with:

```bash
python experiments/mixed_precision/verify_euler.py \
  experiments/mixed_precision/results/phase_d_tier2_20260827
```
