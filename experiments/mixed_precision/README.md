# Mixed-precision exhaustive search

This directory executes the frozen scalar Tier-1 protocol in
`docs/MIXED_PRECISION_PHASE_D_PROTOCOL.md`.

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
