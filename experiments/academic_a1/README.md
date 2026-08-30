# Academic A1 consolidation

This directory contains the bounded, CPU-only numerical-limit measurements
required by `docs/ACADEMIC_A1_PROTOCOL.md`. It does not benchmark execution or
change canonical GradFlow source.

Run the frozen measurement with:

```bash
PYTHONPATH=src python experiments/academic_a1/run_numerical_limits.py \
  --output experiments/academic_a1/evidence/a1_20260830/numerical_limits.json
```

The evidence verifier is intentionally offline and is exercised by the test
suite.
