# Arbitrary-order WENO-JS scalar qualification

This directory contains the reproducible qualification recorder for the
frozen contract in `docs/ARBITRARY_ORDER_WENO_JS_PROTOCOL.md`.

Run it from the repository root with a CUDA-enabled PyTorch environment:

```bash
TORCHINDUCTOR_CACHE_DIR=/tmp/gradflow-weno-js-qualification-cache \
python experiments/weno_js_arbitrary_order/qualify.py \
  --output experiments/weno_js_arbitrary_order/results/qualification.json
```

The recorder refuses an existing output. It constructs exact data, evaluates
the frozen smooth and critical-point families, checks conservation,
CPU/CUDA agreement, gradcheck, and fixed-shape full-graph compilation, and
records source and environment identities. It intentionally records no timing.

The completed interpretation is in
`docs/ARBITRARY_ORDER_WENO_JS_RESULTS.md`. The committed result was generated
from GradFlow commit `479f44b3b0495f5272ed1bad3bf84152897b3b7e`.
