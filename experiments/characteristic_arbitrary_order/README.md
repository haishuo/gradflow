# Characteristic arbitrary-order qualification

This directory contains the recorder for the frozen contract in
`docs/CHARACTERISTIC_ARBITRARY_ORDER_PROTOCOL.md`.

Run it from the repository root with a CUDA-enabled PyTorch environment:

```bash
TORCHINDUCTOR_CACHE_DIR=/tmp/gradflow-characteristic-qualification-cache \
python experiments/characteristic_arbitrary_order/qualify.py \
  --output experiments/characteristic_arbitrary_order/results/qualification.json
```

The recorder refuses an existing output. It evaluates WENO-5 lineage
preservation, the 3-D Euler entropy wave, uniform-state preservation,
conservation, CPU/CUDA agreement, fixed-step autograd, and fixed-shape
full-graph compilation. It records no performance timings.

The committed record was generated from clean GradFlow commit
`2fe0b47a5c0d173a8a11ab83ed780a74e26628f2`. Its interpretation is in
`docs/CHARACTERISTIC_ARBITRARY_ORDER_RESULTS.md`.
