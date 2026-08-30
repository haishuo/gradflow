# Ordinary-PyTorch face-ownership screen results

Date: 2026-08-30 (UTC)

Hardware: NVIDIA GeForce RTX 5070 Ti; PyTorch
`2.9.0.dev20250705+cu128`

## Outcome

Constructing each WENO numerical face once is a **resolved performance win at
every valid 3-D endpoint in the frozen screen**. This is true in eager and
full-graph-compiled PyTorch, at WENO-JS orders 5 and 15, and in float32 and
float64. The compiled speedup ranged from `1.16x` to `2.76x` at `96^3`; the
float32 WENO-5 scale slice resolved a win at every `N` from 16 through 128.

The optimization is therefore justified for GradFlow's ordinary-PyTorch
periodic scalar representation. It is not yet a universal rule for every
backend or boundary treatment. In particular, the earlier native-CUDA G4
implementation paid for global face arrays with roughly twice the workspace,
whereas TorchInductor's ordinary-PyTorch graph used roughly half the measured
temporary allocation. Ownership and physical storage strategy are separate
decisions.

## Primary 3-D factorial

Each timing is the median of 20 randomized complete CUDA-event pairs. Ratios
are `face_once / cell_recompute`; lower is better.

| Order | Dtype | Eager face / recompute (ms) | Eager ratio | Compiled face / recompute (ms) | Compiled ratio (95% bootstrap CI) | Compiled speedup |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 5 | float32 | 3.324 / 6.441 | 0.516 | 0.754 / 1.506 | 0.499 [0.473, 0.515] | 2.00x |
| 5 | float64 | 7.030 / 13.865 | 0.507 | 4.092 / 11.264 | 0.363 [0.362, 0.367] | 2.76x |
| 15 | float32 | 40.428 / 80.694 | 0.501 | 7.997 / 9.286 | 0.861 [0.858, 0.863] | 1.16x |
| 15 | float64 | 86.436 / 172.528 | 0.501 | 33.545 / 49.484 | 0.678 [0.677, 0.679] | 1.47x |

Eager execution exposes the expected near-twofold arithmetic effect at both
orders. Compilation preserves a strong WENO-5 effect, but absorbs more of the
duplicated high-order expression: compiled WENO-15 still benefits, though by
less. This is a compiler-observed result, not evidence that WENO-15 contains
less reusable mathematics.

## Float32 WENO-5 3-D scale slice

| N | Compiled face / recompute (ms) | Paired ratio (95% CI) | Speedup | Face / recompute peak temporary memory |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.289 / 0.374 | 0.806 [0.636, 0.876] | 1.24x | 0.93 / 1.33 MB |
| 32 | 0.367 / 0.496 | 0.823 [0.609, 0.964] | 1.22x | 5.11 / 10.62 MB |
| 64 | 0.313 / 0.517 | 0.616 [0.534, 0.772] | 1.62x | 40.89 / 84.94 MB |
| 96 | 0.754 / 1.506 | 0.499 [0.473, 0.515] | 2.00x | 138.02 / 286.65 MB |
| 128 | 1.805 / 4.303 | 0.421 [0.413, 0.432] | 2.37x | 327.16 / 679.48 MB |

The benefit is present even at the small device-resident endpoints, but grows
materially once reconstruction work dominates fixed launch overhead. At
`96^3` and `128^3`, face-once approximately halves both compiled runtime and
incremental temporary allocation. Process startup, transfers, time
integration, and compilation were deliberately outside this RHS-only clock.

## Correctness and the excluded 1-D points

All eight unique 3-D configurations passed eager representation parity,
compiled-versus-eager error, finiteness, conservation, full-graph capture, and
no-graph-break gates. Both representations use identical WENO coefficients,
flux splitting, and exact generated reconstruction.

All four frozen `N=1,048,576` 1-D configurations failed the
compiled-versus-eager gate and were not timed. A prospectively frozen,
timing-free diagnostic then showed:

- compiled and eager global LF `amax(abs(u))` agree exactly at all 24 probes;
- the discrepancy affects both ownership representations similarly;
- float32 already fails at the diagnostic's first point, `N=65,536`;
- float64 passes at `N=65,536` and fails the fixed pointwise threshold from
  `N=262,144` onward; and
- the discrepancy grows with points per derivative line, not total cell
  count: `96^3` (884,736 cells) passes while 1-D lines with fewer total cells
  can fail.

The evidence is consistent with compiled/eager floating-point reassociation
being amplified by the `1/dx` conservative difference at extreme linear
resolution. It does not identify the responsible software layer, nor show
which endpoint is closer to exact arithmetic. Consequently, this experiment
makes no 1-D performance claim and does not label the behavior a confirmed
TorchInductor defect.

## Scientific conclusion

The targeted answer is:

1. **Most beneficial in this screen:** large 3-D WENO-5, especially float64;
   compiled resident speedups reached `2.37x` (float32 `128^3`) and `2.76x`
   (float64 `96^3`).
2. **Still beneficial:** compiled WENO-15, with `1.16x` float32 and `1.47x`
   float64 speedups at `96^3`.
3. **Memory benefit in ordinary PyTorch:** approximately twofold at moderate
   and large 3-D sizes because the duplicated expression exposes more live
   compiler temporaries.
4. **Not qualified here:** extreme-resolution 1-D compiled execution, CPU,
   MPS, nonperiodic boundaries, and characteristic Euler.

This corroborates the native-CUDA G4 scheduling result without changing the
canonical package. A later implementation may confidently retain logical
single-face ownership in the ordinary-PyTorch path, while separately choosing
whether to materialize, tile, fuse, or recompute those faces in a particular
backend.

## Reproduction and integrity

Commands are recorded in `evidence/face_ownership_20260830/COMMANDS.txt`.
`SHA256SUMS` freezes both JSON records. Run:

```bash
python experiments/face_ownership_screen/verify_screen.py \
  experiments/face_ownership_screen/evidence/face_ownership_20260830
```

No `src/gradflow/` file was changed, no backend was admitted, and no result was
pushed by this experiment.
