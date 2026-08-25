# DVEB trunk-001 screened PyTorch baseline

`weno5_torch.py` is an exact copy of
`/mnt/projects/dveb/comparator/weno5_torch.py` from DVEB commit
`ece17b3610f929207f87a2d35aa0ac893daf9dc4`, copied on 2026-08-25.
Its SHA-256 is
`2cff04949eb4c56ada030975ce5b0ce641abf702fcfc886cda0897578aff23ed`.
The file is evidence, not the GradFlow public API, and must remain unchanged.

The adjacent correctness and environment records are also exact copies:

- `CORRECTNESS.md`: `5355c66cf136f30da44ca394398936dbbf20826136c9933e60c4f3928e74827d`
- `environment.lock`: `491df1b43f312a2a86c0554210aff83a1d3734af8b21ca27afed0c6d76277a3c`
- `artifacts/inspection.json`: `c4fc3dc955636a0c6484787e2a78f60247663144edb854e94ab07cc4cecc6547`

The screen established that the direct roll/elementwise formulation for
right-moving linear advection (`a = alpha = 1`) was correct across its
declared checks and compiled as one graph with zero breaks. TorchInductor
generated backend kernels from ordinary PyTorch; the comparator contains no
handwritten Triton, CUDA, or custom operator.

## Scope limitation discovered during refoundation

The baseline's `gm = gp - df` equals the negative of Gottlieb's `dfm`, but
the subsequent negative-family call negates it again. This sign is invisible
in the screened case because the negative split is identically zero when
`a = alpha > 0`. Direct comparison on 2026-08-25 found machine-scale agreement
for `a=1`, but large disagreement for `a=-1` and for a nonlinear Burgers flux.
The canonical GradFlow implementation therefore keeps the direct structure
and corrects the negative split using Gottlieb's algebra. This baseline is not
silently generalized beyond the workload that was actually screened.

The preserved `inspection.json` predates the documented CUDA-graphs loop
remedy and reports failures for two graph modes; `CORRECTNESS.md` records the
later corrected protocol. Neither file is edited to reconcile the chronology.
