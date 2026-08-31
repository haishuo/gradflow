# Academic U4-F

U4-F is governed by `docs/ACADEMIC_U4F_PROTOCOL.md`.

It prospectively tests whether the PyTorch/TorchInductor backend recovers
resident forward competitiveness against the automatically scheduled DVEB
backend as independent `N=8192` WENO-JS5 lines are batched. The frozen batch
surface is `B = 1, 4, 16, 64, 256, 1024` on one-thread CPU and CUDA.

The campaign is complete. DVEB won resident CUDA at batches 1 and 4, batch 16
was unresolved, and PyTorch/TorchInductor won at batches 64, 256, and 1024.
PyTorch's CPU compiler admitted batch one but raised a retained internal
Inductor assertion at every larger batch. See `docs/ACADEMIC_U4F_RESULTS.md`
and `evidence/u4f_20260831/`.
