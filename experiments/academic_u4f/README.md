# Academic U4-F

U4-F is governed by `docs/ACADEMIC_U4F_PROTOCOL.md`.

It prospectively tests whether the PyTorch/TorchInductor backend recovers
resident forward competitiveness against the automatically scheduled DVEB
backend as independent `N=8192` WENO-JS5 lines are batched. The frozen batch
surface is `B = 1, 4, 16, 64, 256, 1024` on one-thread CPU and CUDA.

No U4-F qualification or timing existed when the protocol was frozen.
