# Academic A1 close-prior-art comparison

Status: **frozen from the Phase-C review dated 2026-08-27**.

This table narrows GradFlow's empirical question. It is not proof of novelty;
an external numerical-CFD/prior-art audit remains an A4 release gate.

| System | Formulation and orders | Generation and execution | Autodiff | Consequence for GradFlow |
| --- | --- | --- | --- | --- |
| OpenSBLI | Structured, curvilinear high-order FD; arbitrary odd-order WENO-JS; characteristic compressible CFD | Python/SymPy generates OPS C/C++ for OpenMP, MPI, CUDA, OpenCL, OpenACC, and multi-GPU | No documented AD | Closest generation and FD-CFD precedent. GradFlow cannot claim first arbitrary-order generated FD-WENO; direct ordinary-PyTorch execution and its measured compiler/numerical behavior remain the distinction. |
| PyWENO / PyClaw | General reconstruction generation; PyClaw reported odd orders 5--17 in FV/wave-propagation solvers | Python/SymPy emits low-level C/C++/OpenCL/CUDA/Fortran components; PyClaw used wrapped Fortran | No | Establishes symbolic arbitrary-order WENO and generated low-level kernels well before GradFlow. |
| HOPE | Arbitrary-order 2-D finite-volume shallow water; order 11 demonstrated | PyTorch convolution and Einstein summation on CPU/NVIDIA GPU | Yes | Direct precedent for arbitrary-order differentiable PyTorch WENO, but not Jiang--Shu finite-difference flux reconstruction or characteristic Euler. |
| JAX-Fluids | Primarily finite-volume compressible CFD; hard-coded WENO families through order 9 plus a separate flux-splitting path | Ordinary JAX/XLA across CPU, GPU, TPU, and multiple accelerators | Yes | Strong differentiable-array and production-CFD precedent. FD and FV must not be conflated, and GradFlow cannot claim first differentiable accelerator WENO. |
| JAX-Shock | Compressible shock-capturing with fixed WENO-5; exact FD/FV classification unresolved in Phase C | JAX GPU solver | Yes; inverse parameter inference reported | Direct precedent for WENO-based inverse use. GradFlow's A3 must be framed as independent validation and utility, not first use. |

## Frozen comparison boundary

The defensible candidate contribution is the integrated characterization of
one exact-generated Jiang--Shu FD-WENO implementation written as maintainable
ordinary PyTorch across orders 5--15: exactness, conditioning, differentiation,
compiler behavior, memory, deployment endpoints, and matched baselines.

The literature record does **not** presently support calling that combination
novel. A2 and A3 must first produce results, and A4 must obtain an external
audit. The full evidence fields, unknowns, URLs, and repository revisions are
preserved in `experiments/literature_review/results/phase_c_20260827/`.
