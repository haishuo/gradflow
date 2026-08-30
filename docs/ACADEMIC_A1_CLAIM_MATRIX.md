# Academic A1 frozen claim matrix

Status: **A1 claim boundary frozen**.

Date: 2026-08-30 (UTC)

The first GradFlow paper is an empirical systems-and-numerics study, not a new
WENO formula or general CFD product announcement. The machine-readable source
is `experiments/academic_a1/evidence/a1_20260830/consolidation.json`.

## Established statements

| ID | Statement | Paper role |
| --- | --- | --- |
| M1 | For orders 5--15, the exact generator reproduces polynomials through every declared candidate and optimal-stencil degree; exact optimal weights are positive and sum to one. | Core mathematics |
| M2 | Generated order five reproduces the known Jiang--Shu coefficients and agrees with the independently written scalar WENO-5 seed inside its frozen oracle bound. | Lineage and oracle |

“Established” here means exact arithmetic or independent-oracle evidence. It
does not establish novelty.

## Observed statements

| ID | Statement | Boundary |
| --- | --- | --- |
| O1 | Scalar periodic generated WENO-JS converges on the frozen smooth family and passes the declared conservation, differentiation, device, and full-graph compilation gates. | Orders 5--15; representative differentiation/compiler orders 5, 11, 15 |
| O2 | Classical WENO-JS loses design order on the frozen higher-order critical-point family. | The selected `sin(2*pi*x)^3` family; no WENO-Z inference |
| O3 | The same generated reconstruction passes the qualified face-frozen Roe-characteristic 3-D Euler path. | Orders 5--15 under the distinct Shu epsilon/LF/grid contract |
| O4 | A scalar binary32 indicator/weight-formation seam can preserve the frozen scalar contract and accelerate compiled execution; it does not pass the strict higher-order characteristic-Euler contract. | Tested policies, cases, tolerances, and RTX 5070 Ti only |
| O5 | Logical face-once construction reduces ordinary-PyTorch 3-D RHS time at every valid screened endpoint and reduces compiler temporary allocation at moderate/large grids. | Periodic scalar WENO-JS5/15 screen; extreme 1-D compiled cases excluded |
| O6 | The matched native-CUDA WENO-5 face-once schedule approaches a twofold large-grid resident speedup while using about twice the global workspace. | Fixed 3-D Shu Euler WENO-5 control only |
| O7 | Coefficient-basis diagnostics and exact rational complexity grow strongly with order, while sampled roundoff onset moves to coarser grids. | Frozen A1 scalar problem and machine |
| O8 | Scalar epsilon `1e-29` remains below the material-change threshold relative to `1e-40` throughout the A1 amplitude sweep; larger epsilons materially change some scale-dependent cases. | Float64, `N=128`, amplitudes `1` through `1e-6`; no new default selected |

## Inference

| ID | Statement | Why it remains inference |
| --- | --- | --- |
| I1 | Earlier roundoff onset at higher order is consistent with increasing coefficient and expression sensitivity. | A1 does not causally separate coefficient conversion, operation ordering, cancellation, or rounded input data. |

## Untested statements and remaining gates

| ID | Statement | Resolution |
| --- | --- | --- |
| U1 | The complete formulation-matched order-5--15 cold/warm/AOT/CPU/GPU performance surface is unknown. | Academic A2 |
| U2 | No independent inverse or sensitivity application yet demonstrates differentiable utility. | Academic A3 |
| U3 | MPS, orders above 15, general multidimensional boundaries, Navier--Stokes, curvilinear geometry, and production aerospace workflows remain unqualified. | Outside the first-paper scope |

## Prohibited statements

| ID | Wording that may not appear as a GradFlow conclusion | Reason |
| --- | --- | --- |
| P1 | First arbitrary-order WENO generator, first PyTorch WENO, first differentiable WENO, or first GPU WENO. | Direct and close prior art exists. |
| P2 | FD-WENO, PyTorch, GPUs, or GradFlow are universally faster or superior. | Results depend on formulation, endpoint, order, shape, precision, hardware, and residency. |
| P3 | Current GradFlow is a general, real-time, production-ready aerospace CFD solver. | The qualified surface is deliberately much narrower. |

## First-paper scope decision

The headline subject is exact-generated ordinary-PyTorch finite-difference
WENO-JS at orders 5--15. The FD/FV campaign and G0--G6 native reformulation
are supporting studies rather than independent headline claims. Native CUDA
is a fixed WENO-5 control. DVEB is an optional fixed comparator and cannot
block the paper.

The claim matrix must be updated with A2 and A3 observations before manuscript
wording is frozen, but its taxonomy and prohibited boundaries are fixed here.
