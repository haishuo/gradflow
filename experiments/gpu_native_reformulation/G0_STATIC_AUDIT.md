# G0 static audit: existing 3-D Shu WENO-5 GPU graphs

Status: **complete as a source audit; GPU profiling remains pending**.

Date: 2026-08-29

## Scope

This audit compares the computation graphs visible in:

- GradFlow's canonical ordinary-PyTorch characteristic Euler path;
- the separately authored DVEB native CUDA ceiling; and
- the frozen DVEB E4 result record.

It does not time a new candidate.  CUDA was not visible to the task process,
so kernel counters, achieved bandwidth, instruction mix, and cache behavior
remain unmeasured.

## Native CUDA specimen

The current CUDA specimen allocates:

- four complete component-major state-sized arrays: `q`, `q1`, `q2`, `r`;
- three line-speed arrays, one per direction; and
- one scalar device timestep.

For every complete SSP-RK3 step it launches:

1. one CFL scan kernel;
2. one single-thread CFL finalization kernel;
3. three line-speed kernels plus one RHS kernel and one update kernel for each
   of three RK stages.

This is **17 kernel launches per step**.

### CFL reduction

The CFL scan performs one global `atomicMax` for every interior cell.  It does
not first reduce within a block.  A block reduction followed by one atomic per
block is an obvious exact-contract schedule candidate.

### Line-speed reduction

Each of the three directional speed kernels:

- assigns one block to a line;
- rereads the conserved state;
- recomputes density, velocities, pressure, and sound speed; and
- reduces the three wave-family speeds in shared memory.

The RHS kernel subsequently recomputes the same primitive quantities while
forming fluxes and Roe data.

### RHS ownership and duplicate work

One RHS thread owns one output cell.  It calls the directional derivative in
all three axes.  Each directional derivative computes the numerical flux at
both adjacent faces.  Therefore one cell evaluates **six complete numerical
faces per stage**.

Every interior periodic face is adjacent to two cells, so its numerical flux
is evaluated twice.  Each face evaluation:

- loads six five-component states;
- forms six physical fluxes;
- recomputes primitive quantities for the two Roe states;
- constructs 25-entry left and right Roe matrices;
- projects four positive and four negative split differences for each of five
  characteristic families;
- evaluates ten nonlinear corrections; and
- projects five characteristic corrections back to five physical fluxes.

The repeated face work avoids a global face-flux buffer and its write/read
traffic.  The proposed face-once schedule trades that saved arithmetic for
explicit communication or shared-memory reuse; static counting alone cannot
decide which is faster.

The compiled RHS is recorded as using **188 registers per thread**, 1,024
bytes of static shared memory, and no local-memory spill.  This is a successful
but low-occupancy-prone register strategy whose achieved occupancy and issue
efficiency require profiling.

### Stage update

After every RHS kernel, a distinct update kernel reads the base state, current
stage, and derivative, then writes the next state.  The last directional pass
could instead consume the partial RHS and perform the stage update, removing
one global derivative/update round trip if the directional schedule makes that
possible.

## Ordinary-PyTorch graph

The canonical PyTorch source is mathematically readable and compiler-friendly,
but its logical graph is whole-array oriented:

- endpoint synchronization is expressed through repeated `cat` operations;
- each direction is moved into a line-major view;
- physical flux, line speeds, and full left/right Roe matrices are expressed
  as whole-grid tensors;
- every required stencil offset constructs projected state and flux tensors;
- candidate values, indicators, denominators, ratios, nonlinear weights, and
  reconstructed characteristic fluxes are whole-grid logical tensors; and
- the three directional arrays are accumulated into a full RHS before the RK
  stage update.

At WENO-5 in 3-D, each left or right Roe tensor logically contains 25 values
per face.  Each projected offset contains five characteristic values per face.
TorchInductor can and does fuse many of these expressions, so logical tensor
existence is **not proof of a global-memory allocation**.  Generated kernel
inspection and profiling are required to determine what is actually retained,
split, or recomputed.

The step-level source also requests duplicated-endpoint synchronization before
the step, around every stage, at the final result, and again inside each RHS
entry.  A full graph compiler may remove some redundant work, but the semantic
graph exposes seven synchronization calls per step, each spanning all three
axes.

## Static opportunity ranking

The following ranking is a hypothesis to test, not a performance conclusion:

1. **Face ownership and reuse.** Potentially halves the dominant characteristic
   face algebra, at the price of communicating five flux values per face.
2. **Warp-distributed face algebra.** Reduces the 188-register single-thread
   working set by distributing families/components across lanes.
3. **Fuse line metrics with pencil loading.** Removes a duplicate state and
   primitive pass while retaining exact line-wise LF semantics.
4. **Fuse directional accumulation with RK update.** Removes the full `r`
   buffer or at least its last read/write cycle.
5. **Hierarchical CFL reduction.** Replaces one global atomic per cell with one
   per block.
6. **Unique periodic storage.** Removes duplicated planes and repeated endpoint
   synchronization; its relative benefit shrinks as `N` grows.
7. **CUDA-graph or persistent launch scheduling.** Most relevant at small and
   repeated-step endpoints after the dominant RHS is improved.

The first three may interact strongly.  Face-once global storage can lose even
when face-once shared-pencil execution wins, and a warp-distributed face may
change the preferred block/pencil geometry.

## G0 decision

There is sufficient untested schedule headroom to continue to G1.  The current
9.638 ms result is a qualified native reference, not a demonstrated maximum.
Following the exploration's declared discovery order, G1 constructs the full
unsafe U0 frontier first.  Exact Shu face-once versus cell-recompute scheduling
is retained as a later causal control and admission test; U0 cannot inherit its
correctness status.

## Identities

| Item | SHA-256 |
| --- | --- |
| DVEB ceiling `cuda.cu` | `c3964d31399bb4d2b68bdd2c33a70aa5263ea3b370a3d94e2dde2f169dfcfb6d` |
| DVEB ceiling `shu_math.h` | `125dd8ec0d60cc4c965e1a8f804b12ae471cf73850e3484520cc400ae0db9009` |
| DVEB portable pipeline CUDA runtime | `f339f75c807ac3932ea080949a8bf3b6ef5ff33a4a4b7c7ebf3708934262106c` |
| GradFlow E4 result document | `0eae9e1c020170584eeb030b0b9ab44c7e3ffb66eedb906c3d068595bda2e04f` |
