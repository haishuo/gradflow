# Academic U4-D three-way DVEB protocol

Status: **frozen before U4-D DVEB qualification or comparative timing**.

Date: 2026-08-30 (UTC)

## Question

At the sole U4-C correctness-admitted scalar grid, can DVEB-generated native
CPU/CUDA code compete with the matched OpenSBLI/OPS and ordinary compiled or
AOT PyTorch implementations under identical mathematics, input bytes, and
execution endpoints?

U4-D is an internal compiler-control extension to U4-C. DVEB is not an
independently maintained external baseline and does not replace OpenSBLI's
role in the manuscript.

## Frozen sources and machine

- GradFlow U4-C closure commit `530b972` and this protocol commit;
- DVEB commit `bd4bc791b6e8f4a2ba2b0b28ecdb3086a4d3d97c`, tree
  `ca0f146b1951e8f02b79c5a7dd37d1dba3bbc44d`;
- DVEB scalar source `examples/weno5/weno5.dveb`, SHA-256
  `b4236d640c8429400f44792fae0198b7eed013676444660eb036d99937584ab8`;
- the OpenSBLI and OPS revisions already pinned by U4-C;
- NVIDIA GeForce RTX 5070 Ti, driver `580.173.02`;
- CUDA 13.0.88 and native `sm_120` for DVEB and OpenSBLI;
- the existing PyTorch 2.9.0 development build with CUDA 12.8; and
- one CPU thread for every CPU lane.

The DVEB repository is read-only for U4-D. The harness makes a detached local
copy at the pinned commit, invokes the real compiler with
`DVEB_CONTRACT=fma`, and hashes its generated sources. The retained adapter may
inject state, call generated launchers, fill periodic halos through the DVEB
runtime, time endpoints, and export arrays. It may not edit or reproduce the
generated WENO mathematics.

## Frozen mathematics and size

U4-D inherits the U4-C scalar contract unchanged:

- `u_t + u_x = 0` on unique periodic nodes `x_j=j/N`;
- finite-difference WENO-JS5 in the Gottlieb-equivalent central-flux plus
  split-difference-correction algebra;
- global LF speed `alpha=1`;
- 12-scaled smoothness indicators and epsilon `1e-29` inside the square;
- one semidiscrete RHS, no time update;
- IEEE float64; and
- the exact retained U4-C `N=8192` state byte array.

`N=8192` is the only U4-C size admitted to external timing. U4-D does not
retroactively relax the larger-grid tolerances or time a three-way comparison
where OpenSBLI was excluded.

## D1: correctness admission

Before any U4-D timing, retrieve the full DVEB CPU and CUDA arrays. Require:

- finite arrays with exactly 8,192 values;
- conservation under the U4-C roundoff-scaled bound;
- DVEB CPU and CUDA versus the frozen U4-C canonical array under
  `maximum_normalized <= 5e-11` and `RMS_normalized <= 5e-12`; and
- DVEB CPU versus CUDA under the same bounds.

Requalify OpenSBLI and GradFlow from the same input during the campaign. A
failed lane is retained and excluded. No performance sample may be interpreted
unless all six device/implementation lanes pass.

## D2: resident three-way comparison

The lanes are:

1. DVEB-generated CPU or CUDA;
2. OpenSBLI-generated OPS sequential CPU or CUDA; and
3. GradFlow `torch.compile(fullgraph=True, dynamic=False)` CPU or CUDA.

For each device:

```text
independent workers per implementation = 6
warmups per worker                     = 5
retained observations per worker       = 20
three-lane order seed                  = 20260830
bootstrap resamples                    = 20,000
thermal stop                           = 80 C
```

The parent randomizes all three implementations within each worker block.
CPU uses monotonic wall time. CUDA uses events around the resident WENO
reconstruction plus divergence. State creation, allocation, input transfer,
output retrieval, compilation, process startup, and explicit native halo
exchange are outside resident samples. No outlier is removed.

For every lane retain all observations, worker medians, median, mean, minimum,
maximum, quartiles, MAD, and sample standard deviation. Analyze all three
pairwise worker-median ratios with deterministic bootstrap 95% intervals. For
ratio `A/B`, an A win requires median below `0.95` and upper interval below
one; a B win requires median above `1.05` and lower interval above one;
otherwise the pair is unresolved. An overall winner must resolve faster than
both alternatives.

## D3: transfer, preparation, and launch

### Pageable transfer inclusive

On CUDA, run one fresh process per lane in randomized order. Each retains 20
observations after five warmups from the same pageable CPU state through H2D,
one RHS, full D2H return, and synchronization. Input construction is outside
the clock. Report distributions and descriptive median ratios; do not promote
the one-worker endpoint to a resolved statistical winner.

### Preparation

Record once:

- DVEB parsing/code generation, its standard native build, and U4-D adapter
  build;
- OpenSBLI symbolic generation, OPS translation, and native build;
- GradFlow JIT first calls; and
- GradFlow AOT export/package build.

These are observations, not stable timing distributions.

### Prepared fresh-process launch to answer

Run three fresh processes per CUDA artifact. Parent wall time begins before
process creation and ends after a finite CPU checksum of the full RHS is
received. Prior builds are excluded. The lanes are the DVEB-generated native
adapter executable, the OpenSBLI-generated native adapter executable, and the
qualified fixed-shape GradFlow AOTInductor package.

## Interpretation limits

- U4-D compares one 1-D scalar float64 order-5 operator at one grid size.
- It does not supersede the earlier 3-D float32 Shu Euler DVEB result.
- It does not qualify DVEB automatic placement, a public scalar ABI,
  arbitrary WENO order, differentiation, full solvers, or other hardware.
- The DVEB adapter exercises genuine compiler-generated launchers but is a
  research adapter, not proof of a finished DVEB user experience.
- Resident, transfer-inclusive, preparation, and launch results answer
  different questions and may not be collapsed.

## Evidence and stop condition

Retain source commits and hashes, generated-source hashes, exact input and
output arrays, build commands/logs, every raw sample, analysis, environment,
SHA-256 manifest, and offline verifier. Close U4-D only after all frozen cells
complete or retain explicit failure, coherent local commits exist, the full
GradFlow regression passes, and the worktree is clean. Do not push without
explicit authorization.
