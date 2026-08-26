# Final DVEB WENO requalification

Status: **FROZEN BEFORE CALIBRATION OR COUNTED TIMING**.

Date: 2026-08-26

## Question and scope

This bounded campaign asks two questions:

1. Does the final, reproducible DVEB Shu Euler artifact preserve the correctness
   and near-native performance previously observed from its uncommitted
   predecessor?
2. Can a placement model calibrated only at declared training sizes choose a
   competitive CPU schedule or CUDA target at disjoint, held-out WENO sizes?

The answer applies only to the matched three-dimensional Shu Euler JS-WENO-5
workload, this machine, the tested step counts, and the recorded endpoints. It
does not qualify DVEB's generic automatic selector. DVEB commit `2f1f3ab`
records that generic selector as **NO-GO**, and no outcome here overrides that
result. GradFlow treats DVEB as read-only.

## Frozen implementations and mathematics

The candidate artifact is the clean, committed DVEB output at commit
`2f1f3ab`:

```text
executable  2b087aed48f999ae2ab0e81dd5fb48a40c289d3ef5292f7dceb491f330a970f8
program     c6e5bd916f951ff412eac99863a74f8c98e5e14b044097a7ad59fe26f704c381
module      555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260
```

The separately authored matched native ceiling is frozen independently:

```text
873a9227196664398012e7d42a27e29ec9cd3610c45a4c61ab40a0688aed3caa
```

The complete workload remains the formulation frozen in
`DVEB_BAKEOFF_PROTOCOL.md`: float32 compressible Euler, duplicated periodic
endpoints on `[0,10]^3`, an isentropic vortex extruded in z,
dimension-by-dimension characteristic Jiang--Shu WENO-5, Shu's
central-flux-plus-nonlinear-correction algebra, `epsilon=1e-6`, per-line and
per-family global LF speeds enlarged by ten percent, CFL 0.1 recomputed each
step, and three complete SSP-RK3 stages per step. Complete final states must be
finite and materialized in pageable host memory.

Before use, GradFlow copies both native executables outside the active
repository, records their source paths and hashes, and thereafter refuses a
changed copy. Calibration and results record the GradFlow and DVEB commits.

## Correctness gate

Before calibration or timing, forced DVEB CPU and CUDA and the independent
native ceiling are compared against the GradFlow PyTorch CPU authority using
complete arrays at:

- `N=6`, one and ten steps;
- `N=32`, one step; and
- `N=128`, one step.

Every lane must be finite and remain within the existing float32 maximum-error
bound of `2e-5`. CPU/CUDA and generated/ceiling discrepancies are reported.

## WENO-specific calibration

Placement preparation is excluded deployment work, measured and retained.
It uses only these training sizes:

```text
N = {7, 12, 24, 40, 56, 72}
steps = {1, 10}
```

At each point and endpoint, the candidates are initially:

```text
cpu_simd[1], cpu_simd[2], cpu_simd[4], cpu_simd[6], cpu_simd[12], cuda
```

Each receives one warmup and seven measured observations in randomized order.
The model records both resident execution and CPU-originating (`cpu-resident`)
latency; `cpu-resident` is the primary selection endpoint.

Before the model is frozen, the same conservative DVEB dominance screen may
remove a CPU schedule only if another CPU schedule is at least two percent
faster at every calibration point for both endpoints and at least five percent
faster somewhere. CUDA is never removed by this screen. Exclusions, all raw
samples, elapsed calibration work, retained candidates, and the final model
hash are recorded. No held-out measurement may influence the model.

## Held-out selector gate

The following sizes are disjoint from calibration and bracket the expected
CPU/CUDA crossover:

```text
N = {8, 16, 32, 48, 64}
steps = {1, 10}
```

At each of the ten points, every retained forced candidate and automatic
placement receive one uncounted warmup and 30 counted fresh-process runs.
Lane order is randomized within each repetition using a frozen seed. An
external monotonic clock includes native loading, initialization, selection,
the complete solve, terminal host materialization, validation, and teardown.
The executable's internally reported CPU-originating and resident intervals
are retained as diagnostics.

The automatic lane passes this workload-specific gate only if:

- all 30 runs at every point select one stable target and complete correctly;
- median regret relative to the fastest retained candidate is at most 1.10;
- at least 80 percent of points have regret at most 1.15;
- no point has regret above 1.30 unless the absolute loss is below 0.25 ms;
- when a CPU schedule is selected, its median is within 1.10 of the best
  retained CPU schedule; and
- no placement decision depends on a held-out result.

`regret = automatic median / fastest retained forced-candidate median` for the
same endpoint and point. Differences below 0.25 ms are also reported as
practical ties. Raw observations, means, medians, minima, maxima, p95 values,
absolute losses, and selected targets are retained.

## Large-grid confirmation

The selector-regret study stops at `N=64`, where alternative CPU schedules can
still be measured responsibly. At `N in {96,128}` for one and ten steps, only
automatic DVEB, forced CUDA, and the independent CUDA ceiling are repeated.
These points confirm large-grid backend behavior; they do not add selector
training data and do not count toward the selector acceptance thresholds.
Each lane receives one warmup and 30 counted fresh-process runs in randomized
blocks.

The final DVEB result may also be compared descriptively with the already
committed GradFlow bake-off. Such a cross-campaign comparison is not treated
as paired timing or as a selector acceptance result.

## Interpretation boundary

A pass would justify a narrowly scoped optional DVEB deployment backend for
this exact WENO application on this machine, with automatic placement only
inside the qualified envelope. It would not justify making DVEB the GradFlow
compiler, claiming arbitrary-workload placement, or claiming portability to
other hardware. A fail is retained and reported; GradFlow will not contort
its formulation or acceptance criteria to rescue DVEB.

No WENO-11/WENO-15 implementation, generic solver framework, new low-level
optimization, or publication performance campaign is authorized by this
protocol.

## Stop condition

Stop after the artifacts and environment are frozen, correctness passes or an
exact failure is recorded, calibration and held-out observations complete,
the workload-specific decision is documented, coherent local commits exist,
and the GradFlow working tree is clean. Do not modify DVEB and do not push this
branch without new explicit authorization.
