# Academic U4-D DVEB three-way results

Status: **complete at the sole U4-C correctness-admitted size**.

Date: 2026-08-30 (UTC)

U4-D adds the pinned DVEB compiler as an internal implementation control to
the exact scalar float64 WENO-JS5 contract used for the U4-C OpenSBLI external
comparison. D1 admitted all six CPU/CUDA implementation lanes before D2/D3
timing began. `N=8192` is the only reported size because the frozen U4-C bounds
excluded OpenSBLI at all larger sizes before timing.

## Resident operator time

Each cell below is the median of six independent worker medians. Every worker
retained 20 observations after five warmups; the three implementations were
randomized within each block. CUDA temperatures remained between 42 and 46 C.

| device | DVEB (ms) | OpenSBLI (ms) | GradFlow (ms) | resolved winner |
|---|---:|---:|---:|---|
| one-thread CPU | `0.19225875` | `0.07879925` | `0.09558975` | OpenSBLI |
| CUDA | `0.01638400` | `0.00985600` | `0.03257600` | OpenSBLI |

All pairwise decisions satisfied the frozen 5% effect and bootstrap-interval
rules. On CPU, OpenSBLI was about `2.44x` faster than DVEB and `1.21x` faster
than GradFlow; GradFlow was about `2.01x` faster than DVEB. On CUDA, OpenSBLI
was about `1.66x` faster than DVEB and `3.29x` faster than GradFlow. DVEB was
about `1.99x` faster than GradFlow.

## Pageable transfer-inclusive CUDA

This endpoint includes upload from the same pageable host array, one RHS, full
download, and synchronization. It has only one fresh worker per lane, so these
are descriptive distributions and **not** resolved statistical wins.

| lane | median (ms) | minimum (ms) | maximum (ms) |
|---|---:|---:|---:|
| DVEB | `0.0387650` | `0.0313290` | `0.0398100` |
| OpenSBLI | `0.0416995` | `0.0407890` | `0.0539500` |
| GradFlow | `0.0686295` | `0.0650290` | `0.0876190` |

The descriptive median ratios were `DVEB/OpenSBLI=0.92963`,
`DVEB/GradFlow=0.56484`, and `OpenSBLI/GradFlow=0.60760`. All three full
returned arrays passed the frozen correctness and conservation gates.

## Preparation and prepared launch

Observed one-off preparation costs were retained separately. The DVEB compiler
and its normal native build took `1.657` seconds, followed by `0.259` seconds
to compile and link the research adapter. OpenSBLI symbolic generation took
`0.647` seconds, instrumentation `0.017` seconds, CPU translation/build
`0.739` seconds, and CUDA build `1.644` seconds. GradFlow's observed JIT first
calls were `5.100` seconds on CPU and `1.452` seconds on CUDA. Its AOT builder
process took `7.562` seconds; the packaged artifact passed the oracle gate.
These are individual observations, not timing distributions, and their build
pipelines are not identical.

Prepared launch-to-answer starts before fresh process creation and ends after
a finite host checksum of the full RHS. Prior builds are excluded.

| prepared artifact | median of three launches (s) |
|---|---:|
| DVEB native executable | `0.196248` |
| OpenSBLI native executable | `0.213175` |
| GradFlow AOTInductor package | `1.405700` |

The descriptive DVEB/OpenSBLI median ratio was `0.92060`. DVEB took about
`0.13961` of GradFlow AOT's launch time (about `7.16x` faster), while OpenSBLI
took `0.15165` (about `6.59x` faster). Three observations per artifact do not
support a resolved winner claim.

## Interpretation

DVEB now genuinely participates in the matched bakeoff, and the result locates
a defensible niche rather than declaring a universal winner:

- OpenSBLI/OPS delivered the best resident execution on both devices for this
  operator and size. DVEB has not matched that external generated-code ceiling.
- DVEB nonetheless occupied the intended middle on CUDA: its generated kernel
  was about twice as fast as ordinary compiled PyTorch while remaining within
  a factor of `1.66` of OpenSBLI.
- DVEB was descriptively fastest once pageable round-trip transfer was included
  and in prepared native launch-to-answer. This suggests lower framework and
  deployment overhead can compensate for a slower resident kernel at this
  small grid, but the limited endpoint replication forbids a statistical win.
- DVEB's CPU result is presently poor: both OpenSBLI and compiled GradFlow beat
  it decisively. The current compiler is therefore not yet a universal
  CPU/CUDA catch-all.
- The reversal between resident and end-to-end rankings demonstrates why
  GradFlow's backend policy must be based on measured endpoints, not on
  implementation labels or device intuition.

This does not establish results for larger grids, 3-D Euler, arbitrary order,
mixed precision, differentiation, or another machine. DVEB is an internal
compiler control; OpenSBLI remains the independent external baseline.

## Evidence

Frozen evidence is in
`experiments/academic_u4d/evidence/u4d_campaign_20260830/`. It retains all 720
resident samples, all 60 transfer-inclusive samples, all nine launch records,
raw outputs, randomized orders, telemetry, full endpoint arrays, build/AOT
records, artifact hashes, commands, and a SHA-256 manifest. The architecture-
specific AOT package is retained outside the repository at the recorded path.
Run `python experiments/academic_u4d/verify_campaign.py` for offline evidence
verification; the external binaries and package are not required.
