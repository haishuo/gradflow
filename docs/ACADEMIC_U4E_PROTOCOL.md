# Academic U4-E prospective DVEB requalification protocol

Status: **prospective protocol frozen before U4-E qualification or comparative timing**.

Date: 2026-08-31 (UTC)

## Question

After DVEB Trunk 005 introduced general, deterministic scheduling for CPU loop
lowering, CUDA launch geometry, and reuse of an adjacent pure intermediate, can
its automatically selected scalar WENO-JS5 artifact compete with the unchanged
OpenSBLI/OPS and ordinary compiled PyTorch lanes under the exact U4-D numerical
and execution contract?

U4-E is prospective.  The U4-D result and its post-hoc diagnosis motivated the
compiler work, but neither those measurements nor DVEB's internal Trunk 005
measurements count as U4-E evidence.  U4-E changes the DVEB artifact only and
reruns every external lane in randomized blocks.

## Frozen sources and artifact

- GradFlow U4-D closure commit
  `14024d98c33728d88d07441a60240ed037383801` and this protocol commit;
- DVEB Trunk 005 closure commit
  `39bd1c323daa3dbce6421a09dc34dc0cd2109d88`, tree
  `3711d334ee48f24717900456f17c6518a1f0bada`;
- DVEB handoff implementation commit
  `ba87208832a3f31a0062cbd3093f463f91cd403f`, tree
  `9d7c9bea745e1205a04871f7c6d88fad10a82e3e`;
- unchanged DVEB language source SHA-256
  `b4236d640c8429400f44792fae0198b7eed013676444660eb036d99937584ab8`;
- immutable `weno5-schedule-v1.tar.gz` SHA-256
  `2342f66416b1b120efd42e0e4ca8838f32cef4c62a13bf43042fb12ef7354ae0`;
- contained ABI-v1 library SHA-256
  `9ff9172b1ac712b8bc97ca9523fd114b2637e5d7825259371ba9850459168443`;
- the exact OpenSBLI/OPS sources, generated artifacts, and revisions qualified
  by U4-D; and
- the unchanged GradFlow implementation at this protocol commit.

DVEB is read-only.  The campaign copies the handoff bundle to a
GradFlow-controlled temporary artifact directory, verifies the bundle and
every extracted member against the handoff manifest, and never invokes or
modifies the DVEB compiler.  The retained U4-E adapter may allocate caller-
owned padded arrays, fill periodic halos, transfer data, call the public ABI,
time declared endpoints, and export arrays.  It may not contain or alter WENO
mathematics.

The public header must compile as C11 and C++17, and the library must report
`dveb_scalar_schedule_abi_version() == 1`.  The device entry point must use a
caller-owned nondefault CUDA stream.  Context creation, allocation, transfer,
halo filling, and synchronization stay outside resident timing.

## Frozen machine and environment

U4-E runs on Forge's NVIDIA GeForce RTX 5070 Ti and AMD Ryzen 9 9900X.  Exact
driver, CUDA, PyTorch, compiler, operating-system, clock, temperature, and
throttle information is recorded at execution.  CUDA must be visible and the
handoff must accept the device as native `sm_120`; otherwise the CUDA cells are
retained as unavailable rather than simulated.

CPU comparison uses exactly one thread for all three lanes, with dynamic OpenMP
teams disabled.  U4-E does not use DVEB's multithreaded CPU result to compete
against one-thread OpenSBLI or PyTorch.

## Frozen mathematics and size

U4-E inherits U4-C/U4-D unchanged:

- `u_t + u_x = 0` on unique periodic nodes `x_j=j/N`;
- finite-difference WENO-JS5 in the Gottlieb-equivalent central-flux plus
  split-difference-correction algebra;
- global Lax--Friedrichs speed `alpha=1`;
- 12-scaled smoothness indicators and epsilon `1e-29` inside squared weights;
- one semidiscrete RHS, without a time update;
- IEEE float64; and
- the exact retained `N=8192` U4-C state bytes, SHA-256
  `7def0f1a410959390af68416a01f92d0ec917a23aaf022f5b90d52c366bb5530`.

The frozen canonical RHS SHA-256 is
`d92a1dd5f20cba9533dd25682fd19ca2d39f584b883b9fee3c994f1dd46b3621`.
`N=8192` remains the only external size: U4-E does not relax U4-C's numerical
bounds after larger OpenSBLI grids were excluded.

## E1: artifact and correctness admission

Before any comparative timing:

1. verify the DVEB repository commit/tree, handoff manifest, bundle, and every
   member hash;
2. compile the public header as C11 and C++17 and load the exact library;
3. query and retain DVEB's automatic schedule without overriding it;
4. require finite full arrays of exactly 8,192 values from all six
   implementation/device lanes;
5. require conservation under
   `32 * epsilon_machine * sum(abs(rhs))`;
6. compare each lane with the canonical array under
   `maximum_normalized <= 5e-11` and `RMS_normalized <= 5e-12`; and
7. require DVEB CPU/CUDA agreement under the same limits.

For this shape, the manifest predicts automatic CPU direct-loop plus
materialization and automatic CUDA block 32 plus materialization.  The query
record, not this prediction, is authoritative; a different decision stops the
campaign for explanation rather than being silently forced.  Expected scratch
is `8 * (8192 + 6) = 65,584` bytes and materialization has two numerical
launches/stages.  The CUDA run result must report no internal synchronization.

Any failed lane remains reported and is excluded.  Resident comparison is
interpretable only if all six lanes pass.

## E2: resident three-way comparison

The automatically selected lanes are:

1. DVEB Trunk 005 through the immutable device-resident ABI v1;
2. OpenSBLI-generated OPS sequential CPU or CUDA; and
3. GradFlow `torch.compile(fullgraph=True, dynamic=False)` CPU or CUDA.

For each device:

```text
independent workers per implementation = 6
warmups per worker                     = 5
retained observations per worker       = 20
three-lane order seed                  = 20260831
bootstrap resamples                    = 20,000
thermal stop                           = 80 C
```

The parent randomizes all three implementations inside each worker block.  CPU
uses monotonic wall time.  CUDA uses events on the caller-owned nondefault
stream around the numerical schedule and its required dependency only.  State
creation, context creation, allocation, halo exchange, input transfer, output
retrieval, process startup, and synchronization used merely to read the event
are outside resident samples.  No outlier is removed.

Every lane retains raw observations, worker medians, median, mean, minimum,
maximum, quartiles, MAD, and sample standard deviation.  Pairwise ratios use
paired worker medians and 20,000 deterministic bootstrap resamples.  For ratio
`A/B`, an A win requires median below `0.95` and the upper 95% interval below
one; a B win requires median above `1.05` and the lower interval above one;
otherwise the pair is unresolved.  An overall winner must resolve faster than
both alternatives.

DVEB's query result, selected CPU loop or CUDA block, reuse policy, launch
count, scratch bytes, and synchronization field are retained for every worker.
Only `auto` participates in the external decision endpoint.  Forced policies
may be used later for diagnosis, but cannot replace an observed automatic lane
or support the U4-E winner claim.

## E3: separately reported operational endpoints

1. **Pageable transfer-inclusive CUDA:** one fresh worker per lane, five
   warmups and 20 observations, from the same pageable host state through a
   complete returned host RHS.  This is descriptive and has no statistical
   winner.
2. **Prepared fresh-process launch-to-answer:** three randomized launches per
   CUDA artifact, from parent process creation through a finite host checksum
   of the complete RHS.  This is descriptive and has no statistical winner.
3. **Preparation:** handoff copy, verification, extraction, and adapter build
   are recorded separately.  DVEB compiler-development or Trunk 005 build time
   is historical and is not charged to the prepared artifact.  OpenSBLI
   generation/build and GradFlow JIT/AOT preparation remain separately named
   observations from their actual preparation paths.

Resident, transfer-inclusive, preparation, and launch-to-answer answer
different questions and may not be collapsed into one ranking.

## Causal and interpretive limits

- The prospective U4-E three-way campaign determines the external ranking of
  the new DVEB artifact; DVEB's internal Trunk 005 timings do not.
- U4-D versus U4-E is a before/after comparison on one machine and contract.
  Because the campaigns are separate, its ratio is descriptive; Trunk 005's
  randomized forced-policy factorial supplies the direct internal causal test.
- U4-E compares one scalar, one-dimensional, float64, order-5 RHS at one size.
  It does not establish arbitrary order, systems, full solvers, gradients,
  automatic device placement, another GPU, or a universal backend winner.
- OpenSBLI remains the independently maintained external generated-code
  baseline.  DVEB remains an internal compiler under active development.
- Consumer-GPU float64 behavior is a hardware-specific observation and does
  not settle A100/H100 performance.
- No result may claim that DVEB matches hand-written CUDA generally; this
  artifact is compared only with the named implementations and contract.

## Evidence and stop condition

Retain the protocol commit; repository and artifact identities; exact commands;
bundle/member/adapter/executable hashes; full qualification and endpoint arrays;
every raw timing sample; randomized orders; schedule queries; telemetry;
exclusions; deterministic analysis; and a SHA-256 evidence manifest with an
offline verifier.

Close U4-E only after all frozen cells complete or retain explicit failures,
the verifier and relevant GradFlow regressions pass, coherent local commits
exist, and the working tree is clean.  Do not push without explicit
authorization.
