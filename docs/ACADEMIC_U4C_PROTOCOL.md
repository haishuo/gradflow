# Academic U4-C external CUDA and performance protocol

Status: **frozen before OpenSBLI CUDA execution or comparative timing**.

Date: 2026-08-30 (UTC)

## Question

For the scalar finite-difference WENO-JS5 operator qualified in U4-B, does
OpenSBLI's independently generated OPS implementation agree across CPU and
CUDA, and what execution and deployment cost does ordinary compiled PyTorch
impose relative to that external implementation on Forge?

U4-C does not compare general CFD applications, finite volume against finite
difference, Euler systems, higher WENO orders, or different epsilon policies.

## Frozen sources and machine

- GradFlow branch state at protocol commit;
- OpenSBLI commit `e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d`,
  tree `0ff053443f6b243b2bd42475f98122306151427d`;
- OPS commit `c0af0f124469e5fd856b594a23ff1206c3e9c7a8`,
  tree `82c3fd0c0b4724c6e8474e16f730e7560845235f`;
- the exact U4-B adapter and generalization patch;
- NVIDIA GeForce RTX 5070 Ti, driver `580.173.02`;
- CUDA toolkit `13.0.88` from `/home/haishuo/cuda-13.0`;
- native target `sm_120`; and
- sequential OPS for the CPU external lane and CUDA OPS for its GPU lane.

The older system CUDA 12.0 compiler is not used because it predates native
`sm_120` code generation. This is a toolchain compatibility choice, not an
optimization variant.

## Frozen mathematics

- `u_t + u_x = 0` on unique periodic nodes `x_j=j/N`;
- finite-difference WENO-JS5;
- physical flux `f(u)=u`;
- LF speed `alpha=1`;
- nonlinear power two;
- GradFlow 12-scaled epsilon `1e-29`, represented by OpenSBLI's standard
  epsilon `1e-29/12`;
- one semidiscrete RHS, with no time update; and
- IEEE float64 only.

The performance state is fixed as

```text
u_j = 0.4 + sin(2*pi*37*j/N) + 0.1*cos(2*pi*91*j/N).
```

Its integer modes are exactly periodic. It exercises nonlinear weights more
strongly than a single low-frequency wave while remaining deterministic and
smooth. No state may be changed after observing timing or parity.

## C1: CUDA correctness admission

Before timing, build the U4-B application with the OPS CUDA backend and native
`sm_120` code. For the U4-B `N=64` states `u_a`, `u_b`, and constant `0.37`:

- require finite CPU and CUDA arrays with identical shape and ordering;
- require OpenSBLI CUDA versus OpenSBLI sequential CPU at `rtol=0`,
  `atol=2e-12`;
- require OpenSBLI CUDA versus canonical GradFlow at `rtol=0`,
  `atol=2e-12`;
- require the constant residual maximum to be at most `2e-12`; and
- apply the U4-B conservation bound independently to every array.

At every performance size, admit each implementation/device lane only if its
output is finite and conservative and agrees with the canonical GradFlow CPU
float64 array under both normalized bounds:

```text
max(abs(candidate-reference)) / max(max(abs(reference)), 1) <= 5e-11
RMS(candidate-reference)      / max(max(abs(reference)), 1) <= 5e-12.
```

CUDA comparison must retrieve the full array after synchronization. A failed
size/lane is retained as `correctness_excluded` and is not timed.

## C2: warm operator comparison

The prospectively frozen sizes are:

```text
N = 8,192; 131,072; 1,048,576; 8,388,608.
```

The lanes are:

1. OpenSBLI generated OPS sequential CPU;
2. GradFlow `torch.compile(fullgraph=True, dynamic=False)` CPU with one intra-
   op and one inter-op thread;
3. OpenSBLI generated OPS CUDA; and
4. GradFlow `torch.compile(fullgraph=True, dynamic=False)` CUDA.

The primary comparisons are within device: OpenSBLI/GradFlow CPU and
OpenSBLI/GradFlow CUDA. Cross-device ratios are secondary and do not identify
a universally best framework. Eager PyTorch is not repeated because U4-C asks
about the strongest already-qualified ordinary-PyTorch representation, and A2
already characterized eager versus compiled execution.

For each admitted lane and size:

```text
independent workers per implementation = 6
warmups per worker                     = 5
retained observations per worker       = 20
pair-order random seed                 = 20260830
bootstrap resamples                    = 20,000
thermal stop                           = 80 C
```

The parent randomizes which implementation worker runs first within each
same-device block. CPU samples use a monotonic wall clock. CUDA samples use
CUDA events around the already-resident reconstruction plus divergence and
synchronize before reading elapsed time. Initialization, allocation, output
retrieval, compilation, and process startup are outside warm samples. No
outlier is removed.

Retain every observation, worker medians, aggregate median, mean, minimum,
maximum, quartiles, MAD, sample standard deviation, paired worker-median
ratios, and a deterministic bootstrap 95% interval. Define the ratio as
`OpenSBLI time / GradFlow time`; values above one favor GradFlow. A resolved
GradFlow win requires median ratio above `1.05` and bootstrap lower bound above
one. A resolved OpenSBLI win requires median ratio below `0.95` and bootstrap
upper bound below one. Otherwise the point is unresolved.

## C3: transfer, preparation, and launch endpoints

These endpoints are reported separately from C2 and never substituted for it.

### Transfer inclusive

At every admitted CUDA size, retain 20 observations after five warmups from a
pageable CPU state through host-to-device transfer, one RHS, device-to-host
full-array return, and synchronization. OpenSBLI uses its public OPS data-set
set/fetch interfaces; GradFlow uses explicit tensor transfers. Input creation
is outside the clock. If a backend cannot express this endpoint without
changing its numerical machinery, record `not_implemented`.

### Preparation cost

Record once, without treating it as a stable timing distribution:

- OpenSBLI symbolic generation;
- OPS translation;
- OPS sequential and CUDA compilation;
- GradFlow CPU and CUDA first-call compilation; and
- any AOTInductor package build attempted for the launch endpoint.

### Prepared process launch to answer

At the smallest and largest correctness-admitted sizes, run three fresh
processes per available prepared artifact. Parent wall time begins before
process creation and ends after a finite CPU checksum of the full RHS is
received. It includes runtime/library startup, state construction, allocation,
one RHS, required transfer, checksum, and teardown; it excludes prior artifact
build.

The OpenSBLI artifact is its already-built executable. The GradFlow artifact
must be an AOTInductor package for the fixed shape. A warm JIT cache is not an
AOT artifact and may not fill a missing package endpoint. If AOT packaging is
unsupported, record `not_implemented` and leave that launch comparison open.

## Memory and environment

Record peak CUDA memory where each runtime exposes it without profiling the
timed region, host peak RSS for each worker, exact compiler commands, package
versions, CPU/GPU/driver identity, CUDA target, and generated-source hashes.
An OOM, compiler failure, unsupported architecture, or thermal stop is a
retained result, not permission to shrink a size.

## Interpretation rules

- U4-C can characterize this one operator on this one machine only.
- OpenSBLI is an adapted external baseline; stock example applications remain
  unmatched.
- Build cost, warm resident execution, transfer-inclusive execution, and
  launch-to-answer answer different questions and must not be collapsed.
- Consumer-GPU float64 limitations remain part of the hardware context.
- No result establishes superiority for Euler, three-dimensional CFD,
  arbitrary order, A100/H100 hardware, or complete application solves.

## Evidence and stop condition

Retain the protocol commit, complete harness, adapter/instrumentation hashes,
raw arrays used for admission, raw timing samples, build logs, commands,
machine-readable analysis, SHA-256 manifest, and offline semantic verifier.

U4-C closes only when C1 has a recorded decision and all admitted C2/C3 cells
have completed or have an explicit retained failure. Create coherent local
commits and leave the working tree clean. Do not push without explicit
authorization.
