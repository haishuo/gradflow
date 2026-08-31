# Academic U4-E prospective DVEB requalification results

Status: **complete at the sole U4-C correctness-admitted size**.

Date: 2026-08-31 (UTC)

U4-E prospectively replaced only U4-D's original DVEB artifact with the
immutable Trunk 005 scheduling handoff.  All six lanes passed E1 before timing.
The external comparison used DVEB's automatic policies without overrides:
direct/materialized on one-thread CPU and block-32/materialized on CUDA.

Terminology clarification: the frozen machine-readable key `gradflow` denotes
the repository's PyTorch/TorchInductor implementation. It is a legacy evidence
alias, not the GradFlow system as a whole. See `BACKEND_IDENTITY.md`.

## Resident operator time

Each value is the median of six independent worker medians.  Every worker
retained 20 observations after five warmups, and implementation order was
randomized inside each block.

| device | DVEB (ms) | OpenSBLI (ms) | PyTorch/TorchInductor (ms) | resolved winner |
|---|---:|---:|---:|---|
| one-thread CPU | `0.05995225` | `0.07954950` | `0.09555675` | DVEB |
| CUDA | `0.00928000` | `0.01015200` | `0.03295200` | DVEB |

On CPU, the paired worker-median DVEB/OpenSBLI ratio was `0.74072` with a
bootstrap 95% interval `[0.72199, 0.76050]`; DVEB/PyTorch was `0.62535` with
interval `[0.60417, 0.62998]`.  DVEB was therefore about `1.35x` faster than
OpenSBLI and `1.60x` faster than PyTorch/TorchInductor.

On CUDA, DVEB/OpenSBLI was `0.91417` with interval `[0.89688, 0.93104]`, and
DVEB/PyTorch was `0.27919` with interval `[0.27339, 0.28863]`.  DVEB was about
`1.09x` faster than OpenSBLI and `3.58x` faster than ordinary compiled
PyTorch.  All four DVEB pairwise decisions satisfy the frozen 5% effect and
bootstrap-interval rules.  CUDA temperatures stayed between 43 and 47 C, with
no active throttle reason in any retained pre/post observation.

## Before/after stability

The U4-D and U4-E campaigns are separate, so these ratios are descriptive,
not a paired causal estimate.  DVEB's internal Trunk 005 factorial is the
direct scheduling ablation.

| device/lane | U4-D (ms) | U4-E (ms) | U4-E/U4-D | descriptive speedup |
|---|---:|---:|---:|---:|
| CPU DVEB | `0.19225875` | `0.05995225` | `0.31183` | `3.21x` |
| CPU OpenSBLI | `0.07879925` | `0.07954950` | `1.00952` | `0.99x` |
| CPU PyTorch/TorchInductor | `0.09558975` | `0.09555675` | `0.99965` | `1.00x` |
| CUDA DVEB | `0.01638400` | `0.00928000` | `0.56641` | `1.77x` |
| CUDA OpenSBLI | `0.00985600` | `0.01015200` | `1.03003` | `0.97x` |
| CUDA PyTorch/TorchInductor | `0.03257600` | `0.03295200` | `1.01154` | `0.99x` |

The two unchanged competitors reproduced within about 3% of U4-D while the
DVEB lane changed substantially.  This is consistent with the independently
qualified compiler scheduling changes rather than a machine-wide speed shift,
while still respecting the cross-campaign limitation.

## Separately reported operational endpoints

Pageable transfer-inclusive CUDA has one worker per lane and is descriptive:

| lane | median (ms) | minimum (ms) | maximum (ms) |
|---|---:|---:|---:|
| DVEB | `0.026475` | `0.026140` | `0.027479` |
| OpenSBLI | `0.042580` | `0.041440` | `0.054089` |
| PyTorch/TorchInductor | `0.067080` | `0.063530` | `0.087540` |

The descriptive median ratios are `DVEB/OpenSBLI=0.62177`,
`DVEB/PyTorch=0.39468`, and `OpenSBLI/PyTorch=0.63476`.

Prepared launch-to-answer has three randomized launches per artifact and is
also descriptive:

| prepared CUDA artifact | median (s) |
|---|---:|
| DVEB ABI adapter | `0.195773` |
| OpenSBLI/OPS executable | `0.211554` |
| PyTorch AOTInductor package | `1.393930` |

The descriptive DVEB/OpenSBLI ratio is `0.92540`.  DVEB took about `0.14045`
of PyTorch AOTInductor's launch time. Neither operational endpoint supports a
statistical winner claim.

Handoff copy, verification/extraction, C11/C++17 header checks, and ABI adapter
build were recorded separately.  The adapter build took `0.281` seconds.
PyTorch/TorchInductor's observed qualification first calls were `4.974`
seconds on CPU and `1.451` seconds on CUDA. These preparation observations are
not resident samples and their pipelines are not equivalent.

## Interpretation

For this narrow contract, DVEB has now demonstrated its intended value: a
general compiler scheduling change moved generated scientific code from behind
both competitors to a resolved resident win, without WENO-name special cases,
runtime candidate racing, a forced research schedule, or hand-written CUDA in
the GradFlow integration.  The artifact also preserves a small C ABI and
caller-owned asynchronous CUDA stream rather than requiring GradFlow to know
compiler internals.

PyTorch was a performance hypothesis, not GradFlow's identity. For this
specific regime, the hypothesis that clever ordinary-PyTorch compilation
could recover enough of its abstraction and runtime weight to be the fastest
admitted path is rejected: it lost to both generated native implementations
on CPU and CUDA. Because the resident CUDA endpoint excluded compilation and
transfer, that result is not explained by Python startup alone; the generated
device schedule also matters. The prepared-launch endpoint separately exposes
substantial Python/framework startup weight.

PyTorch may still redeem the performance hypothesis in a larger, higher-order,
multidimensional, batched, differentiated, or more deeply fused regime where
fixed overhead is amortized and fusion has more work to combine. That is now a
prospective hypothesis requiring measurement, not a reason to disregard this
result. The WENO-JS5 mathematics in this comparison is unchanged by which
backend won.

This does **not** establish a universal DVEB advantage.  The result is one
one-dimensional scalar float64 WENO-JS5 RHS at `N=8192` on one consumer GPU and
one CPU.  It does not cover larger external grids, 3-D Euler, arbitrary order,
multiple CPU threads, gradients, full solvers, automatic device placement,
datacenter GPU float64, or another machine.  OpenSBLI remains the independent
external generated-code baseline; DVEB remains an internal compiler under
development.

## Evidence

Frozen evidence is under
`experiments/academic_u4e/evidence/u4e_campaign_20260831/`.  It retains all 720
resident samples, all 60 transfer-inclusive samples, all nine launch records,
full endpoint arrays, randomized orders, automatic schedule metadata,
telemetry, hashes, commands, and deterministic analysis.  Run
`python3 experiments/academic_u4e/verify_campaign.py` for offline verification;
external binaries, CUDA, and the AOT package are not required.

## Closure

After the evidence and verifier commit, the complete CUDA-visible GradFlow
regression passed with `355 passed`, `12 skipped`, and one PyTorch deprecation
warning in 53.84 seconds.  The skips are the existing opt-in portable/device
DVEB ABI tests that require separately supplied external manifests or
executables; U4-E instead verifies its immutable handoff, ABI adapter, schedule
metadata, full arrays, and hashes directly.
