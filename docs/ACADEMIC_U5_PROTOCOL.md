# Academic U5 stable-PyTorch replication protocol

Status: **frozen prospectively before any U5 numerical or timing execution**.

Date: 2026-08-31 (UTC)

## Question

Do the paper's central numerical, compiler, performance, deployment, and
differentiation observations reproduce on a stable PyTorch release, and does
the U4-F batched CPU compiler failure persist there?

This is a reproduction gate. It is not an optimization campaign and it does
not replace the dated development-build evidence when results disagree.

## Stable release selection

U5 uses PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, Python `3.12`, and Triton
`3.7.1`. PyTorch 2.13 was the current stable release when this protocol was
frozen; the official release announcement is
<https://pytorch.org/blog/pytorch-2-13-release-blog/>. The installed wheel
reports PyTorch commit `cf30153c4c131c8164ee7798e5022d810682e2cb` and includes
native `sm_120` code generation for Forge's RTX 5070 Ti.

The release was selected by date and support status, not by observed GradFlow
performance. The complete package inventory, executable identity, PyTorch
configuration, driver, and hardware record must be captured with the evidence.

## Frozen replication surface

No mathematical source, tolerance, warmup count, repetition count, compiler
option, or tested shape may be changed for U5. The following existing surfaces
are rerun with the stable interpreter:

1. **A1 numerical limits:** the complete order 5--15 coefficient, roundoff,
   epsilon, finiteness, and conservation characterization.
2. **A2 performance:** all 46 registered configurations and 90 eligible
   CPU/CUDA workers, including every original correctness exclusion, graph
   record, warm eager/compiled timing, CPU thread observation, transfer slice,
   and memory record.
3. **A2 deployment:** the order 5/11/15 fixed-shape AOT packages and both the
   prepared-cache and isolated-empty-cache launch-to-answer slices.
4. **A3 differentiation:** the complete analytic-observation inverse problem,
   finite-difference gradient sweep, derivative-free control, resolution
   study, CPU/CUDA eager/compiled correctness, graph behavior, and timing.
5. **U4-F regime map:** the same `N=8192`, batches
   `1,4,16,64,256,1024`, DVEB artifact, correctness bounds, worker structure,
   and resident timing protocol. In particular, CPU batches above one are
   attempted without a fallback so the previous Inductor scheduler assertion
   can either reproduce or disappear honestly.

The frozen development-build records remain the comparison authority. U5 may
describe agreement, disagreement, or newly exposed failures; it may not erase
or silently reinterpret the prior observations.

## Execution discipline

- Each existing harness is invoked unchanged through the selected stable
  interpreter.
- TorchInductor caches are isolated as required by the original protocols.
- CUDA timings remain device-resident CUDA-event measurements where originally
  specified; transfer-inclusive and launch-to-answer endpoints remain
  separately labelled.
- Every performance lane is gated by its original correctness and graph rules.
- Compiler failure, timeout, OOM, nonfinite output, conservation failure, or
  tolerance failure is retained as a result. No eager fallback, alternate
  tensor representation, tolerance relaxation, or source edit is permitted.
- Thermal stops and randomized lane order remain those of the source protocol.
- AOT build time remains reported but outside prepared-package invocation time.
- DVEB is rerun only as the frozen U4-F control; no DVEB development or tuning
  is part of U5.

## Comparison rules

U5 reports exact environment changes and, for matching admitted cells:

- numerical admission changes;
- graph and graph-break changes;
- stable/development median timing ratios with the original dispersion;
- AOT build, warm, and launch-to-answer changes;
- A3 recovered parameter, derivative agreement, and execution-cost changes;
- U4-F winner/unresolved decisions under the existing five-percent and paired
  bootstrap rule; and
- whether the batched CPU compiler assertion is fixed, persists, or changes
  failure mode.

Timing differences are descriptive across two sequential software
environments on one machine. They are not paired samples and are not attributed
causally to one compiler change without further evidence.

## Stop condition

U5 closes when every registered stage has either completed or produced a
retained failure, the evidence verifies offline, the stable/development
comparison is documented, the full repository regression is rerun in the
project environment, and the working tree is clean. Only then may the paper
replace or qualify its nightly-based claims.

