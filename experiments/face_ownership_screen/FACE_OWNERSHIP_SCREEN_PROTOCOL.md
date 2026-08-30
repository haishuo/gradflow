# Ordinary-PyTorch face-ownership screening protocol

Status: **frozen before implementation, compilation, or timing**.

Date: 2026-08-30 (UTC)

## Question

For exact-generated scalar finite-difference WENO-JS in ordinary PyTorch,
when does constructing each numerical interface flux once and reusing it in
both adjacent cell differences outperform explicitly reconstructing both
adjacent faces for every output cell?

This is a targeted representation screen motivated by the closed G0--G6
native-CUDA study. It is not G7, does not modify canonical GradFlow source,
does not admit a backend, and does not extend the native CUDA implementation.

## Fixed mathematics

Both representations use exactly the existing `WENOJS` generated algebra:

- periodic unique scalar state;
- multidimensional inviscid Burgers flux `f(u)=u^2/2`;
- one global Lax--Friedrichs speed `max(abs(u))` per directional RHS;
- native state precision throughout;
- WENO-JS orders 5 or 15;
- equal spacing `2*pi/N` in every direction; and
- a semidiscrete conservative RHS summed dimension by dimension.

The input is one deterministic, smooth, sign-changing periodic field. There
is no time integration, boundary closure, limiter, mixed precision, custom
operator, hand-written CUDA, or Triton source.

## Representations

`face_once` is the canonical logical form: reconstruct `F[i+1/2]` once, then
form `(F[i-1/2] - F[i+1/2]) / dx` by shifting the reconstructed face tensor.

`cell_recompute` uses the same split fluxes and generated reconstruction, but
independently reconstructs the current face and a one-cell-shifted copy for
the previous face. Translation equivalence makes the exact-arithmetic result
the same while exposing duplicated nonlinear reconstruction work to eager
PyTorch and TorchInductor.

No artificial graph barrier, custom kernel, cloning, materialization request,
or compiler-disabling token may be inserted to force either implementation to
win. Compiler common-subexpression elimination is part of the observed
ordinary-PyTorch endpoint.

## Frozen screen

The primary equal-workload factorial is:

```text
device       = CUDA
orders       = {5, 15}
dtype        = {float32, float64}
dimensions   = {1, 3}
large shapes = {1-D: 1,048,576; 3-D: 96^3}
modes        = {eager, torch.compile(fullgraph=True)}
```

The scale slice is compiled and eager float32 WENO-5 in 3-D at:

```text
N = {16, 32, 64, 96, 128}
```

The `96^3` scale point is the same execution as its factorial point and is
recorded once. This produces twelve unique mathematical configurations and
24 representation/mode comparisons.

The screen intentionally excludes CPU, MPS, characteristic Euler, boundaries,
and additional orders. Positive results identify where a later academic
matrix should include face ownership; they do not generalize to those omitted
domains. Float64 results are explicitly bounded to the consumer RTX 5070 Ti's
FP64 capability.

## Correctness gate

Before timing each configuration:

1. evaluate both eager representations on identical input bytes;
2. compare eager `face_once` with eager `cell_recompute`;
3. compile both with `torch.compile(fullgraph=True)` and require one captured
   graph with no graph break;
4. compare each compiled output with its own eager output;
5. require finite outputs and a conservative global RHS sum; and
6. audit the experimental source for host transfers inside the numerical
   functions.

Thresholds are fixed as normalized maximum/RMS error:

| Comparison | float32 max / RMS | float64 max / RMS |
| --- | ---: | ---: |
| eager representation parity | `2e-5 / 2e-6` | `2e-12 / 2e-13` |
| compiled versus own eager | `5e-5 / 5e-6` | `5e-11 / 5e-12` |

Normalization uses `max(max(abs(reference)), 1)`. The conservation bound is
`32 * eps(dtype) * sum(abs(rhs))`. A failed or out-of-memory configuration is
recorded and excluded from speed interpretation; no threshold or shape may be
changed after observation.

## Timing and memory

For every passing endpoint:

```text
warmups                         = 5 per representation
randomized complete pair blocks = 20
random seed                     = 20260830
bootstrap resamples             = 20,000
thermal stop                    = 80 C
```

CUDA events measure device-resident RHS execution. Input construction,
compilation, transfers, output inspection, synchronization for reporting, and
process startup are outside this clock. Wall-clock first-call compilation is
recorded separately. No timed observation includes a host/device transfer.

Each block randomizes `face_once` and `cell_recompute`. No outlier is removed.
Report raw event samples, order, temperature, clocks, compile time, graph
counts, pointwise errors, and incremental peak allocated CUDA memory for one
post-warmup call.

Define the paired ratio as `face_once / cell_recompute`. A resolved face-once
win requires median ratio below `0.95` and bootstrap 95% upper bound below
`1.0`. A resolved recompute win reverses those inequalities above `1.05` and
`1.0`. Everything else is unresolved. The study reports all points rather
than selecting a universal representation.

## Interpretation boundary

This screen can establish whether duplicated face reconstruction remains
visible and costly in ordinary PyTorch on one GPU, and whether its value
changes with order, precision, dimension, compilation, or grid size.

It cannot establish:

- characteristic-Euler parity or speed;
- CPU or cross-GPU behavior;
- a production automatic-selection threshold;
- that TorchInductor physically materializes every logical face tensor;
- that global face storage is preferable to a future tiled representation;
- an arbitrary-order native-CUDA result; or
- publication readiness.

## Stop condition

Stop after the twelve-configuration correctness gate, the 24 paired timing
endpoints or recorded failures, memory and compiler records, interpretation,
immutable evidence with SHA-256 hashes, a verifier/regression test, and a
clean local commit. Do not modify `src/gradflow/`, begin a broader campaign,
or push without explicit authorization.
