# Phase-D scalar mixed-precision performance protocol

Protocol freeze date: 2026-08-27 UTC.

The Tier-1a and Tier-1b numerical searches are complete and immutable. This
protocol asks whether their passing demotions save time on the local RTX 5070
Ti. Accuracy classifications are imported from the verified Tier-1b record;
timing cannot upgrade a failed assignment.

## Frozen endpoints

Five precision policies are measured:

1. `all_f64`: every internal block binary64;
2. `indicators_f32`: only smoothness indicators binary32;
3. `weight_formation_f32`: only unnormalized weight formation binary32;
4. `indicators_and_weight_formation_f32`: both passing candidate blocks
   binary32; and
5. `all_internal_f32`: every internal block binary32, retained only as an
   explicitly inaccurate hardware floor.

The first four are numerically eligible, subject to their order-specific
`tight` or `engineering` classifications. The fifth failed the numerical gate
and can never be recommended from this experiment.

Representative orders are WENO-5, WENO-11, and WENO-15. The measured operation
is one scalar periodic Burgers RHS over `2^20 = 1,048,576` binary64 state
values resident on CUDA. The deterministic state is

`0.3 + 0.6*sin(2*pi*x) + 0.1*cos(6*pi*x)`,

the flux is `q^2/2`, the explicit global LF alpha is `1.5`, and `dx=1/N`.
Host/device transfers are outside the timed region.

This is a large scalar reconstruction throughput test, not a full CFD step.
It is deliberately bounded to the code whose precision seam was qualified.

## Frozen execution modes

Each order/policy pair runs in a fresh Python process:

- ordinary eager PyTorch; and
- `torch.compile(..., fullgraph=True, dynamic=False)`.

Compiled first-call latency is measured from invocation through synchronized
completion with a fresh per-worker TorchInductor cache. It is recorded
separately and excluded from warm execution time.

For each execution mode:

- 5 unrecorded warm-up calls;
- 30 recorded calls;
- CUDA-event timing for every call;
- device synchronization before extracting measurements; and
- median, first quartile, third quartile, minimum, maximum, and arithmetic mean
  in milliseconds.

Peak allocated CUDA memory during the recorded calls is retained. The runner
records Python, PyTorch, CUDA, driver, GPU, source revision, dirty state,
command, policy assignments, and output finiteness.

## Interpretation

The primary comparison is warm compiled device-resident median time against
`all_f64` at the same order. Eager timing diagnoses unfused cast/kernel costs.
Compile latency describes deployment economics but does not enter the warm
speed ratio.

A candidate is performance-positive only if its median is lower than
`all_f64`. Differences smaller than 5% are reported as practically unresolved
by this bounded single-machine campaign even if their sample medians differ.
This 5% rule is an interpretation threshold, not a statistical confidence
interval.

Results apply to the RTX 5070 Ti and this software stack. The consumer GPU's
weak binary64 throughput makes an FP32 benefit plausible, but no A100/H100 or
CPU conclusion can be inferred. A scalar-kernel win also cannot be extrapolated
to characteristic Euler until Tier 2 passes.
