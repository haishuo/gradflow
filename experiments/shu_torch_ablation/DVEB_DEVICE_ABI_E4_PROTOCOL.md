# Frozen DVEB device-ABI E4 requalification protocol

Status: frozen before counted timing. This is a bounded addendum to
`DVEB_ABI_BAKEOFF_PROTOCOL.md`; it does not alter or replace that campaign.

## Question

Does DVEB portable device ABI v2 become competitive in the already-defined
E4 endpoint once it can accept caller-owned CUDA state directly?

Only E4 is reopened. E1--E3, automatic placement, the capacity frontier, and
all previous raw records remain unchanged.

## Frozen implementation and semantics

The candidate is the artifact produced from DVEB commit `0d12788`. Its v2
context is created before timing for one cubic grid and CUDA device. Each call:

1. receives the same immutable contiguous CUDA `float32` state;
2. receives a separate caller-owned CUDA output allocated before timing;
3. performs the full Shu Euler 3-D WENO-5 CFL plus requested SSP-RK3 steps;
4. uses the current PyTorch CUDA stream;
5. returns only after that stream is synchronized; and
6. performs no H2D/D2H copy or device allocation in the ABI run operation.

Context creation, workspace allocation, library loading, initial H2D state
creation, and final validation D2H are outside E4 timing. Wall time around the
public callable is primary. DVEB's internal event time is diagnostic only.

The existing direct-eager, persistent `torch.compile`, and packaged
AOTInductor resident lanes are rerun rather than compared only to measurements
from another thermal/time block. Their semantics remain those frozen in the
parent protocol.

## Gate and points

Before counted timing, DVEB v2 must pass full-array comparison against the
canonical PyTorch implementation at `(N,steps)={(6,1),(6,10),(32,1)}` with
`rtol=0`, `atol=2e-5`. A non-default PyTorch stream and exact input/output
alias are separately required to work. The timed lane uses distinct input and
output.

Counted points are exactly:

```text
steps=1:  N={8,16,32,64,96,128}
steps=10: N={16,32,64,128}
```

At each point, six randomized worker blocks contain all four lanes. Every
worker performs five uncounted warmups followed by five counted calls, giving
30 observations per lane and point. The seed is `20260827`. No point, lane,
tolerance, warmup count, or repetition count may change after seeing timings.

Report minimum, median, arithmetic mean, maximum, failures, and block wins.
“Winner” is the lowest median. “Competitive” means within 10% of that median.
Preserve all raw worker observations, ordering, telemetry, artifact hashes,
and environment records.

## Claim boundary

This experiment can qualify DVEB v2 only for this fixed WENO-5 program,
machine, precision, grid family, and E4 endpoint. It does not qualify automatic
placement, arbitrary order, other equations, other hardware, or a general
performance claim.
