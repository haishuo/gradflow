# Academic U4-F pre-campaign infrastructure amendment

Status: **frozen before any U4-F campaign qualification or resident timing**.

Date: 2026-08-31 (UTC)

## Trigger

An implementation smoke test compiled the new adapters and exercised batch
four without retaining any resident timing. DVEB CPU and both CUDA
qualification paths executed. The unchanged direct PyTorch formulation reached
one full CUDA graph, but TorchInductor's CPU backend raised an internal
scheduler assertion for the two-dimensional batched tensor.

This is an execution-availability observation, not a performance result. It
exposed an ambiguity in the frozen wording: whether one unavailable device
should suppress an otherwise independently qualified comparison on the other
device.

## Frozen clarification

Admission and timing are per `(batch, device)` comparison cell:

- both `dveb_native` and `pytorch_inductor` must execute and pass the full
  canonical, finiteness, shape, and per-row conservation gates on that device;
- only that device cell may then be timed;
- an unavailable or failed CPU cell does not authorize or prohibit CUDA
  timing, and vice versa; and
- CPU/CUDA agreement is required and reported when both same-backend device
  lanes execute, but is recorded as unavailable rather than used to erase an
  independently admitted device cell when one device cannot execute.

Every failure, return code, and diagnostic is retained. No fallback, graph
break, eager substitution, alternate PyTorch representation, relaxed bound,
removed batch, or runtime timing is permitted.

## Non-changes

The amendment does not change the mathematics, inputs, batch surface,
precision, hardware, backend artifacts, worker counts, warmups, samples,
randomization, bootstrap procedure, effect threshold, or endpoint. It exists
solely so an implementation failure is reported as a localized result rather
than either aborting the evidence record or silently suppressing a valid
device comparison.
