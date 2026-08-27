# Phase-D mixed-precision compiled-parity addendum

Protocol freeze date: 2026-08-27 UTC.

The first complete performance execution exposed a record-design omission: it
retained eager and compiled output finiteness and extrema, but not pointwise
compiled-versus-eager parity for every mixed policy. No performance record is
accepted until this omission is corrected. The initial execution is a pilot
and is not a frozen result.

For every order/policy pair in the unchanged performance protocol, the worker
must retain:

- maximum absolute compiled-versus-eager RHS difference;
- RMS absolute difference;
- maximum difference normalized by `max(abs(eager_rhs))`; and
- RMS difference normalized by `max(abs(eager_rhs))`.

Every numerically eligible policy must have maximum normalized difference no
larger than `5e-5` and RMS normalized difference no larger than `1e-5`. These
bounds are frozen before pointwise differences are observed. They are looser
than binary64 roundoff to allow compiler reassociation and mixed binary32
subexpressions, but stricter than the Tier-1b `engineering` class.

The explicitly inaccurate all-internal-binary32 endpoint records the same
metrics but cannot fail or pass the eligible-policy gate because it already
failed numerical qualification.

Timing parameters, input, policies, orders, fresh-cache process isolation, and
the 5% performance interpretation rule remain unchanged. The final benchmark
must come from a clean source revision containing this addendum and the
pointwise recorder.
