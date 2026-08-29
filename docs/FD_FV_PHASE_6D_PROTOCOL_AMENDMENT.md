# FD/FV Euler Phase 6D protocol amendment

The timed Phase 6D campaign completed all 92 frozen worker processes before the
aggregate writer failed. TorchInductor's `num_bytes_accessed` instrumentation
reported zero for the generated CPU graphs, and the original aggregate writer
attempted to divide two zero counters. The timing observations and generated
code inventories were already complete and remained intact under `raw/`.

This amendment permits exactly one infrastructure correction:

- interpret a zero denominator in an optional compiler-metric ratio as
  "unavailable", represented by JSON `null`;
- retain the independently populated IR-node metric as the other predeclared
  traffic-expansion signal;
- aggregate the 92 existing raw records without rerunning any timed worker;
- record both the timing-source commit and the aggregation-correction commit,
  with hashes of both source states.

The amendment does not change the matrix, samples, thresholds, eligibility
rules, causal decision tree, or numerical gates. It authorizes no optimization,
new timing cell, or Phase 6E work.
