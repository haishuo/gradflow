# Academic external-baseline gate

Status: **blocking manuscript gate; protocol only; no campaign begun**.

## Question

What execution cost does the ordinary-PyTorch formulation impose relative to
a mathematically matched low-level implementation and, where one can be made
genuinely comparable, an independently maintained external system?

## Admission before timing

Every baseline must declare and match, or explicitly disclose a mismatch in:

- finite-difference versus finite-volume formulation;
- WENO family and order;
- flux splitting and alpha policy;
- epsilon and smoothness-indicator scaling;
- grid and periodic-endpoint convention;
- boundary handling and time-integration endpoint;
- dtype and state layout;
- resident versus transfer-inclusive timing; and
- semidiscrete RHS, full step, or complete-solve scope.

A fast lane that fails the frozen correctness comparison is an exclusion, not
a speedup. An unmatched finite-volume implementation is relevant prior art but
is not labeled a direct finite-difference performance baseline.

## Required controls

1. The existing separately written native CUDA WENO-5 control must be exported
   and presented under its exact mathematical and endpoint contract, or omitted
   from the manuscript entirely.
2. DVEB may be included as a compiler-abstraction control only under a matched
   contract; it is not an independent external system.
3. At least one independently maintained external implementation should be
   qualified when a defensible match is possible. Candidate selection requires
   a frozen compatibility audit before benchmarking.
4. If no external implementation can be matched without changing the
   mathematics or endpoint, the paper must report that evidence gap rather
   than manufacture an apples-to-oranges speedup.

## Required reporting

Report retained observations, medians, dispersion, correctness exclusions,
software revisions, build costs, warm and launch-to-answer endpoints, and all
known formulation mismatches. The resulting claim remains hardware- and
contract-specific.

This gate is untested. It does not authorize a new benchmark campaign by
itself.
