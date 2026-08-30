# Academic external-baseline gate

Status: **blocking manuscript gate; U4-A and U4-B correctness qualification
complete; no external timing begun**.

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

The performance portion of this gate remains untested. This document does not
authorize a benchmark campaign by itself.

## U4-A and U4-B disposition

The source-pinned compatibility audit is complete. OpenSBLI is the selected
`matched_operator_candidate`, but its stock applications are not admitted:
epsilon scaling and the scalar semidiscrete endpoint require a small,
preserved adapter and correctness qualification. PyWENO is classified as a
reconstruction/code-generation building block. JAX-Fluids and HOPE are
application-context systems whose finite-volume contracts do not enter the
direct FD operator table.

The bounded U4-B adapter retained OpenSBLI's WENO, characteristic LLF,
divergence, periodic exchange, generated kernels, and OPS execution. Its
float64 scalar order-5 residual passed the frozen pointwise, constant-state,
conservation, and convergence gates and is classified
`matched_operator_adapted_qualified`.

See `ACADEMIC_U4A_PROTOCOL.md`, `ACADEMIC_U4A_RESULTS.md`,
`ACADEMIC_U4B_PROTOCOL.md`, and `ACADEMIC_U4B_RESULTS.md`. OpenSBLI remains
excluded from performance tables until a separate timing constitution and
CUDA correctness gate are completed.
