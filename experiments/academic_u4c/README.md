# Academic U4-C external CUDA and performance study

U4-C is governed by `docs/ACADEMIC_U4C_PROTOCOL.md`. It first qualifies the
U4-B OpenSBLI operator on CUDA, then compares only correctness-admitted,
mathematically matched OpenSBLI and compiled GradFlow endpoints.

No U4-C timing existed when the protocol was frozen.

The C1 CUDA qualification is complete and passed. Its machine-readable record
is in `evidence/u4c_c1_20260830/`, with interpretation in
`docs/ACADEMIC_U4C_C1_RESULTS.md`. Comparative timing remains a separate gate.

C2 is complete. Only `N=8192` passed the prospectively frozen external
pointwise bounds; OpenSBLI won both admitted one-thread CPU and CUDA resident
comparisons. Three larger sizes are retained as correctness exclusions, not
silently discarded measurements. See `docs/ACADEMIC_U4C_C2_RESULTS.md` and
`evidence/u4c_c2_20260830/`.
