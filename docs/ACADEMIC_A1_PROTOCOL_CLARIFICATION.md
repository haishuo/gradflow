# Academic A1 protocol operational clarification

Status: **frozen before implementation or execution**.

Date: 2026-08-30 (UTC)

This clarification resolves implementation-level definitions left implicit in
`ACADEMIC_A1_PROTOCOL.md`. It does not change the matrix, thresholds, or claim
boundary.

1. Cell-average moment matrices use raw monomials `x^k` on the unit-grid
   cells named by each exact offset. Their binary64 2-norm condition numbers
   are explicitly coordinate-basis diagnostics.
2. The smoothness restriction uses an orthonormal QR basis for
   `{v : sum(v)=0}` and computes the eigenvalue ratio of `Q^T B Q`; no
   numerical eigenvalue threshold is used to discard modes.
3. Exact-payload bit lengths cover every numerator and denominator in the
   candidate coefficients, optimal weights, full-stencil coefficients,
   smoothness matrices, and LDLT factors.
4. Roundoff errors compare the computed RHS with the analytic RHS evaluated
   in float64. L1 means mean absolute error. The first sampled roundoff onset
   is the first point after the sampled minimum whose L2 error exceeds the
   preceding point by 5%; it is `null` if none exists.
5. Epsilon-sweep error norms are divided by the prescribed input amplitude.
   The normalized RHS difference is
   `max(abs(rhs-rhs_baseline)) / max(max(abs(rhs_baseline)), amplitude,
   1e-300)`. Error-ratio classification uses normalized L2 error and treats an
   exactly zero baseline as a ratio of one only when both errors are zero.
6. Epsilon-sweep conservation uses the same dtype-scaled gate as N2. The
   `1e-40` lane is a numerical comparison baseline, not an oracle or proposed
   default.
