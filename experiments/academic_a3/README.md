# Academic A3 inverse-advection experiment

The frozen protocol is `docs/ACADEMIC_A3_PROTOCOL.md`.

A3 infers one positive linear-advection speed from sixteen analytic terminal
observations by differentiating through an order-11 finite-difference WENO-JS
SSP-RK3 solve. Centered finite differences, the exact continuum solution, and
a derivative-free golden-section minimizer provide independent checks.

No canonical numerical source is modified by this experiment.

The primary Forge evidence is preserved under `evidence/a3_20260830/`. The
interpreted result is `docs/ACADEMIC_A3_RESULTS.md`.
