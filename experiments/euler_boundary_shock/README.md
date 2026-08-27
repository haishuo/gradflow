# Euler boundary/shock Phase A

This experiment prepares independent references before GradFlow implements a
nonperiodic Euler path. It follows `docs/EULER_BOUNDARY_SHOCK_PROTOCOL.md` and
does not import `gradflow`, PyTorch, or DVEB.

The exact Sod oracle solves the ideal-gas Euler Riemann pressure equation and
checks the star state, wave ordering, positivity, and Rankine--Hugoniot jump.
The separate numerical reference uses cell-centered finite-volume WENO-Z in
primitive variables, HLLC fluxes, transmissive constant-extrapolation ghost
cells, and SSP-RK3. This differs from the GradFlow finite-difference
Jiang--Shu/global-LF formulation under test.

The numerical reference must refine monotonically against the exact Sod
solution before it is used for Shu--Osher. Its highest Shu--Osher grid is
compared with the next-highest grid, and no reconstruction positivity fallback
is permitted in the frozen reference runs.

From a clean source commit, run:

```bash
conda run -n gradflow python \
  experiments/euler_boundary_shock/prepare_phase_a.py
```

The recorder refuses to overwrite an existing output directory. It writes
compressed reference arrays, the resolution studies, oracle-derived future
acceptance thresholds, source/environment identities, and `SHA256SUMS`.

No GradFlow boundary implementation, timing, optimization, or publication
claim is part of Phase A.

The frozen record is under `results/phase_a_20260827/`; its interpretation and
artifact hashes are documented in
`docs/EULER_BOUNDARY_SHOCK_PHASE_A_RESULTS.md`.

Verify the committed record with:

```bash
conda run -n gradflow python \
  experiments/euler_boundary_shock/verify_phase_a.py
```

Phase B applies GradFlow's generated characteristic finite-difference WENO-JS
to periodic and transmissive one-dimensional Euler problems. Its frozen
protocol is in `docs/EULER_BOUNDARY_SHOCK_PHASE_B_PROTOCOL.md`, and its result
record is under `results/phase_b_20260827/`.

Verify the Phase-B record and its dependency on the Phase-A oracle with:

```bash
conda run -n gradflow python \
  experiments/euler_boundary_shock/verify_phase_b.py
```
