# FD/FV Euler Phase-6A contract and oracle results

Status: **passed; no production FV Euler implementation or performance timing
was created**.

The prospective protocol was frozen at commit `9d1b567`. The oracle generator
ran from clean source commit
`93b8749fef8611e3a5450329d2509f9c6ef26fb2`.

The immutable records are:

| Artifact | SHA-256 |
|---|---|
| `experiments/fd_fv_euler/results/phase_6a_20260828/contract.json` | `00ffe129fff3bb5e1f1ccea817ba6a5164adc46d489e20709854e93de9121c9d` |
| `experiments/fd_fv_euler/results/phase_6a_20260828/projections.npz` | `56670eb847c8fe643f96f55275edf20a1c9a01957d248c24fd81ff3ebe6f27a4` |

## Decision

Phase 6A admits a future correctness-only Phase 6B implementation of one
matched dimension-by-dimension FV Euler formulation. It does not admit timing.

The matched seed identifiers are:

```text
fd_classical_characteristic_js5_global_lf_euler1d_v1
fv_dimensional_characteristic_js5_global_matrix_lf_euler1d_v1
```

Both use ideal-gas Euler, WENO-JS5, epsilon `1e-6`, 12-scaled indicators,
face-frozen Roe projection, the same line-global characteristic LF speeds with
`1.1` enlargement, conservative flux differencing, SSP-RK3, CFL `0.1`, and
float64 qualification.

The necessary mathematical difference remains visible. FD reconstructs split
characteristic physical fluxes from point values. FV reconstructs left/right
characteristic states from conservative cell averages and applies the matched
characteristic matrix-LF interface flux. The study does not pretend that those
operations are algebraically interchangeable.

## Independent evidence retained

Phase 6A verified and reused the exact Sod Riemann solver, the separately
written NumPy FV WENO-Z/HLLC reference procedure, the frozen 12,800-cell
Shu--Osher array, the inherited thresholds, and the qualified GradFlow FD
Euler record. Every identity matches the hashes frozen in the protocol.

The high-resolution Shu--Osher reference is numerical rather than exact and
originates from an FV method. That provenance is explicit. FD receives fixed
interpolation to its point coordinates; FV receives conservative restriction
to its cell averages. The two resulting discrete oracle arrays are never
directly compared.

## Smooth projection result

For the periodic entropy wave, analytic FV cell averages agreed with
32-point Gauss--Legendre integration to at most `8.882e-16`. Exact periodic
RHS sums were at most `3.054e-15` for the point projection and `2.165e-15` for
the cell-average projection.

The maximum point-versus-cell-average state differences at time zero were:

| N | Maximum difference |
|---:|---:|
| 24 | `2.8289e-4` |
| 36 | `1.2639e-4` |
| 54 | `5.6401e-5` |
| 81 | `2.5065e-5` |

Their approximately second-order decrease is the expected projection
difference. It demonstrates why copying one method's initial array into the
other would be mathematically wrong even though both use cell-center
coordinates.

## Exact Sod cell-average result

At `N=(200,400,800)`, exact conservative cell averages were integrated after
splitting every cell at known rarefaction, contact, and shock locations.

- 32- versus 64-point quadrature disagreement was at most `1.333e-15`.
- The largest domain-integral error against exact boundary-flux balance was
  `6.661e-16`.
- Minimum exact average density was `0.125`.
- Minimum primitive pressure derived from the exact average was
  `0.10000000000000002`.

The exact final integrated conserved state is

```text
(mass, momentum, energy) = (0.5625, 0.18, 1.375).
```

The nonzero momentum is required by the unequal left and right pressure fluxes;
it is not a conservation failure.

## Shu--Osher projection result

The 12,800-cell conserved reference was block-averaged to 200, 400, and 800
cells with restriction factors 64, 32, and 16. The largest fine-versus-
restricted global conserved-integral discrepancy was `1.776e-15`.

The initial FD point and FV cell-average states differ by `5.204e-4`,
`1.302e-4`, and `3.255e-5`, again displaying the expected projection
difference rather than an initialization inconsistency. The right sinusoidal
density averages are analytic, and the `x=-4` interface lies on a cell face
for every frozen grid.

## Frozen Phase-6B gate

The future production FV implementation must pass, before timing:

- exact point/cell-average projection identity;
- uniform-state preservation;
- smooth spatial and complete-solve convergence;
- boundary-flux conservation;
- stage-by-stage positive density and pressure;
- inherited Sod and Shu--Osher thresholds;
- float64 eager CPU execution and CPU/CUDA agreement;
- full-graph compilation with zero graph breaks;
- independently checked smooth directional derivatives; and
- no hidden host/device transfer or scalar extraction.

The best-practical lane remains registered but empty. The inherited
WENO-Z/HLLC oracle is not silently promoted into a production competitor.

## Verification and claim boundary

The independent verifier checks both artifact hashes, every inherited and
source identity, regenerates every stored array exactly, recomputes every gate,
and confirms that the oracle path imports neither PyTorch nor GradFlow
production code.

Extending the experimental constitution exposed that the older Phase-5A
verifier regenerated its contract using mutable current documentation. Its
immutable record remained intact. The verifier now checks the contract's
source hashes against the original Git blobs at commit `0c974ab` and still
regenerates the numerical oracle cases byte-for-byte. No Phase-5A artifact or
decision changed.

The final configured Forge suite passed `323` tests, including CUDA-visible and
DVEB-linked gates. Fourteen warnings came from PyTorch's deprecated internal
JIT scripting API; no GradFlow test failed.

Phase 6A establishes a fair, representation-aware Euler comparison contract.
It establishes no FV Euler implementation, FD/FV winner, performance result,
multidimensional result, mixed-precision policy, DVEB result, data-center-GPU
result, or publication claim.
