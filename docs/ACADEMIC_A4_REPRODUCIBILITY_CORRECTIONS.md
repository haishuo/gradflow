# Academic A4 reproducibility corrections

Date: 2026-08-30 (UTC)

These corrections were made before the release index and release-candidate tag
were created. They change neither a numerical method nor frozen evidence.

## Checkout import path

The first A4 full-suite dry run collected 162 tests and stopped with 13 import
errors because an obsolete editable-install record did not expose `src/`.
`pyproject.toml` now declares both the repository root and `src/` as pytest
import paths. The root is needed by subprocess verifiers that import the
`experiments` namespace; `src/` supplies the package. The clean-room driver
sets the same two paths explicitly.

## Archival identity versus regenerated floating point

Older nonlinear and Euler FD/FV verifiers conflated two different checks:

- committed evidence files must retain exact SHA-256 identity; and
- independent floating-point quadrature regeneration must agree numerically.

On Forge, exact regeneration differed from the frozen values by at most
`1.887379141862766e-15` in the Phase-5A JSON diagnostics and
`4.440892098500626e-16` in the Phase-6A arrays. The scientific gates still
passed, but byte/array equality failed. Such last-bit variation is not a valid
cross-machine reproduction criterion.

The evidence manifests and committed-file hashes remain exact and unchanged.
Only the independently regenerated comparison now uses `rtol=0` and
`atol=5e-14`, still forty times tighter than the Phase-5A oracle tolerance of
`2e-12`. The downstream Phase-6B verifier uses the same bound when repeating
the Phase-6A projection check. This is a portability correction, not a relaxed
scientific acceptance threshold.
