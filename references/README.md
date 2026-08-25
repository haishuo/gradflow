# Authoritative research references

This directory keeps source references separate from executable GradFlow code.
The copies listed below were made on 2026-08-25 and verified byte-for-byte by
SHA-256 comparison with their source paths.

## Sigal Gottlieb MATLAB WENO-5

Source at preservation time:
`tests/reference_implementations/gottlieb_matlab/` in GradFlow commit
`4c861fdf4ec31932a8dd815ae9884be8ceba3a37`. The files first entered the
GradFlow history in commit `ff882987db22f8f677fd623410deb88d3aadb5af`.
Comments in `weno.m` and `weno5.m` identify Sigal Gottlieb (2003-09-07) and
a self-contained-function modification by Daniel Higgs (2007-06-22).

The scalar WENO-5 source uses point samples on `linspace(-1, 1, 101)`, so both
ends of the periodic interval are present. Its periodic extension deliberately
omits the two endpoint samples when copying values across the boundary:
`u(i-md:end-1)` on the left and `u(2:md+2)` on the right. This permits distinct
left and right traces at the duplicated coordinate in the discontinuous
`sign(x)` test. It uses global Lax-Friedrichs splitting with
`alpha = max(abs(f'(u)))`, `epsilon = 1e-29`, the 12-times-scaled Jiang--Shu
smoothness indicators, and `(epsilon + indicator)^2` in the weight algebra.
`BurgersTest.m` actually selects right-moving linear advection, `f(u)=u`, and
advances 75 steps with `dt = 0.5*dx` using SSP-RK3. The name describes the
example driver, not the active flux in the preserved file.

`reference_data.h5` is the selected pointwise oracle fixture for the current
scalar WENO-5 seed. The MATLAB sources, rather than the HDF5 container format,
remain the authoritative description of the convention.

| File | SHA-256 |
|---|---|
| `gottlieb_matlab/BurgersTest.asv` | `67b3177d809bca368e33cd0b9d35cdbc704c9374986b8a3fa6444b83b5efc6aa` |
| `gottlieb_matlab/BurgersTest.m` | `85d07c2e5b33e4a0b9c3eee9f6a23f049bcc9fa11b7e132c801f943fcd342022` |
| `gottlieb_matlab/reference_data.h5` | `f1a127f3b0d3c1d33acb1d9aaa1da5fb3dff100a58e614d4b0a5630b08e49a39` |
| `gottlieb_matlab/weno.m` | `fd555073570885197b8f46d9029ec5ee751c0c104a62277a17137f83c8ad09f6` |
| `gottlieb_matlab/weno5.m` | `fd555073570885197b8f46d9029ec5ee751c0c104a62277a17137f83c8ad09f6` |
| `gottlieb_matlab/weno7.asv` | `e7ded28a2074cdade2552b2373bae26b50f9600ede9573dc7981debf1156395c` |
| `gottlieb_matlab/weno7.m` | `6fa0b9a1822c1da93d80c8728f5528dff9c131978f82dd281cf08c1fd55b00e5` |

The WENO-7 files are retained because they accompanied the source set. They
are not used by GradFlow and do not authorize a WENO-7 implementation claim.

## Chi-Wang Shu original Jiang--Shu Fortran

Hai-Shuo Shu received this code directly from his father, Professor Chi-Wang
Shu. It was deliberately preserved before later experiments. These copies
came from the clean tracked paths `fortran/original/weno.f` and
`fortran/original/comm.inc` in the read-only `weno-reference` checkout at
commit `4d40511b03e2e4b27e33d02a7ae1f27550d72ba3`.

This is the authentic ancestral Jiang--Shu finite-difference WENO program,
but it is not interchangeable with Gottlieb's scalar MATLAB specialization.
It solves the two-dimensional compressible Euler system, reconstructs in
characteristic fields using Roe eigenvectors, uses `epweno = 1e-6`, and uses
per-characteristic global speeds enlarged by 10 percent (`am(m) *= 1.1`) for
Lax--Friedrichs splitting. It supports the program's RK3 and RK4 modes.
Consequently it is lineage evidence, not the pointwise oracle for the current
scalar implementation.

| File | SHA-256 |
|---|---|
| `jiang_shu_fortran/weno.f` | `9f1231516ef92b496333475ef29bfbba23afe77423163e7797bc8775a50186c5` |
| `jiang_shu_fortran/comm.inc` | `efc977da6582767cfa20ef76b0c3a0ace83e64083ca78f161668124e4cdbe3a7` |

Modified Fortran descendants are intentionally excluded. Their history is
recoverable from the GradFlow and `weno-reference` histories and from the
archives recorded in `docs/ARCHIVE_MANIFEST.md`.

## Redistribution status

No license or public-redistribution permission for either reference source
set was found in the supplied material. No license terms are inferred here.
The local research copies are preserved because provenance matters and the
task explicitly requires them. Whether either set may be included in a public
release remains an unresolved question requiring documented permission or a
separate rights review.
