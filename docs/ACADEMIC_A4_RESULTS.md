# Academic A4 replication, audit, and release result

Status: **local release candidate complete; independent gates pending**.

Date: 2026-08-30 (UTC)

## Result

GradFlow now has a hashed, citable internal academic release candidate:

- tag: `academic-v0.1.0-rc1`;
- commit: `99a2a806fdaedb6cc32cdad2d621144d014865de`;
- indexed payload: 1,855 files and 204,131,624 bytes;
- artifact-index SHA-256:
  `88743a3acaa15c20081822f83f426429e62849734a78915dbd9fdbae661d1d4e`;
- clean-room mode: isolated local clone with no hard links and no network; and
- clean-room result: 293 passed, 72 declared skips, zero failures, followed by
  successful A1, A2, A3, and A4 offline verification.

The clone was clean before and after the run. The 72 skips are explicit CUDA
visibility and optional external-DVEB-fixture gates; CUDA was not simulated.
CUDA-visible A2/A3 results remain preserved in their original immutable
campaign records.

This establishes local artifact integrity. It is not a second-machine result.

## Reproducibility corrections found by A4

The dry run caught two release-engineering defects before the tag:

1. checkout tests depended on stale editable-install path behavior; and
2. older FD/FV verifiers required byte/array equality when independently
   regenerating floating-point quadrature.

The first was corrected by making checkout import paths explicit. For the
second, frozen files retain exact SHA-256 checks, while regenerated values use
`rtol=0`, `atol=5e-14`. Observed drift was at most
`1.887379141862766e-15` for the Phase-5A JSON and
`4.440892098500626e-16` for the Phase-6A arrays. The new reproduction bound is
forty times tighter than the existing `2e-12` oracle tolerance. No canonical
method, evidence bytes, or scientific gate changed. See
`ACADEMIC_A4_REPRODUCIBILITY_CORRECTIONS.md`.

## Environment and artifact freeze

The Forge environment records Python 3.11.13, PyTorch
2.9.0.dev20250705+cu128, CUDA runtime 12.8, an AMD Ryzen 5 7600X, and an NVIDIA
GeForce RTX 5070 Ti. The exact development wheel may no longer be obtainable;
that limitation is explicit rather than hidden behind a misleading lock file.

The index covers the complete tracked repository payload, including source,
tests, protocols, raw records, references, and negative results. Self-hashes
and post-tag audit reports are deliberately outside the indexed payload and
have their own checksums.

## Data-center FP64 value of information

An A100/H100 rental is **not mandatory for the bounded first-paper claim**.
The current paper can report Forge binary64 performance as specific to a
consumer GPU with weak FP64 throughput. A data-center run becomes worth paying
for if the final paper makes cross-hardware or data-center FP64 claims, or if a
reviewer identifies it as decisive. Until then it is a valuable portability
extension, not a correctness prerequisite.

## Rights audit

Public redistribution is unresolved in three places:

- GradFlow has no selected top-level software license;
- no public-redistribution permission was found for the Gottlieb MATLAB set;
  and
- no public-redistribution permission was found for the original Jiang--Shu
  Fortran set.

The local research copies remain preserved and hashed. A public archival
artifact needs documented permissions or a tested source-excluded package.
No license terms have been invented.

## Gates that remain open

A4 is not closed and the paper is not submission-ready until:

1. the frozen sentinel matrix is run on a physically distinct suitable
   machine; and
2. an independent numerical-CFD/WENO reviewer completes the formulation,
   correctness, performance, claim, and prior-art audit.

Executable handoff documents are in
`ACADEMIC_A4_SECOND_MACHINE_PACKET.md` and
`ACADEMIC_A4_EXTERNAL_REVIEW_PACKET.md`. Their absence cannot be substituted
by another Forge run or author self-review.

## Permitted paper status

The evidence is sufficient to begin a manuscript draft, construct figures and
tables from immutable records, and seek external review. Claims must remain
bounded to the tested formulation, orders, problems, hardware, endpoints, and
failure exclusions. Paper wording is provisional until the two independent
gates respond.
