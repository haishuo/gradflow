# Academic A4 rc2 artifact result

Status: **local rc2 release candidate complete; independent gates pending**.

Date: 2026-08-31 (UTC)

## Result

The stable-release and external-baseline work is now preserved in a second
hashed internal release candidate:

- tag: `academic-v0.1.0-rc2`;
- tagged commit: `c5e8ab81ef5b33a2138b2db33afc538398b6f57f`;
- indexed source commit: `4f2d3a5`;
- indexed payload: 3,192 files and 672,252,945 bytes;
- artifact-index SHA-256:
  `57b59b84286f0f2071d1662fc31e732e18bef591824af03e6c41479610ba8608`;
- clean-room mode: isolated local clone, no hard links, no network; and
- clean-room result: all 15 commands passed.

The CUDA-visible clone passed 355 tests with 12 declared external-DVEB-ABI
skips and one existing PyTorch deprecation warning. It then passed the A1, A2,
A3, prior rc1, U4-A through U4-F, U5, and tagged rc2 offline verifiers. The
clone was clean before and after execution.

This establishes local integrity of the expanded artifact. It is not a
second-machine result or an independent numerical review.

## Reproduction correction

The first clean-room attempt exposed one driver defect: the new U4-A verifier
invocation omitted its required evidence-directory argument. The test suite
and every other sentinel passed. The failed target was preserved locally as
`academic-v0.1.0-rc2-attempt1`; the unpublished `rc2` tag was then recreated
only after the driver and payload index were corrected. The final clean-room
run passed all 15 commands. No numerical source, evidence byte, tolerance, or
scientific conclusion changed.

## What rc2 adds over rc1

rc2 preserves:

- the qualified OpenSBLI external operator and deployment endpoints;
- the matched OpenSBLI, PyTorch, and DVEB U4 comparisons;
- the U4-F batched backend regime map;
- the complete stable PyTorch 2.13 reproduction, including unfavorable CUDA,
  compilation, and AOT changes; and
- updated second-machine and external-review packets that identify rc2 and
  stable U5 as the primary paper evidence.

The July-2025 PyTorch development build remains in the artifact as a
version-sensitivity stratum, not the primary performance environment.

## Remaining gates

The academic artifact is still not submission-ready until:

1. the frozen replication packet runs on a physically distinct suitable
   machine;
2. an independent numerical-CFD/WENO reviewer completes the recorded audit;
3. public redistribution of the Gottlieb MATLAB and original Jiang--Shu
   Fortran sources is resolved or a tested source-excluded public artifact is
   built; and
4. a top-level GradFlow software license is selected before public release.

An A100/H100 run remains optional for the bounded paper. Without it, FP64
claims stay specific to the RTX 5070 Ti and no datacenter-GPU extrapolation is
permitted.

No tag or branch was pushed as part of rc2 construction.

