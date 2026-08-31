# Academic A4 external numerical-CFD and prior-art review packet

Status: **ready for an independent reviewer; review not yet performed**.

## Artifact to review

Review tagged candidate `academic-v0.1.0-rc2` and begin with:

- `docs/ACADEMIC_SCOPE.md`;
- `docs/ACADEMIC_A1_CLAIM_MATRIX.md`;
- `docs/ACADEMIC_A1_PRIOR_ART_COMPARISON.md`;
- `docs/FORMULATION_LINEAGE.md`;
- `docs/ACADEMIC_A1_RESULTS.md`;
- `docs/ACADEMIC_A2_RESULTS.md`;
- `docs/ACADEMIC_A3_RESULTS.md`; and
- `docs/ACADEMIC_U4E_RESULTS.md`;
- `docs/ACADEMIC_U4F_RESULTS.md`;
- `docs/ACADEMIC_U5_RESULTS.md`;
- the protocols and machine-readable evidence referenced by those documents.

## Requested audit

Please return a dated report identifying the reviewed commit/tag and answer:

1. Are the Jiang--Shu finite-difference formulation, epsilon conventions,
   flux splitting, characteristic reconstruction, and boundary conventions
   described correctly and without conflating FD and FV WENO?
2. Do the smooth, critical-point, conservation, boundary, Sod, Shu--Osher,
   device, and independent-oracle checks justify the bounded numerical claims?
3. Are the order-dependent failure boundaries and negative mixed-precision
   results interpreted conservatively?
4. Do the performance tables compare admitted mathematics, expose compilation,
   transfer, process, residency, and AOT boundaries, and avoid universal
   hardware claims?
5. Does the A3 inverse problem establish a genuine, independently checked use
   of differentiation without overstating application breadth?
6. Does the prior-art table omit or materially misclassify any close system,
   particularly arbitrary-order FD-WENO generation, differentiable WENO,
   ordinary-PyTorch CFD, or comparable compiler/AOT studies?
7. Which claims should be weakened, removed, or supported by another test?
8. Are there fatal defects, major revisions, minor revisions, or no identified
   blockers for drafting the paper?

Please cite exact file paths, claim identifiers, equations, or table rows for
each finding. The authors will preserve the report, responses, and changes as
an A4 record. Authorship or endorsement is neither assumed nor requested by
this audit.
