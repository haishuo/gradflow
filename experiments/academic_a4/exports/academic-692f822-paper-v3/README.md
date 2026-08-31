# GradFlow academic paper export v3

Export ID: `academic-692f822-paper-v3`

This export extends paper export v2 with the frozen U4-C, U4-D, and U4-E
matched-control evidence at GradFlow commit
`692f822ef7fef9770247ac56e3526b0f3ac2436c`. It is intentionally pinned to an
exact commit rather than mislabeled as a release candidate; the eventual
`academic-v0.1.0-rc2` clean-room release will receive a new export identity.

V3 adds no experiment. It adds:

- the U4-C external OpenSBLI/OPS admission surface and deployment endpoint;
- the U4-D historical three-way result;
- the prospective U4-E DVEB requalification and correctness records;
- resident, transfer-inclusive, and prepared-launch summaries; and
- the normative GradFlow-system/backend terminology mapping.

Generate and verify with:

```bash
python3 experiments/academic_a4/export_paper_data_v3.py
python3 experiments/academic_a4/verify_paper_export_v3.py
```

Downstream consumers must vendor the complete directory and verify
`paper_data.json` against `export_manifest.json` before rendering.
