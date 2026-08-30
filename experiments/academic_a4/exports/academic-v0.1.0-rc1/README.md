# GradFlow academic export: `academic-v0.1.0-rc1`

This directory is the immutable, paper-facing output of GradFlow's local A1--A4
academic campaign. A downstream manuscript repository may vendor this whole
directory and generate presentation artifacts from `paper_data.json`.

Generate from the committed source evidence with:

```bash
python experiments/academic_a4/export_paper_data.py
```

Verify the dataset by comparing its SHA-256 digest with
`export_manifest.json`. The manifest also pins the GradFlow release tag and
commit and hashes every source evidence file used by the exporter.

This export does not close the pending second-machine replication or
independent numerical-CFD review gates. It contains no manuscript prose and
does not grant redistribution rights for GradFlow's separately preserved
historical reference sources.
