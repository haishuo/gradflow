# GradFlow academic paper export v2

Export ID: `academic-v0.1.0-rc1-paper-v2`

This reporting-complete export derives from the immutable A1--A3 evidence
frozen by GradFlow tag `academic-v0.1.0-rc1`. It supersedes v1 for manuscript
generation without modifying or invalidating the original export.

V2 adds no experiment. It adds the previously omitted reporting surface:

- descriptive timing dispersion for the scalar `64^3` comparison;
- all three clean-cache deployment observations, including CPU lanes;
- AOT build, package, loading, and warm-timing records;
- the directly exported gradient and inverse objectives;
- optimizer evaluation counts and parameter errors; and
- eager/compiled differentiation timing distributions and first-call costs.

Generate and verify with:

```bash
python3 experiments/academic_a4/export_paper_data_v2.py
python3 experiments/academic_a4/verify_paper_export_v2.py
```

Downstream consumers must vendor the complete directory and verify
`paper_data.json` against `export_manifest.json` before rendering.
