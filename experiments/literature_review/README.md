# Literature-review artifacts

This directory contains machine-readable records for GradFlow's bounded
literature and claim reviews. The review protocol is frozen in
`docs/LITERATURE_REVIEW_PHASE_C_PROTOCOL.md`; the corresponding interpretation
is in `docs/LITERATURE_REVIEW_PHASE_C_RESULTS.md`.

Verify the committed Phase-C records with:

```bash
python experiments/literature_review/verify_phase_c.py
```

The records contain bibliographic metadata, source URLs, classifications, and
hashes. They do not redistribute papers.
