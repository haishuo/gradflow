# Moody second-machine evidence

This directory is the byte-preserved output of the prospectively frozen
Academic A4 Moody run described by `docs/ACADEMIC_A4_MOODY_PROTOCOL.md`.

- Source host: `moody-serious`
- Source directory: `/mnt/projects/gradflow-a4-moody-20260831/evidence`
- Scientific tag: `academic-v0.1.0-rc2`
- Scientific commit: `c5e8ab81ef5b33a2138b2db33afc538398b6f57f`
- Import method: `rsync -a --checksum` over SSH from Moody to Forge
- Imported UTC date: `2026-08-31`

`second_machine.json`, `raw/`, and `SHA256SUMS` are the original campaign
outputs. The repository-local `README.md` is not covered by the remote
`SHA256SUMS`; it records only the import boundary. Verify the imported evidence
with:

```bash
python3 experiments/academic_a4/moody/verify_moody.py \
  experiments/academic_a4/evidence/moody_20260831
```

The controller status is `fail_needs_investigation` because four general-suite
tests assumed artifacts deliberately absent from the replication packet. The
36 scientific A2 workers, A1 campaign, A3 campaign, graph contract, numerical
admission, and dedicated rc2 verifier all completed successfully. The exact
distinction is documented in `docs/ACADEMIC_A4_MOODY_RESULTS.md`.
