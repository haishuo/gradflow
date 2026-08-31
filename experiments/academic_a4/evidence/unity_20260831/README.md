# Academic A4 Unity result

This directory is the imported, immutable result of the prospectively frozen
Unity second-machine attempt described by
`docs/ACADEMIC_A4_UNITY_PROTOCOL.md`.

- SLURM replication job: `63844884`
- compute node: `gypsum-gpu005`
- source tag: `academic-v0.1.0-rc2`
- source commit: `c5e8ab81ef5b33a2138b2db33afc538398b6f57f`
- result status: `fail_needs_investigation`
- source result archive SHA-256:
  `78ab01e5ab93ed8ee56e5d09c863eb8d27a05fa23bda723aae5d5e437d66c342`

`second_machine.json`, its `raw/` directory, and `SHA256SUMS` are the
controller's scientific record. `metadata/` contains the exact staged
controller, SLURM logs, environment freeze, submission record, and staging
hashes. The source Git bundle is deliberately omitted because its complete
history is already preserved separately and the record pins the tested tag
and commit.

The failure is retained as evidence. Stable PyTorch 2.13/Triton rejected the
allocated Tesla M40 because compute capability 5.2 is below Triton's supported
minimum of 7.0. CUDA eager mathematics remained admitted, but the frozen
compiled-CUDA and material-usefulness gates therefore could not pass. See
`docs/ACADEMIC_A4_UNITY_RESULTS.md` for the bounded interpretation.
