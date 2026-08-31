# Academic U5 stable-PyTorch replication

The prospective constitution is
[`docs/ACADEMIC_U5_PROTOCOL.md`](../../docs/ACADEMIC_U5_PROTOCOL.md).

U5 reuses the frozen A1, A2, A3, and U4-F harnesses without changing their
scientific contracts. Evidence and the cross-version analysis will live under
`evidence/u5_20260831/`; temporary compiler caches, AOT packages, and full
qualification arrays remain outside the repository and are represented by
hashes and records.

The selected interpreter is expected at
`/mnt/projects/dveb/.venv-torch/bin/python`. Its location is not part of the
scientific claim: the evidence pins the executable and package identities, and
the release can be recreated from the recorded package versions. Running this
interpreter does not modify DVEB source or artifacts.

