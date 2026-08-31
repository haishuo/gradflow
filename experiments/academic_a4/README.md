# Academic A4 release work

This directory contains the deterministic artifact index, its verifier, and
the clean-checkout reproduction driver for the first GradFlow academic release
candidate.

The release payload is every tracked file except the index itself and the
post-release audit records named in `build_artifact_index.py`. The exclusion
prevents self-referential hashes and separates the tested scientific payload
from the report about testing it.

See `docs/ACADEMIC_A4_PROTOCOL.md` for the frozen independent replication,
external-review, rights, and data-center value-of-information rules.

The SLURM adapter for the prospective UMass Unity second-machine run is under
`experiments/academic_a4/unity/` and is governed by
`docs/ACADEMIC_A4_UNITY_PROTOCOL.md`.

Release candidate 2 extends that contract through U4 and stable-release U5.
Its amendment is `docs/ACADEMIC_A4_RC2_PROTOCOL.md`, its interpreted result is
`docs/ACADEMIC_A4_RC2_RESULTS.md`, and its evidence is under
`evidence/a4_rc2_20260831/`.
