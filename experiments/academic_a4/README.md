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

