# Academic A4 rc2 artifact amendment

Status: **frozen before rc2 indexing and clean-room execution**.

Date: 2026-08-31 (UTC)

This amendment retains every rule in `ACADEMIC_A4_PROTOCOL.md` and extends the
local release candidate through the completed external-baseline U4 studies and
stable-PyTorch U5 replication. It does not close the second-machine,
independent-review, rights, or licensing gates.

## Release identity

- candidate tag: `academic-v0.1.0-rc2`;
- prior candidate: `academic-v0.1.0-rc1`, retained permanently;
- primary PyTorch toolchain evidence: stable `2.13.0+cu130` from U5;
- development-build evidence: retained as a version-sensitivity stratum; and
- paper repository: separate and not part of the GradFlow source payload.

## Added payload and sentinels

The rc2 index covers every tracked GradFlow file except its own index and
post-tag rc2 audit records. In addition to the rc1 sentinels, the isolated
clone must verify:

- U4-A through U4-F external-baseline evidence;
- U5 environment, checksums, semantics, stable/development comparison, and
  backend-regime conclusions; and
- the rc2 artifact index itself against tagged Git bytes.

The clean-room test suite runs with CUDA visible on Forge. Optional external
DVEB ABI tests may remain skipped because their environment variables are not
part of the ordinary package contract; the committed U4/U5 evidence verifiers
independently check the frozen DVEB observations.

## Updated paper boundary

The paper may no longer present the July-2025 development wheel as the primary
performance environment. Stable U5 data must lead. The earlier data may be
used to demonstrate toolchain-version sensitivity, provided sequential runs
are not called paired and no single compiler change is assigned causally.

GradFlow remains backend-neutral. PyTorch, DVEB, OpenSBLI, and CPU/CUDA lanes
are implementations evaluated by correctness-admitted regime; none is renamed
as GradFlow itself and none is declared the universal winner.

## Stop condition

rc2 local work completes when the prospective payload commit is clean, the
index is generated from it, the candidate is tagged locally, an isolated
no-hardlink clone passes every sentinel without network access, the audit
record verifies, updated second-machine and external-review packets identify
rc2, and a versioned paper export is generated from tagged evidence.

No push is implied. The candidate remains `external_review_pending` until a
physically distinct machine and an independent numerical-CFD/WENO reviewer
complete their respective packets.

