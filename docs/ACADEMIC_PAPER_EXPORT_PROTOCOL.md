# Academic paper export protocol

Status: **frozen GradFlow-to-paper boundary**.

Date: 2026-08-30 (UTC)

## Repository boundary

GradFlow is the research software and evidence authority. It does not contain
the manuscript, manuscript figures, rendered tables, bibliography, or
submission machinery. Those belong to a separately versioned downstream paper
repository.

GradFlow provides a compact, immutable, machine-readable export. The paper
repository vendors one exact export, verifies its hash, and derives its prose,
figures, and tables without reading the live GradFlow checkout. This mirrors
the established MVNMLE separation: validation produces versioned artifacts;
the paper consumes them.

The governing order remains:

> correctness > performance > convenience

## Frozen research question

> How accurately, differentiably, and efficiently can one exact-generated
> Jiang--Shu finite-difference WENO implementation execute as maintainable
> ordinary PyTorch from orders 5 through 15, relative to mathematically
> matched CPU, compiler-generated, and native-GPU baselines?

The study does not claim a new WENO formula, the first use of WENO in PyTorch,
universal GPU superiority, or production aerospace readiness.

## Evidence and export ownership

The v2 export is derived from these committed authorities:

- `experiments/academic_a1/evidence/a1_20260830/numerical_limits.json`;
- `experiments/academic_a2/evidence/a2_20260830/analysis.json`; and
- `experiments/academic_a3/evidence/a3_20260830/campaign.json`.

The export program is
`experiments/academic_a4/export_paper_data.py`. Its versioned output is under
`experiments/academic_a4/exports/academic-v0.1.0-rc1/` and contains:

- `paper_data.json`, the downstream dataset; and
- `export_manifest.json`, recording the release tag and commit, generator
  hash, source-evidence hashes, dataset size, and dataset hash.

The recorded tag and commit identify the frozen source-evidence state. The
export itself is identified independently by the SHA-256 digest of
`export_manifest.json`, so downstream consumers do not imply that the exporter
was already present in the earlier release-candidate tag.

Raw campaign evidence remains in GradFlow and is never modified by export.
Changing or extending the exported reporting surface requires a new
prospectively named export; published exports are not silently overwritten.
The initial `academic-v0.1.0-rc1` export remains immutable. The
`academic-v0.1.0-rc1-paper-v2` export adds complete manuscript reporting fields
without adding or rerunning an experiment.

Paper export v3, `academic-692f822-paper-v3`, extends v2 with the frozen U4-C,
U4-D, and U4-E matched-control evidence and the normative backend-identity
policy. It is pinned to exact GradFlow commit
`692f822ef7fef9770247ac56e3526b0f3ac2436c`; it is not mislabeled as a release
candidate. Its additional authorities are the committed C2/C3, U4-D, E1, and
U4-E JSON records listed in its manifest. The eventual clean-room rc2 release
will receive a new export identity rather than silently changing v3.

Paper export v4, `academic-v0.1.0-rc2-paper-v4`, supersedes v3 after U4-F,
the stable-PyTorch U5 replication, and the rc2 clean-room audit. It makes
stable PyTorch 2.13 the primary toolchain evidence and retains the earlier
development build explicitly as a version-sensitivity stratum.

Paper export v5, `academic-33c469b-unity-paper-v5`, extends v4 with the
prospectively frozen Unity result. Unity's allocated Tesla M40 could execute
CUDA eager mathematics but could not instantiate the stable TorchInductor
backend because its compute capability is below Triton's supported floor.
The export classifies this as a negative legacy-GPU portability observation,
not completion of the suitable modern-GPU second-machine gate. The separately
frozen Moody result requires a later immutable export; v5 will not be changed
in place.

Paper export v6, `academic-3992e06-moody-paper-v6`, extends v5 with the
prospectively frozen Moody result from GradFlow evidence commit
`3992e06939005d27b2d99017992e68d383b5034f`. All 36 A2 workers and the A1/A3
scientific gates completed without admission or graph failure on the distinct
Ryzen 7 7700/RTX 4070 SUPER system. The export also preserves the controller's
literal `fail_needs_investigation` status: its general-suite sentinel had four
failures caused by the tag-restricted packet omitting the older rc1 tag and
machine-specific Phase 6E AOT artifacts. V6 exports both the usable scientific
replication and that formal packet limitation; it does not relabel the source
record or silently repair the prospectively frozen run.

Paper export v7, `academic-f65803e-moody-paper-v7`, supersedes v6 for paper
rendering without changing a measurement. It restores information that v6's
summary omitted: each Moody CPU worker timed one and six PyTorch intra-op
threads, and six threads supplied every selected CPU value. V7 also exports a
matched primary/Moody disposition. Five of six CPU/CUDA winners reproduced;
binary64 WENO-15 reversed from CPU on the primary RTX 5070 Ti system to CUDA
on Moody's RTX 4070 SUPER. The binary32 CUDA advantage and graph/admission
structure reproduced. This is explicitly partial performance-ordering
reproduction, not a complete replicated phase diagram.

## Downstream paper contract

The paper repository must:

1. vendor an exact GradFlow export in a release-specific input directory;
2. verify the vendored dataset against its export manifest before generation;
3. generate all displayed numeric values from that dataset or identify a
   specific GradFlow evidence path and release commit;
4. keep claim status and scope boundaries explicit;
5. identify GradFlow by repository URL, release tag, and commit; and
6. distinguish a completed modern-GPU scientific replication from its
   documented all-sentinels packet limitation, and treat independent CFD
   review as pending until that external gate closes.

An external or mathematically matched low-level comparison is also a blocking
manuscript gate under `docs/ACADEMIC_EXTERNAL_BASELINE_GATE.md`. An unsupported
native-control aside is not a substitute for exporting and presenting the
comparison.

The paper repository must not import GradFlow internals, reach through a
relative path into a neighboring checkout, or treat the mutable branch tip as
an evidence source.

## External gates

Local paper preparation does not close Academic A4. Before submission, a
physically distinct machine should execute the frozen A4 replication packet,
and an independent numerical-CFD/WENO reviewer should audit the candidate.
The completed Unity attempt is a negative legacy-GPU portability stratum and
does not close the gate because its allocated Tesla M40 cannot run the frozen
compiled backend. Moody's prospectively frozen suitable modern-GPU follow-up
has completed and supplies a bounded scientific replication, with the four
transport-packet sentinel failures retained as an explicit formal limitation.
