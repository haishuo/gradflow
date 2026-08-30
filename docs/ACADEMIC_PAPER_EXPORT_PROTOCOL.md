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

The export is derived only from these committed authorities:

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
Changing a value requires a new prospectively named export; published exports
are not silently overwritten.

## Downstream paper contract

The paper repository must:

1. vendor an exact GradFlow export in a release-specific input directory;
2. verify the vendored dataset against its export manifest before generation;
3. generate all displayed numeric values from that dataset or identify a
   specific GradFlow evidence path and release commit;
4. keep claim status and scope boundaries explicit;
5. identify GradFlow by repository URL, release tag, and commit; and
6. treat second-machine replication and independent CFD review as pending
   until those gates actually close.

The paper repository must not import GradFlow internals, reach through a
relative path into a neighboring checkout, or treat the mutable branch tip as
an evidence source.

## External gates

Local paper preparation does not close Academic A4. Before submission, a
physically distinct machine should execute the frozen A4 replication packet,
and an independent numerical-CFD/WENO reviewer should audit the candidate.
Unity is a possible future target but is not claimed available or suitable
until its actual allocation is known.
