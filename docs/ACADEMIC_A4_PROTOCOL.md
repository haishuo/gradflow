# Academic A4 replication, audit, and release protocol

Status: **frozen before A4 release work**.

Date: 2026-08-30 (UTC)

## Purpose

A4 turns the completed A1--A3 studies into an auditable release candidate. It
does not add a numerical method, optimize a backend, or widen the first-paper
claim. The governing order remains:

> correctness > performance > convenience

The release candidate may be locally complete while independent gates remain
pending. A Forge rerun is not a second-machine replication, and an author
review is not an external numerical-CFD/prior-art audit.

## Frozen release states

The A4 status is one of:

1. `local_release_candidate`: the source, evidence, environment record,
   rights record, artifact index, and clean-checkout reproduction pass locally;
2. `external_review_pending`: the local release candidate passes but either
   the second-machine replication or external audit is incomplete; or
3. `academic_release_ready`: both independent gates also pass and all public
   redistribution blockers are resolved or the affected files are excluded
   from the public artifact without changing the scientific record.

Only state 3 closes A4. State 2 is a citable *internal release candidate*, not
a submission-ready public artifact.

## Frozen scientific sentinels

The release index covers the canonical source and all A1--A3 evidence. The
following compact sentinels are rerun from a clean checkout:

- the complete test suite, with CUDA/MPS skips reported rather than simulated;
- the offline semantic and checksum verifiers for A1, A2, and A3;
- reference hashes and byte identity checks; and
- a source-tree integrity check against the release manifest.

These checks reproduce the recorded conclusions from immutable observations;
they do not replace the costly original performance campaigns. A local
same-machine performance rerun may diagnose drift, but cannot satisfy the
independence gate.

## Second-machine replication contract

A suitable replication machine must be physically distinct from Forge and
record its CPU, operating system, Python, PyTorch, accelerator, driver, CUDA,
and memory identities. It must run:

1. all clean-checkout sentinel checks;
2. the A1 numerical-limit campaign or a prospectively declared subset that
   includes orders 5, 11, and 15 in binary32 and binary64;
3. the A3 inverse and gradient campaign in binary64 on CPU and, when CUDA is
   present, CUDA; and
4. the following A2 performance sentinels with three fresh isolated workers
   per lane: scalar WENO-JS orders 5, 11, and 15 at `64^3`, binary32 and
   binary64, CPU eager/compiled and CUDA eager/compiled when present.

Correctness admission precedes timing. Performance is reported as a new
machine-specific observation; it need not reproduce Forge's exact ratios.
Replication passes when the same qualitative conclusions hold:

- qualified scalar CPU/CUDA results agree under the frozen parity policy;
- compiled execution captures one graph with zero graph breaks;
- large 3-D CUDA is materially useful in at least one admitted binary32 lane;
- binary64 conclusions remain explicitly hardware-conditioned; and
- A3 autograd, centered differences, and derivative-free recovery agree under
  the frozen A3 tolerances.

Any failure is retained and investigated before paper wording is frozen.

## External audit contract

At least one reviewer with numerical CFD/WENO expertise who is not an author
must receive the external-review packet. The audit asks the reviewer to check:

- the WENO-JS formulation and finite-difference/finite-volume terminology;
- oracle independence, boundary conventions, convergence, conservation,
  critical-point behavior, and characteristic reconstruction limitations;
- whether parity thresholds and exclusions support each numerical claim;
- whether timing endpoints and baselines support each performance claim;
- the established/observed/inferred/untested claim labels; and
- omissions or misclassification in the close-prior-art comparison.

The reviewer, date, version reviewed, findings, author responses, and any
resulting changes must be recorded. Silence or informal conversation does not
count as an audit.

## Data-center FP64 value-of-information rule

Renting A100/H100-class hardware is not required merely because Forge uses a
consumer GPU. It is justified before the first submission only if all are
true:

1. the paper would otherwise make a cross-hardware or FP64 GPU-performance
   claim that the RTX 5070 Ti cannot support;
2. the second-machine result does not already supply a suitable data-center
   accelerator observation;
3. the exact frozen jobs, maximum cost, and decision they can change are
   written before rental; and
4. the expected information changes the paper more than reporting the current
   hardware limitation honestly.

Under the current bounded claim, data-center rental is **not mandatory**. It
would add a valuable portability stratum, but absence of that stratum forbids
general data-center FP64 claims rather than invalidating the Forge result.

## Redistribution rule

No license terms are inferred. The Gottlieb MATLAB and Jiang--Shu Fortran
research copies remain locally preserved with hashes and provenance. Until
documented permission or a rights review resolves their status, they are a
public-release blocker. A public artifact may instead omit their bytes and
provide provenance, hashes, acquisition instructions where lawful, and
independently redistributable fixtures; such omission must be tested and must
not be described as permission to redistribute the originals.

## Frozen artifact contents

The A4 index records SHA-256 hashes for:

- canonical package source and project metadata;
- tests and mathematical/reference fixtures used by paper claims;
- A1--A3 protocols, results, scripts, raw records, and checksum manifests;
- the formulation, research-direction, scope, and prior-art records; and
- the environment, rights, reproduction, and external-review documents.

Generated caches, AOT binaries, temporary compiler products, modified
Fortran descendants, and untracked files are excluded. AOT package hashes and
preparation times already recorded by A2 remain evidence, but the
machine-specific binaries are not claimed portable.

## Clean-checkout reproduction

The release commit is cloned locally without shared working-tree files and
without network access. From that clone, the frozen interpreter runs the test
suite, A1--A3 verifiers, reference verification, and A4 manifest verifier.
Commands, exit codes, skips, durations, commit identity, environment identity,
and log hashes are preserved. A dirty source tree or a checksum mismatch
fails the reproduction.

## Stop condition

A4 local work stops after the protocol, environment record, rights decision,
artifact index, review/replication packets, release commit, and clean-checkout
record are committed and verified. The branch may then be preserved remotely
under explicit authorization. The roadmap must continue to label A4 as
externally pending until actual second-machine replication and the recorded
external audit pass.
