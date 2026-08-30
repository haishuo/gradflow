# Academic manuscript production protocol

Status: **frozen before paper-table, figure, and prose generation**.

Date: 2026-08-30 (UTC)

## Purpose

This phase converts the immutable A1--A4 research record into a venue-neutral
manuscript package. It does not reopen numerical development, add an
experiment, or treat unavailable second-machine replication as optional
evidence.

The governing order remains:

> correctness > performance > convenience

## Frozen paper question

> How accurately, differentiably, and efficiently can one exact-generated
> Jiang--Shu finite-difference WENO implementation execute as maintainable
> ordinary PyTorch from orders 5 through 15, relative to mathematically
> matched CPU, compiler-generated, and native-GPU baselines?

The paper is an empirical systems-and-numerics study. It does not claim a new
WENO formula, the first use of WENO in PyTorch, universal GPU superiority, or
production aerospace readiness.

## Immutable data authorities

Figures and tables are generated only from:

- `experiments/academic_a1/evidence/a1_20260830/`;
- `experiments/academic_a2/evidence/a2_20260830/`;
- `experiments/academic_a3/evidence/a3_20260830/`; and
- the A1 consolidation's already-hashed predecessor records.

Narrative values must either be generated into the paper dataset or be linked
to an exact evidence path. Raw campaign files are not modified.

## Frozen initial output set

The deterministic generator produces:

1. an order-by-order coefficient-conditioning and numerical-limit table;
2. a scalar `64^3` CPU/CUDA performance table for orders 5--15;
3. a 3-D crossover figure for WENO-JS5 and WENO-JS15;
4. a cross-order resident/copy-inclusive speedup figure in binary32 and
   binary64;
5. a clean-cache JIT versus AOT launch-to-answer figure;
6. a centered-difference gradient-validation figure; and
7. an inverse-resolution bias figure.

Machine-readable CSV/JSON accompanies every rendered figure or table. The
generator records input and output SHA-256 hashes. Rendering metadata is fixed
where supported; a second immediate generation must reproduce identical
output hashes in the frozen Forge environment.

## Manuscript structure

The initial draft contains:

1. abstract;
2. introduction and bounded contribution statement;
3. mathematical and implementation formulation;
4. verification and numerical-limit methodology;
5. execution and deployment methodology;
6. numerical, performance, AOT, and differentiation results;
7. limitations and threats to validity; and
8. conclusions and externally pending work.

FD/FV Phase 1--6, DVEB language development, commercial APIs, Navier--Stokes,
automatic backend selection, UI work, and further native CUDA optimization are
not central claims. They may be cited as supporting or future studies without
expanding this paper's experimental contract.

## Claim traceability

Every principal manuscript claim receives an identifier with status
`established`, `observed`, `inferred`, `untested`, or `prohibited`, plus exact
supporting files. Second-machine portability and external novelty approval
remain `untested`/`pending`; manuscript prose cannot silently promote them.

## Local completion and external gates

Local manuscript production completes when generated outputs are
deterministic, their hashes verify, the draft contains no unresolved numeric
placeholders, claim traceability is complete, and the repository test suite
passes.

This does not close Academic A4. Before submission, a physically distinct
machine should execute the frozen A4 replication packet, and an independent
numerical-CFD/WENO reviewer should audit the candidate. Unity is a possible
future target but is not claimed available or suitable until its actual
hardware/software allocation is known.
