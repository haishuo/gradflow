# Academic A2 harness-scope correction

Status: **recorded before campaign resumption or analysis**.

Date: 2026-08-30 (UTC)

The initial orchestrator applied both `cpu` and `cuda` devices to every
mathematical configuration. The frozen A2 protocol instead declares the
characteristic float32 `64^3` E1 scale points as CUDA-only.

The mistake was detected after the order-five `64^3` CPU and CUDA workers
completed and while the order-fifteen CPU worker was running. The campaign was
interrupted. The partial order-fifteen worker produced no record. The complete
order-five CPU record is preserved, labeled `protocol_eligible=false`, and
excluded from every decision and table. It is not deleted, promoted, or used
as a replacement observation.

Before resumption, the orchestrator is corrected to dispatch only CUDA for
configurations carrying role `E1_scale`. No formulation, registered shape,
threshold, timing rule, or eligible endpoint changes.

## Fresh-process cache-state correction

The first C1 execution revealed that a fresh Python process does not imply an
empty TorchInductor disk cache. Its JIT lanes inherited the cache populated by
the preceding core and AOT campaigns. Those observations remain useful and
are labelled **fresh process, prepared persistent cache**. They may not be
described as clean-cache compilation.

Before interpreting the C1 result, the harness was extended with a
prospectively fixed `isolated` cache policy. Every repetition receives a newly
created empty `TORCHINDUCTOR_CACHE_DIR`, which is removed after that child
process exits. The same frozen orders, dimensions, sizes, eligible lanes,
three repetitions, mathematics, and correctness decisions are used. AOT
packages remain prepared artifacts by definition; running them under the
isolated environment checks that their deployment endpoint does not depend on
the prior JIT cache.

This correction adds no favorable point, changes no tolerance, and removes no
observation. It supplies the uncached deployment endpoint required to keep
uncached JIT, cached JIT, and prepared AOT distinct.

## Reference-health audit rule

The frozen protocol requires a finite, conservative CPU eager output before a
performance result is interpreted. The device worker admitted some individual
characteristic outputs whose own conservation check passed even though the
CPU reference for that mathematical configuration had failed its conservation
precondition. Raw timings are preserved, but the deterministic analysis marks
every lane in such a configuration ineligible. This is enforcement of the
frozen gate, not a post-observation threshold change.
