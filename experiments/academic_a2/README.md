# Academic A2 performance matrix

The frozen protocol is `docs/ACADEMIC_A2_PROTOCOL.md`. The harness records
each mathematical configuration in an isolated worker, gates every lane
against CPU eager output, and only then collects timings.

No canonical numerical source is modified by this experiment.

The primary Forge campaign is preserved under
`evidence/a2_20260830/`. `deployment.json` is the prepared persistent-cache
fresh-process slice; `deployment_isolated_cache.json` gives every JIT process
a separate empty TorchInductor cache. `analysis.json` is derived
deterministically from the raw records. The interpreted result is
`docs/ACADEMIC_A2_RESULTS.md`.
