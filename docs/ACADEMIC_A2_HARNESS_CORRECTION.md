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
