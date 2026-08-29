# FD/FV Euler Phase 6E protocol amendment

Status: frozen after the preserved initial AOT qualification and before the
corrected device-loop requalification.

Amendment date: 2026-08-29 UTC.

## Preserved observations

The initial AOT campaign is immutable at
`experiments/fd_fv_euler/results/phase_6e_aot_20260829/`.

All four host-controlled packages built and were numerically eligible, but
loading each package under a pristine TorchInductor cache compiled six generic
C++ runtime helper probes. The original protocol requires no runtime
compilation. That gate is not weakened: the host-controlled lane remains
ineligible and is not rerun or timed in Phase 6E.

All four device-loop exports failed before AOTInductor backend lowering. The
candidate initialized minimum density and minimum pressure with the same
scalar tensor object, so two `torch.while_loop` carried inputs aliased. This
violates the documented structured-control-flow input contract. It is an
illegal harness construction, not evidence about whether AOTInductor can lower
a valid adaptive loop.

## Authorized correction

This amendment permits exactly one device-loop legality correction:

- create separate, non-aliasing initial scalar tensors for minimum density and
  minimum pressure;
- leave every equation, operation, threshold, state, CFL rule, stage,
  diagnostic, package policy, and performance gate unchanged;
- rerun only the four device-loop build and qualification cells under the new
  series identity `phase_6e_device_r1_20260829`; and
- preserve both attempts and their source identities.

This is not optimization and authorizes no performance timing by itself. The
corrected lane reaches timing only if every original Lane-C gate passes. No
further implementation correction is authorized inside Phase 6E. A second
failure is the final bounded Phase-6E device-loop result.
