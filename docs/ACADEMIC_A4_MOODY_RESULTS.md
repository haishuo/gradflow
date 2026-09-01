# Academic A4 Moody second-machine result

Status: **modern-GPU scientific replication complete; formal aggregate
sentinel remains limited by four transport-packet assumptions**.

Date: 2026-08-31 (UTC)

## Frozen identity

The prospectively frozen Moody campaign executed the unmodified Academic A4
surface from annotated tag `academic-v0.1.0-rc2`, commit
`c5e8ab81ef5b33a2138b2db33afc538398b6f57f`, on a physically distinct
standalone machine:

- AMD Ryzen 7 7700, eight cores and 16 hardware threads;
- NVIDIA GeForce RTX 4070 SUPER, 12 GiB, compute capability 8.9;
- Python 3.12.3;
- stable PyTorch 2.13.0+cu126, commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`;
- CUDA runtime 12.6 and NVIDIA driver 570.133.20; and
- initial `OMP`, `MKL`, and `OPENBLAS` environment variables set to one; each
  CPU A2 worker subsequently timed PyTorch intra-op thread counts one and six.

The run began at `2026-08-31T20:26:05.991426+00:00` and completed at
`2026-08-31T21:43:12.469543+00:00`, an elapsed 4,626.478 seconds. Exact
environment output, package versions, raw worker records, and hashes are under
`experiments/academic_a4/evidence/moody_20260831/`.

## Scientific qualification

All 36 independently launched A2 workers completed. Every expected record was
parsed, no correctness-admission failure occurred, and every compiled case
captured as one graph with zero graph breaks. The A1 numerical-limit campaign
completed. The A3 derivative and inverse gates passed on CPU and CUDA, and
both eager and compiled CUDA objective/gradient evaluations were admitted.
The dedicated A1, A2, A3, U5, and rc2 A4 offline verifiers all returned zero.
All 95 files named by the campaign `SHA256SUMS` verify after import.

At scalar `64^3`, the fastest admitted lanes produced the following median of
three fresh worker medians. The selected CPU lane used six PyTorch intra-op
threads in every row; six threads beat one thread in every contributing eager
and compiled worker. Ratios are properties of Moody and are not pooled with
Forge observations.

| Order | dtype | fastest CPU (ms) | fastest CUDA (ms) | CPU/CUDA |
| ---: | --- | ---: | ---: | ---: |
| 5 | binary32 | 2.241363 | 0.243200 | 9.216 |
| 11 | binary32 | 9.315638 | 0.991760 | 9.393 |
| 15 | binary32 | 15.563876 | 2.650576 | 5.872 |
| 5 | binary64 | 5.507698 | 1.702880 | 3.234 |
| 11 | binary64 | 19.673676 | 5.955424 | 3.303 |
| 15 | binary64 | 29.556987 | 20.225535 | 1.461 |

Thus, the frozen binary32 proposition replicated on a second modern CUDA
machine: resident CUDA was materially faster than the best measured six-thread
CPU lane at every tested order. The full CPU/CUDA ordering did not reproduce.
On the primary RTX 5070 Ti system, binary64 WENO-15 favored CPU by 1.083 times;
on Moody's RTX 4070 SUPER, it favored CUDA by 1.461 times. This reversal is
positive evidence that binary64 backend choice is machine- and toolchain-stack
conditional. It does not isolate a GPU-architecture cause, predict A100/H100
behavior, or establish universal GPU superiority.

Lane identity also changed. For the matched orders 5, 11, and 15, the primary
system selected eager CUDA for binary64 orders 5 and 15, whereas Moody selected
compiled CUDA for all six order/dtype cells. Both systems selected six-thread
compiled CPU for these cells. Because the systems also differ in GPU, CPU,
driver, and CUDA wheel/runtime, the observation demonstrates stack-specific
selection rather than a causal TorchInductor architecture effect.

The A3 differentiation observation also transferred: compiled order-11
binary64 forward execution was 2.554 ms on CPU and 2.981 ms on CUDA; compiled
objective-plus-gradient execution was 5.253 ms on CPU and 7.846 ms on CUDA.
Both compiled graphs contained one graph and zero breaks. This is a
portability observation, not evidence that differentiation is preferable to
the derivative-free baseline.

## Why the controller status is not `pass`

The immutable controller correctly recorded `fail_needs_investigation`
because its aggregate `pytest` sentinel returned one: 351 tests passed, 12
were skipped, and four failed. None of those four failures was a numerical,
CUDA, graph-capture, or benchmark-worker failure:

1. two A4 tests invoked the earlier `academic-v0.1.0-rc1` tag, which the
   deliberately tag-restricted rc2 transport bundle did not include; and
2. two FD/FV Phase 6E tests required machine-specific AOT packages that the
   second-machine packet deliberately did not transport.

The dedicated rc2 verifier passed. The run is not relabeled post hoc, and the
four failures remain visible. Under the letter of the prospectively frozen
protocol, the all-sentinels formal gate is therefore limited. Scientifically,
the complete modern-GPU A1/A2/A3 replication surface is usable and supports a
bounded second-machine portability claim. No selective worker rerun or timing
substitution was performed.

## Boundaries

This result:

- replicates the registered ordinary-PyTorch numerical and graph-capture
  behavior, plus the bounded binary32 CUDA advantage, on one physically
  distinct modern consumer-GPU system;
- records a binary64 WENO-15 CPU/CUDA winner reversal rather than claiming
  full performance-ordering reproduction;
- does not replicate the DVEB or OpenSBLI/OPS matched-control lanes;
- does not qualify data-center binary64 hardware;
- does not close the independent numerical-CFD/prior-art audit;
- does not resolve reference redistribution or project licensing; and
- does not support a universal backend, hardware, or real-time-CFD claim.
