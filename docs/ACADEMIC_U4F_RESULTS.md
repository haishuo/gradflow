# Academic U4-F batched-line regime-map results

Status: **complete on the frozen surface**.

Date: 2026-08-31 (UTC)

U4-F prospectively tested whether the PyTorch/TorchInductor backend recovers
resident forward competitiveness against automatically scheduled DVEB when
independent `N=8192` scalar binary64 WENO-JS5 lines are batched. The batch axis
is an ensemble/pencil axis, not a multidimensional PDE.

## Correctness and execution admission

All DVEB CPU/CUDA lanes and all PyTorch CUDA lanes passed the full-array
pointwise, RMS, finiteness, shape, and per-row conservation gates. Every
PyTorch CUDA cell compiled as one graph with zero graph breaks.

PyTorch/TorchInductor CPU passed at `B=1` but failed to compile at every
batched shape `B=4--1024`. The retained exception is an internal Inductor CPU
scheduler assertion in `simplify_and_reorder`; no eager fallback, graph break,
alternate representation, or relaxed gate was substituted. Under the
pre-campaign infrastructure amendment, those five CPU cells are explicit
execution exclusions while independently admitted CUDA cells remain valid.

The largest admitted normalized CUDA errors remained far inside the frozen
bounds: at most `3.77e-14` maximum and `4.27e-15` RMS versus limits `5e-11`
and `5e-12`. The maximum batch contained 8,388,608 reconstructed points.

## Resident CUDA result

Each lane retained 120 observations: six independent workers, five warmups,
and 20 samples per worker. Times below are medians of worker medians. The ratio
is paired `PyTorch/DVEB`; a resolved decision requires both a 5% material
effect and a bootstrap interval excluding one.

| batch | points | DVEB (ms) | PyTorch (ms) | PyTorch/DVEB (95% interval) | decision |
|---:|---:|---:|---:|---:|:---|
| 1 | 8,192 | `0.009072` | `0.032128` | `3.54646` (`3.50264`, `3.63993`) | DVEB |
| 4 | 32,768 | `0.021456` | `0.035904` | `1.67400` (`1.63821`, `1.72435`) | DVEB |
| 16 | 131,072 | `0.068544` | `0.071096` | `1.03808` (`1.03665`, `1.04215`) | unresolved |
| 64 | 524,288 | `0.242624` | `0.209992` | `0.86504` (`0.86318`, `0.86788`) | PyTorch |
| 256 | 2,097,152 | `0.935848` | `0.755728` | `0.80756` (`0.80579`, `0.80959`) | PyTorch |
| 1024 | 8,388,608 | `3.715840` | `3.052200` | `0.82140` (`0.82056`, `0.82232`) | PyTorch |

The observed transition is bounded but clear. DVEB won at batches one and
four; batch 16 was only 3.8% apart and therefore intentionally unresolved;
PyTorch won at batches 64, 256, and 1024. Equivalently, PyTorch was about
`1.16x`, `1.24x`, and `1.22x` faster than DVEB at those three largest batches.
No crossover is interpolated between the tested points.

At batch 256, PyTorch reached the largest observed median throughput,
approximately `2.775e9` reconstructed points/s, versus DVEB's `2.241e9`.
At batch 1024 the corresponding throughputs were `2.748e9` and `2.258e9`
points/s. These are operator-throughput measurements, not time-step or full-
solver rates.

## CPU result

At batch one, the prospective rerun reproduced U4-E's ranking:

| batch | DVEB (ms) | PyTorch (ms) | PyTorch/DVEB (95% interval) | decision |
|---:|---:|---:|---:|:---|
| 1 | `0.0599395` | `0.09559925` | `1.59512` (`1.59203`, `1.60616`) | DVEB |

No comparative CPU timing is reported for batches 4--1024 because the
PyTorch/TorchInductor CPU lane did not execute. DVEB's successful
qualification is not converted into a one-lane performance claim.

## Automatic DVEB schedule

DVEB selected materialization and two numerical stages everywhere. Its CUDA
block changed with the submitted surface:

| batch | CUDA block | scratch bytes |
|---:|---:|---:|
| 1 | 32 | 65,584 |
| 4 | 128 | 262,336 |
| 16 | 256 | 1,049,344 |
| 64 | 256 | 4,197,376 |
| 256 | 256 | 16,789,504 |
| 1024 | 256 | 67,158,016 |

Every CUDA query and run reported no internal synchronization. The GradFlow
adapter supplied shape and caller-owned storage/stream facts but did not force
a loop, block, or reuse policy.

## Interpretation

U4-F answers the PyTorch “redemption” question positively and conditionally.
The U4-E loss was real, but it did not generalize even across the same
mathematics on the same GPU. Once 64 or more independent lines were presented
together, TorchInductor's fused ordinary-PyTorch program became the resolved
resident winner over DVEB's automatically scheduled native path.

The result also validates GradFlow's backend-neutral premise: neither source
language nor compiler should be selected globally. A data-driven policy would
choose DVEB for the small admitted CUDA batches, treat batch 16 as unresolved,
and choose PyTorch for the three largest tested batches. This study supplies
research evidence for such a future selector; it does not implement one.

The negative CPU result is independently important. On the dated PyTorch
development build, adding a leading batch dimension exposed an Inductor
backend failure even though CUDA compiled the same source as one graph. This
strengthens, rather than replaces, the need for the stable-release U5 rerun.
It does not prove that current stable PyTorch has the same defect.

U4-F does not compare batched OpenSBLI, transfers, launch-to-answer, higher
orders, true multidimensional fluxes, systems, gradients, float32, multiple
CPU threads, another machine, or datacenter FP64 hardware. DVEB remains the
winner in some regimes and PyTorch in others; no universal backend claim is
admitted.

## Evidence

The repository-wide CUDA-visible regression suite passed after the campaign:
355 tests passed, 12 expected tests were skipped, and one existing PyTorch
deprecation warning was reported in 53.72 seconds.

Frozen evidence is under
`experiments/academic_u4f/evidence/u4f_20260831/`. It retains all qualification
diagnostics and hashes, 1,680 resident observations across admitted cells, raw
logs, randomized orders, graph records, automatic DVEB schedules, telemetry,
commands, and a SHA-256 manifest. Large arrays were compared in full but kept
ephemeral under the prospective protocol; deterministic input construction and
full input/canonical/output digests are retained.

Run:

```bash
PYTHONPATH=src python3 experiments/academic_u4f/verify_campaign.py
```

for offline verification.
