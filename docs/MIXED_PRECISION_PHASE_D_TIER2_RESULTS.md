# Phase-D Tier-2 characteristic-Euler mixed-precision results

Execution date: 2026-08-27 UTC.

Status: **the complete frozen Tier-2 matrix executed and independently
verified; its top-level decision is `FAIL`**.

## Main result

The scalar mixed-precision seam does not transfer unchanged through
Roe-characteristic Euler reconstruction at every order.

Two WENO-5 assignments passed every applicable Euler gate at their inherited
`engineering` tolerance:

- binary32 smoothness indicators with binary64 weight formation and
  normalization; and
- binary32 smoothness indicators plus binary32 unnormalized weight formation,
  with binary64 normalization.

No mixed assignment passed the complete `tight` contract at orders 7 through
15. Binary32 unnormalized-weight formation alone also narrowly failed the
WENO-5 `tight` local gate. Consequently, no mixed policy is presently eligible
for a higher-order characteristic-Euler performance claim or production
default. The all-binary64 path remains the qualified default.

This is a useful negative boundary rather than a numerical-run failure. Every
recorded integration completed, every shock and compiler/device gate passed,
and the result artifact verified. The rejections arise from deliberately
sensitive local and gradient tests.

## Gate summary

| Gate | Passed | Total | Interpretation |
|---|---:|---:|---|
| Local Euler parity | 8 | 24 | All six FP64 controls and two engineering WENO-5 policies passed |
| One-period smooth integration | 12 | 12 | Every policy/order remained stable and close to FP64 |
| Differentiation | 8 | 12 | FP32 indicators failed at orders 11 and 15 |
| Compiler and CPU/CUDA agreement | 12 | 12 | One graph, zero breaks, and declared parity passed |
| Sod and Shu--Osher shocks | 24 | 24 | Every policy/order passed independent and FP64-terminal gates |
| Static numerical-loop audit | 1 | 1 | No forbidden transfer or custom-kernel token found |

The local gate contains four policies at each order 5, 7, 9, 11, 13, and 15.
The other numerical gates use the frozen representative orders 5, 11, and 15.

## Local sensitivity boundary

The strict local contract requires normalized maximum error at most `1e-5`
and normalized RMS error at most `1e-6`. The mixed-policy worst errors were:

| Order | Indicators FP32 | Weight formation FP32 | Both FP32 |
|---:|---:|---:|---:|
| 5 | `4.237e-6 / 1.141e-6` | `4.237e-6 / 1.254e-6` | `4.237e-6 / 1.102e-6` |
| 7 | `5.296e-6 / 1.634e-6` | `4.237e-6 / 1.437e-6` | `5.296e-6 / 1.426e-6` |
| 9 | `5.296e-6 / 1.446e-6` | `3.177e-6 / 1.276e-6` | `5.296e-6 / 1.370e-6` |
| 11 | `7.176e-6 / 1.520e-6` | `5.296e-6 / 1.641e-6` | `7.175e-6 / 1.718e-6` |
| 13 | `1.386e-5 / 1.703e-6` | `5.296e-6 / 1.645e-6` | `1.387e-5 / 1.704e-6` |
| 15 | `7.573e-5 / 7.041e-6` | `6.355e-6 / 2.074e-6` | `7.574e-5 / 7.042e-6` |

Each entry is `maximum normalized Linf / maximum normalized RMS` across the
frozen local cases. WENO-5 indicator-containing policies pass because their
scalar result already assigned them the looser `engineering` class
(`5e-4 / 1e-4`). Weight-formation-only WENO-5 retains `tight` status and
misses only its RMS threshold; that status was not relaxed after observation.

The largest discrepancies occur in the `1e-7` near-equilibrium cases, where
the derivative signal is normalized by its physical amplitude rather than by
the order-one background state. This test exposes loss that ordinary smooth
wave errors can conceal.

## Differentiation boundary

Binary32 unnormalized-weight formation retained directional finite-difference
agreement at all representative orders. Its normalized gradient differences
from FP64 were below `1.3e-11` in Linf.

Policies with binary32 indicators passed the WENO-5 engineering gradient gate,
but failed the fixed `2e-5` directional finite-difference criterion at higher
order:

| Order | Indicator-policy relative directional error |
|---:|---:|
| 5 | approximately `1.14e-5` |
| 11 | approximately `1.11e-4` |
| 15 | approximately `2.32e-4` |

At order 11 the mixed autograd gradient itself remained within approximately
`8.43e-7` normalized L2 of the FP64 gradient; at order 15 it remained within
`7.77e-5`. Nevertheless, the same mixed function's autograd directional
derivative did not agree with centered finite differences. Binary32 indicator
quantization changes the discrete function while the cast's autograd rule
passes derivatives through, so gradient closeness to FP64 is not sufficient
to qualify the differentiated computation.

## Time integration and shocks

All one-period entropy-wave integrations completed 1,781 SSP-RK3 steps with
positive density and pressure. Worst mixed terminal parity was far inside the
declared terminal thresholds; for example, the combined-policy normalized
Linf discrepancy was approximately `3.31e-9` at WENO-5, `1.89e-11` at
WENO-11, and order `1e-13` at WENO-15.

All 18 mixed shock runs completed at 800 points and passed both the independent
Phase-A oracle and the committed all-FP64 Phase-B terminal comparison. The
largest mixed-versus-FP64 terminal discrepancy was the WENO-15 Shu--Osher
indicator-only Linf value `1.304e-4`, below the `2e-3` tight bound. Positivity
was retained without clipping, retry, adaptive epsilon, or scheme
substitution.

These results show that the tested demotions can produce visually and
integrally close forward solutions even when they fail the stricter
small-signal or differentiability contract. Under GradFlow's
correctness-before-performance rule, passing shocks cannot override those
failures.

## Compiler and device result

Every representative order and policy captured exactly one CPU and one CUDA
graph with zero graph breaks under `torch.compile(fullgraph=True)`. All
compiled/eager and CPU/CUDA comparisons passed their frozen bounds. The
canonical Euler path contains no hidden host/device transfer in its numerical
loop; explicit dtype conversion remains confined to the audited WENO precision
helper.

## Consequence and next research boundary

No higher-order Euler performance campaign is authorized by this result. The
next correctness-first question is mathematical, not a threshold adjustment:
why near-equilibrium characteristic signals amplify the scalar-qualified seam,
and whether an independently motivated precision rule or reformulation can
preserve strict local and differentiable behavior. Any new policy requires a
new protocol committed before execution.

The two engineering-qualified WENO-5 policies are eligible for a separately
frozen WENO-5 Euler timing study, but such a study would support only an
engineering-tolerance WENO-5 claim. It cannot be generalized to WENO-7 through
WENO-15.

## Claim boundary

This experiment does not select a production default, prove mixed-precision
safety for arbitrary Euler states, qualify Navier--Stokes, measure performance,
modify DVEB, or establish a publication claim. It also does not show that FP64
is necessary everywhere. It identifies where these three particular scalar-
qualified assignments do and do not satisfy a precommitted characteristic-
Euler contract on this implementation and test family.

## Reproducibility

Machine-readable record:
`experiments/mixed_precision/results/phase_d_tier2_20260827/qualification.json`.

- Protocol commit: `4f92b57`
- Numerical runner/source commit:
  `46a11a7286a913f8f8a4c4a0dbd83886361b42ef`
- Qualification JSON SHA-256:
  `34bd715d9801b284dcc21c431b3c6eb435a52e63026182eef924fb35311afe88`
- Checksum manifest SHA-256:
  `73640b781b57732f980d6cc525401cf5971f8d79d03dd62b859fac7a6e5321e8`
- Environment: Python 3.11.13, PyTorch `2.9.0.dev20250705+cu128`, CUDA 12.8,
  NVIDIA GeForce RTX 5070 Ti, one CPU thread
- Result files: one aggregate JSON, 18 compressed mixed shock arrays, and a
  manifest hashing every result file

The independent verifier recomputes every recorded gate decision from raw
metrics, validates the frozen matrices and thresholds, checks source and
control hashes, and derives the top-level `FAIL` independently.
