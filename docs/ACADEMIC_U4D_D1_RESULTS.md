# Academic U4-D D1 qualification results

Status: **passed — `all_six_lanes_qualified`**.

Date: 2026-08-30 (UTC)

This is the prospective correctness gate from the committed U4-D protocol.
No U4-D comparative performance result was inspected before this gate passed.

## Result

The pinned DVEB compiler generated its ordinary CPU and CUDA implementations
from the unmodified scalar `weno5.dveb` source under `DVEB_CONTRACT=fma`.
All DVEB, OpenSBLI, and GradFlow lanes produced finite, conservative float64
arrays and passed the frozen U4-C normalized-error bounds at `N=8192`.

| lane | maximum normalized error | RMS normalized error |
|---|---:|---:|
| DVEB CPU | `1.5700811045372667e-15` | `2.4532517258394792e-17` |
| DVEB CUDA | `2.5121297672596267e-14` | `3.615157031501409e-15` |
| OpenSBLI CPU | `1.130458395266832e-12` | `2.4567458830409808e-14` |
| OpenSBLI CUDA | `8.415634720319749e-13` | `2.1026683551910345e-14` |
| GradFlow CPU | `6.280324418149067e-15` | `2.6611324051688867e-15` |
| GradFlow CUDA | `2.5121297672596267e-14` | `4.26528140859316e-15` |

DVEB CPU versus CUDA also passed, with maximum normalized difference
`2.5121297672596226e-14` and RMS normalized difference
`3.614990549587643e-15`. Both GradFlow lanes retained one full graph and zero
graph breaks.

## Interpretation

DVEB is now correctness-admitted to the exact U4-C scalar comparison. This is
strong evidence that the existing DVEB scalar source expresses the intended
Gottlieb-equivalent WENO-JS5 algebra; it is not yet performance evidence.
Unlike OpenSBLI, DVEB remains an internal compiler control rather than an
independent external baseline. The retained adapter exercises genuine
compiler-generated launchers but does not establish a public scalar DVEB ABI.

## Evidence

The frozen record is in
`experiments/academic_u4d/evidence/u4d_d1_20260830/`. It contains the six full
RHS arrays, source and generated-source hashes, build and execution logs,
commands, environment metadata, and a SHA-256 manifest. Run
`python experiments/academic_u4d/verify_qualification.py` for offline
verification.
