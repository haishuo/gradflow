# Forced-target DVEB ABI bakeoff results

Status: **completed; all frozen correctness and timing gates passed**.

Historical scope note: this report concerns portable ABI v1. The later frozen
device-ABI v2 addendum made DVEB eligible for E4 and supersedes only the v1
“unsupported” conclusion there. See `DVEB_DEVICE_ABI_E4_RESULTS.md`; E1--E3
below remain unchanged.

All primary values are medians of the frozen 30-observation design. 
Fresh application, first call, warm call, and resident execution remain separate.

## Result in one paragraph

DVEB validates its existence for this matched workload, but not as one universal winner. Its CPU-12 target wins small first-`run` calls; its CUDA target wins the larger standalone applications and every warm repeated-call point. At `N=128`, ten steps, DVEB-CUDA takes 1220.866 ms as a standalone invocation versus 30504.386 ms for Fortran, and 161.599 ms per warm CPU-in/CPU-out run versus 718.068 ms for AOT PyTorch. AOT PyTorch is the best supported resident-state route, while DVEB ABI v1 cannot enter that endpoint because it does not accept device-resident state.

## Correctness and capacity gates

All ten lanes passed full-array comparison at four frozen points. The worst pairwise float32 error was `1.430511475e-06` against the frozen `2e-05` bound; duplicated endpoints were exact.

Every lane completed the uncounted capacity pilot through `N=160` for one and ten steps. At `N=160`, ten steps, the reported peak CUDA allocations were 0.312 GiB for DVEB-CUDA, 5.092 GiB for AOT PyTorch, and 8.082 GiB for eager PyTorch. These are implementation-reported allocator peaks, not whole-system GPU-memory measurements, and capacity-pilot timings are not used as performance claims.

## E1: fresh standalone application (process creation through validated CPU result)

| N | Steps | Winner | Median ms | Competitive lanes |
|---:|---:|---|---:|---|
| 8 | 1 | fortran | 2.497 | fortran |
| 16 | 1 | fortran | 8.922 | fortran |
| 32 | 1 | fortran | 54.559 | fortran |
| 64 | 1 | fortran | 396.510 | fortran |
| 96 | 1 | dveb-cuda | 1052.343 | dveb-cuda |
| 128 | 1 | dveb-cuda | 1157.027 | dveb-cuda |
| 16 | 10 | fortran | 75.281 | fortran |
| 32 | 10 | fortran | 515.970 | fortran |
| 64 | 10 | dveb-cuda | 1033.667 | dveb-cuda |
| 128 | 10 | dveb-cuda | 1220.866 | dveb-cuda |

## E2: first `run` in a fresh worker (ready CPU input through returned CPU result)

| N | Steps | Winner | Median ms | Competitive lanes |
|---:|---:|---|---:|---|
| 8 | 1 | dveb-cpu12 | 1.890 | dveb-cpu6, dveb-cpu12 |
| 16 | 1 | dveb-cpu12 | 4.348 | dveb-cpu12 |
| 32 | 1 | dveb-cpu12 | 23.860 | dveb-cpu12 |
| 64 | 1 | aot-inductor | 88.027 | aot-inductor |
| 96 | 1 | aot-inductor | 112.574 | aot-inductor |
| 128 | 1 | aot-inductor | 158.324 | aot-inductor |
| 16 | 10 | dveb-cpu12 | 31.552 | dveb-cpu12 |
| 32 | 10 | aot-inductor | 87.679 | aot-inductor |
| 64 | 10 | dveb-cuda | 134.972 | dveb-cuda |
| 128 | 10 | dveb-cuda | 272.158 | dveb-cuda |

## E3: warm repeated `run` (ready CPU input through newly returned CPU result)

| N | Steps | Winner | Median ms | Competitive lanes |
|---:|---:|---|---:|---|
| 8 | 1 | dveb-cuda | 0.348 | dveb-cuda, aot-inductor |
| 16 | 1 | dveb-cuda | 0.411 | dveb-cuda, aot-inductor |
| 32 | 1 | dveb-cuda | 0.901 | dveb-cuda |
| 64 | 1 | dveb-cuda | 5.271 | dveb-cuda |
| 96 | 1 | dveb-cuda | 15.845 | dveb-cuda |
| 128 | 1 | dveb-cuda | 78.899 | dveb-cuda |
| 16 | 10 | dveb-cuda | 1.432 | dveb-cuda |
| 32 | 10 | dveb-cuda | 3.298 | dveb-cuda |
| 64 | 10 | dveb-cuda | 16.711 | dveb-cuda |
| 128 | 10 | dveb-cuda | 161.599 | dveb-cuda |

## E4: resident PyTorch numerical execution (synchronized, no H2D/D2H)

| N | Steps | Winner | Median ms | Competitive lanes |
|---:|---:|---|---:|---|
| 8 | 1 | aot-inductor | 0.327 | aot-inductor |
| 16 | 1 | aot-inductor | 0.385 | aot-inductor |
| 32 | 1 | aot-inductor | 0.971 | persistent-compile, aot-inductor |
| 64 | 1 | aot-inductor | 8.570 | persistent-compile, aot-inductor |
| 96 | 1 | aot-inductor | 29.174 | persistent-compile, aot-inductor |
| 128 | 1 | aot-inductor | 69.791 | persistent-compile, aot-inductor |
| 16 | 10 | aot-inductor | 3.692 | aot-inductor |
| 32 | 10 | aot-inductor | 9.608 | persistent-compile, aot-inductor |
| 64 | 10 | aot-inductor | 85.395 | persistent-compile, aot-inductor |
| 128 | 10 | aot-inductor | 698.143 | persistent-compile, aot-inductor |

## DVEB useful-region rule

A DVEB target qualifies only when it is within 10% of the winner at two adjacent counted sizes or at the same size in both step strata.

- `E1:dveb-cpu6`: not qualified
- `E1:dveb-cpu12`: not qualified
- `E1:dveb-cuda`: QUALIFIED
- `E2:dveb-cpu6`: not qualified
- `E2:dveb-cpu12`: QUALIFIED
- `E2:dveb-cuda`: QUALIFIED
- `E3:dveb-cpu6`: not qualified
- `E3:dveb-cpu12`: not qualified
- `E3:dveb-cuda`: QUALIFIED

## Cold-cache diagnostic

These smaller diagnostic samples are excluded from winner classification.

| N | Steps | Empty-cache first compile+call ms | Pristine AOT call ms | Ratio |
|---:|---:|---:|---:|---:|
| 64 | 1 | 47111.3 | 87.6 | 537.8x |
| 128 | 1 | 45045.3 | 158.8 | 283.6x |
| 128 | 10 | 47804.8 | 789.9 | 60.5x |

## Interpretation

- **Standalone program:** Fortran wins through `N=64` for one step and through `N=32` for ten steps. DVEB-CUDA wins every larger counted E1 point.
- **First `run` with a ready CPU state:** DVEB CPU-12 wins the small region; AOT PyTorch wins the middle region; DVEB-CUDA wins the two large ten-step points.
- **Warm CPU-in/CPU-out service:** DVEB-CUDA wins all counted E3 points. Its small-grid margins over AOT are narrow, but it becomes materially faster as work grows.
- **Device-resident throughput:** AOT and persistent compiled PyTorch are effectively tied at most larger E4 points. AOT has the lower median at every counted point. DVEB ABI v1 is unsupported here, not measured as a loser.
- **Cold compilation:** An empty TorchInductor cache costs about 45–48 seconds inside the first call at the diagnostic points. AOT removes that compilation event but still pays Python/package launch overhead in E1.

## Boundaries

`DVEB Auto` was not tested or calibrated. The protocol used forced targets only. DVEB ABI v1 has no public resident-state interface and therefore does not participate in E4. Internal CUDA execution timing remains diagnostic only. The capacity pilot establishes only that all lanes ran through `N=160`; it does not establish a hardware maximum. No arbitrary-order, Navier--Stokes, real-time aerospace, cross-machine, or publication claim follows from this campaign.

Prepared-manifest SHA-256: `92718ccb229c40345aebfa5d20c7e2e0918daef2f60c3615ffe5bbef711b336f`.
