# Academic A4 Unity second-machine result

Status: **complete negative portability observation; A4 second-machine gate
not closed**.

Date: 2026-08-31 (UTC)

## Frozen execution

Unity executed the prospectively frozen protocol in
`ACADEMIC_A4_UNITY_PROTOCOL.md` from a clean detached checkout of
`academic-v0.1.0-rc2`, commit
`c5e8ab81ef5b33a2138b2db33afc538398b6f57f`. SLURM job `63844884` completed
normally on node `gypsum-gpu005`; the controller ran for 10,814.998 seconds
and retained the final status `fail_needs_investigation`.

The allocated host exposed two Intel Xeon E5-2620 v3 sockets and one Tesla M40
24 GB accelerator. The software stratum was Python 3.12.3, stable PyTorch
2.13.0+cu126 at commit `cf30153c4c131c8164ee7798e5022d810682e2cb`,
CUDA runtime 12.6, and NVIDIA driver 580.173.02. The M40 reports compute
capability 5.2.

All controller output and its internal SHA-256 manifest verify under
`experiments/academic_a4/evidence/unity_20260831/`.

## What passed

- The complete A1 numerical-limit campaign returned zero.
- The A3 derivative window and inverse-recovery gates passed.
- A3 CPU eager and compiled execution passed.
- A3 CUDA eager objective and gradient agreed with the binary64 reference;
  objective absolute error was `3.3881e-19` and gradient absolute error was
  `1.3824e-18`.
- Every A2 CPU eager and compiled lane and every A2 CUDA eager lane passed the
  registered numerical admission.
- All five historical A1/A2/A3/U5/A4-rc2 offline evidence verifiers returned
  zero.
- The imported controller evidence has 108 files and its original inner
  checksum manifest verifies completely.

## Why the frozen qualification failed

PyTorch reported the same error for all 18 A2 compiled-CUDA workers and for
the A3 compiled-CUDA lane:

```text
GPUTooOldForTriton: ... Triton only supports devices of CUDA Capability >= 7.0,
but your device is of CUDA capability 5.2
```

This is a backend-support boundary, not a WENO numerical disagreement. With
compiled CUDA unavailable, the frozen graph criterion failed, A3's aggregate
CPU/CUDA compiled criterion failed, and no binary32 CUDA lane met the
material-usefulness rule.

At `64^3`, the available CUDA-eager median lost to the fastest compiled-CPU
median in all six cells:

| Order | Dtype | compiled CPU (ms) | eager CUDA (ms) | CUDA/CPU |
| ---: | :---: | ---: | ---: | ---: |
| 5 | float32 | 5.552685 | 9.988736 | 1.798902 |
| 5 | float64 | 13.018495 | 13.812320 | 1.060977 |
| 11 | float32 | 29.578675 | 43.705936 | 1.477616 |
| 11 | float64 | 44.493414 | 71.854671 | 1.614951 |
| 15 | float32 | 63.008689 | 99.287041 | 1.575767 |
| 15 | float64 | 92.725984 | 156.664108 | 1.689538 |

The complete test-suite sentinel recorded 344 passes, 12 declared skips, and
11 failures. Seven failures were direct compiled-CUDA tests rejected by the
same M40/Triton boundary. Four were replication-packet infrastructure limits:
the tag-only bundle did not contain the older rc1 tag required by two legacy
A4 tests, and two Phase-6E checks expected machine-specific AOT packages that
were deliberately not transported. These four failures do not change the A1,
A2, or A3 numerical observations, but they correctly prevent a claim that the
entire portable test-suite sentinel passed.

## Scientific interpretation

Unity is a valid negative performance-portability stratum. It establishes
that "CUDA visible" is insufficient: the selected high-level compiler stack
also imposes a GPU-architecture floor. It does not test whether the Forge
phase diagram replicates on a second modern accelerator, because the frozen
compiled lanes could not exist on the allocated M40.

The result therefore does not close A4 and does not contradict Forge. A
separately frozen run on Moody's RTX 4070 SUPER (compute capability 8.9) is the
appropriate modern-GPU replication. No Unity constraint was changed and no
selective rerun was made after observing the result.
