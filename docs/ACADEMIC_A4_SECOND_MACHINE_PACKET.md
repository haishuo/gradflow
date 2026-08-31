# Academic A4 second-machine replication packet

Status: **ready to execute on a physically distinct machine**.

The prospective Unity/SLURM instantiation is frozen in
`docs/ACADEMIC_A4_UNITY_PROTOCOL.md`; its controller and submission files are
under `experiments/academic_a4/unity/`. It uses the designated persistent
workspace `/work/pi_zchen2_umassd_edu/hshu`, never the login staging directory.

## Minimum machine

A 64-bit Linux host with Python 3.10 or newer, enough RAM for the test suite,
and a supported PyTorch installation can reproduce the numerical sentinels.
The performance stratum should also have a CUDA-capable NVIDIA GPU and a
compiler toolchain supported by `torch.compile`. A data-center accelerator is
useful but not required.

## Setup

From a clean checkout of tag `academic-v0.1.0-rc2`, create an isolated
environment and install
the project:

```bash
python -m pip install -e '.[test]'
```

Record the exact commit, dirty status, CPU, operating system, Python, package
versions, GPU, driver, CUDA runtime, memory, and relevant thread settings.
Do not silently substitute binary32 for binary64 or simulate an unavailable
device.

## Required checks

Run:

```bash
python -m pytest
python experiments/academic_a1/verify_a1.py experiments/academic_a1/evidence/a1_20260830
python experiments/academic_a2/verify_a2.py experiments/academic_a2/evidence/a2_20260830
python experiments/academic_a3/verify_a3.py experiments/academic_a3/evidence/a3_20260830
python experiments/academic_u5/verify_u5.py experiments/academic_u5/evidence/u5_20260831
python experiments/academic_a4/verify_a4_rc2.py experiments/academic_a4/evidence/a4_rc2_20260831 --ref academic-v0.1.0-rc2
```

Then execute the prospectively frozen numerical/performance sentinels in
`docs/ACADEMIC_A4_PROTOCOL.md`, using stable PyTorch rather than attempting to
recreate the historical nightly wheel. Preserve raw stdout/stderr, exit codes,
wall-clock durations, environment identity, and SHA-256 hashes. Timing workers
must be fresh isolated processes and retain failed or excluded points.

## Interpretation

Exact Forge timings are not the replication target. The target is independent
numerical agreement and the qualitative conclusions declared in the A4
protocol. Report all ratios as properties of the replication machine. A
negative or divergent result is scientifically useful and must not be deleted.

The completed packet should state either `pass`, `pass_with_limitations`, or
`fail_needs_investigation`, with a reason for every limitation.
