# Academic A4 Moody replication protocol

Status: **frozen prospectively before Moody scientific execution**.

Date: 2026-08-31 (UTC)

## Purpose

This protocol instantiates the frozen Academic A4 second-machine contract on
Moody, a physically distinct standalone Linux workstation. It preserves the
Unity scientific surface without pretending that Moody is a SLURM system. A
successful run can close the second-machine scientific gate; it cannot close
the independent CFD/prior-art audit or the unresolved redistribution review.

The governing order remains:

> correctness > performance > convenience

## Machine and location contract

All persistent files live below `/mnt/projects`. The scientific checkout is a
clean detached checkout of annotated tag `academic-v0.1.0-rc2`, commit
`c5e8ab81ef5b33a2138b2db33afc538398b6f57f`. The frozen run directory is
`/mnt/projects/gradflow-a4-moody-20260831`.

Hardware identity was observed before execution solely to qualify the machine
and is not a result-dependent selection: AMD Ryzen 7 7700 (8 cores, 16
threads), NVIDIA GeForce RTX 4070 SUPER (12 GiB, compute capability 8.9), and
Ubuntu Linux. Exact CPU, OS, Python, PyTorch, accelerator, driver, CUDA
runtime, memory, package, and thread identities are captured in the evidence.

## Environment contract

The environment is isolated under the frozen run directory and uses Python
3.10 or newer, stable PyTorch 2.13.0 from the CUDA 12.6 wheel index, and the
rc2 project test dependencies. The NVIDIA driver is host-provided. A missing
environment prerequisite or failed installation is retained; another PyTorch
version is not silently substituted.

No system CUDA toolkit, CMake, Fortran compiler, DVEB binary, OpenSBLI binary,
or Forge compiler cache is required for this A4 PyTorch replication. Those
systems remain separate comparison strata.

## Frozen execution surface

The standalone controller runs exactly the scientific matrix frozen for
Unity:

1. the complete rc2 test suite and A1/A2/A3/U5/A4 offline sentinels;
2. the complete A1 numerical-limit campaign;
3. the complete A3 CPU/CUDA inverse and gradient campaign; and
4. three fresh A2 workers for every combination of scalar WENO-JS order
   5/11/15, binary32/binary64, and CPU/CUDA at `64^3`.

Each A2 worker retains five warmups, 20 randomized eager/compiled pairs,
correctness admission, graph records, compiler failures, CUDA telemetry,
resident timings, and the existing transfer slice. Every worker receives an
empty private TorchInductor cache. CPU observations use the existing one- and
six-thread contract.

No tolerance, numerical source, warmup count, repetition count, compiler
option, size, dtype, lane, or acceptance criterion may change after execution
begins. Installation and environment-construction time is recorded separately
and is not a scientific runtime observation.

## Decision and evidence contract

The unchanged A4 qualitative decision requires clean sentinels, completed A1
and A3 gates, all 36 A2 workers, one graph with zero breaks for every admitted
compiled worker, at least one binary32 `64^3` CUDA result below 0.95 times the
fastest CPU median, and no unreported admission failure. The final status is
`pass`, `pass_with_limitations`, or `fail_needs_investigation` under the
existing controller rules.

Exact Forge or Unity timings are not replication targets. All raw output,
errors, commands, return codes, durations, environment identity, qualification
decisions, and SHA-256 hashes are retained. Binary64 results are explicitly
conditioned on Moody's RTX 4070 SUPER. No result from this run is generalized
to data-center accelerators.

Stop after this one frozen run completes or is retained as a failure. Do not
tune or selectively rerun it. A new attempt requires a new prospective run
identity and a written reason.
