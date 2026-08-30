#!/usr/bin/env python3
"""Run the frozen U4-C C3 transfer and prepared-launch endpoints."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_performance import (
    ADAPTER,
    INSTRUMENTER,
    MAKEFILE,
    MAXIMUM_LIMIT,
    OPENSBLI_COMMIT,
    OPENSBLI_TREE,
    OPS_COMMIT,
    OPS_TREE,
    PATCH,
    RMS_LIMIT,
    ROOT,
    SAMPLES,
    SEED,
    THERMAL_STOP_C,
    WARMUPS,
    comparison,
    digest,
    ensure_patch,
    execute,
    git_value,
    parse_external_samples,
    parse_json,
    require,
    stats,
    telemetry,
    write_checksums,
)


HERE = Path(__file__).resolve().parent
GRADFLOW_WORKER = HERE / "gradflow_worker.py"
AOT_BUILDER = HERE / "build_aot.py"
AOT_WORKER = HERE / "aot_launch_worker.py"
SIZE = 8192


def external_checksum(stdout: str) -> float:
    marker = "U4C_CHECKSUM "
    for line in stdout.splitlines():
        if marker in line:
            return float(line.split(marker, 1)[1].strip())
    raise RuntimeError("OpenSBLI launch did not emit a checksum")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensbli-root", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--sympy-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    opensbli = args.opensbli_root.resolve()
    ops = args.ops_root.resolve()
    cuda = args.cuda_root.resolve()
    hdf5 = args.hdf5_root.resolve()
    work = args.work_root.resolve()
    packages = args.package_dir.resolve()
    evidence = args.evidence_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    packages.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    raw = evidence / "raw"
    raw.mkdir()
    commands = [
        f"(cd {shlex.quote(str(Path.cwd()))} && "
        f"{shlex.join([sys.executable, *sys.argv])})"
    ]

    if (
        git_value(opensbli, "rev-parse", "HEAD") != OPENSBLI_COMMIT
        or git_value(opensbli, "rev-parse", "HEAD^{tree}") != OPENSBLI_TREE
    ):
        raise RuntimeError("OpenSBLI source differs from frozen revision")
    if (
        git_value(ops, "rev-parse", "HEAD") != OPS_COMMIT
        or git_value(ops, "rev-parse", "HEAD^{tree}") != OPS_TREE
    ):
        raise RuntimeError("OPS source differs from frozen revision")

    environment = os.environ.copy()
    native_flags = (
        "-Xcompiler=-fPIC -O3 -g -std=c++11 "
        "-gencode arch=compute_120,code=sm_120"
    )
    environment.update(
        {
            "OPS_INSTALL_PATH": str(ops / "ops"),
            "OPS_COMPILER": "gnu",
            "CUDA_INSTALL_PATH": str(cuda),
            "HDF5_INSTALL_PATH": str(hdf5),
            "MPICXX": "g++",
            "NVCCFLAGS": native_flags,
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "PYTHONPATH": os.pathsep.join(
                [str(args.sympy_root.resolve()), str(opensbli)]
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(
                [
                    str(cuda / "lib64"),
                    str(hdf5 / "lib"),
                    environment.get("LD_LIBRARY_PATH", ""),
                ]
            ),
        }
    )
    hdf5_link = " ".join(
        [
            f"-L{cuda / 'lib64'}",
            "-lops_hdf5_seq",
            f"-L{hdf5 / 'lib'}",
            "-lhdf5_hl -lhdf5 -lz",
        ]
    )
    ensure_patch(opensbli, commands, environment)

    c2_arrays = (
        ROOT
        / "experiments"
        / "academic_u4c"
        / "evidence"
        / "u4c_c2_20260830"
        / "qualification_arrays"
    )
    state_path = c2_arrays / f"n{SIZE}_state.bin"
    canonical_path = c2_arrays / f"n{SIZE}_canonical.bin"
    canonical = np.fromfile(canonical_path, dtype=np.float64)
    if canonical.shape != (SIZE,):
        raise RuntimeError("C2 canonical record is unavailable")

    application = work / f"n{SIZE}"
    application.mkdir()
    shutil.copy2(MAKEFILE, application / "Makefile")
    preparation: dict[str, Any] = {}
    generated, elapsed = execute(
        [sys.executable, str(ADAPTER), "--case", "state_a", "--size", str(SIZE)],
        cwd=application,
        env=environment,
        commands=commands,
    )
    require(generated, "C3 OpenSBLI generation")
    preparation["opensbli_symbolic_generation_seconds"] = elapsed
    instrumented, elapsed = execute(
        [
            sys.executable,
            str(INSTRUMENTER),
            "opensbli.cpp",
            "opensbliblock00_kernels.h",
        ],
        cwd=application,
        env=environment,
        commands=commands,
    )
    require(instrumented, "C3 OpenSBLI instrumentation")
    preparation["u4c_instrumentation_seconds"] = elapsed
    built, elapsed = execute(
        [
            "make",
            "opensbli_cuda",
            "NVCC_FLAG_SET=1",
            f"NVCCFLAGS={native_flags}",
            "CXXFLAGS=-O3 -fPIC -Wall -g -std=c++11 -DU4C_CUDA",
            f"HDF5_LIB_SEQ={hdf5_link}",
        ],
        cwd=application,
        env=environment,
        commands=commands,
    )
    require(built, "C3 OpenSBLI CUDA build")
    preparation["ops_translation_and_cuda_build_seconds"] = elapsed
    (raw / "opensbli_build.stdout").write_text(built.stdout)
    (raw / "opensbli_build.stderr").write_text(built.stderr)

    grad_env = os.environ.copy()
    grad_env["PYTHONPATH"] = str(ROOT / "src")
    grad_env["TORCHINDUCTOR_CACHE_DIR"] = str(work / "torchinductor_cache")

    transfer_records: dict[str, Any] = {}
    transfer_order = ["opensbli", "gradflow"]
    random.Random(SEED).shuffle(transfer_order)
    for lane in transfer_order:
        before = telemetry()
        if before["temperature_c"] >= THERMAL_STOP_C:
            raise RuntimeError("thermal stop before C3 transfer worker")
        output_path = evidence / f"transfer_{lane}_rhs.bin"
        if lane == "opensbli":
            lane_env = environment | {
                "U4C_MODE": "transfer",
                "U4C_WARMUPS": str(WARMUPS),
                "U4C_SAMPLES": str(SAMPLES),
                "U4C_STATE_PATH": str(state_path),
                "U4C_RHS_PATH": str(output_path),
                "OPS_BLOCK_SIZE_X": "256",
            }
            completed, wall = execute(
                ["./opensbli_cuda"],
                cwd=application,
                env=lane_env,
                commands=commands,
            )
            require(completed, "C3 OpenSBLI transfer")
            record = {
                "samples_milliseconds": parse_external_samples(completed.stdout),
                "process_wall_seconds": wall,
            }
        else:
            completed, wall = execute(
                [
                    sys.executable,
                    str(GRADFLOW_WORKER),
                    "--size",
                    str(SIZE),
                    "--device",
                    "cuda",
                    "--mode",
                    "transfer",
                    "--input",
                    str(state_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                env=grad_env,
                commands=commands,
            )
            require(completed, "C3 GradFlow transfer")
            record = parse_json(completed.stdout)
            record["process_wall_seconds"] = wall
        after = telemetry()
        if after["temperature_c"] >= THERMAL_STOP_C:
            raise RuntimeError("thermal stop after C3 transfer worker")
        candidate = np.fromfile(output_path, dtype=np.float64)
        record["correctness"] = comparison(candidate, canonical)
        if not record["correctness"]["passed"]:
            raise RuntimeError(f"C3 {lane} transfer correctness failed")
        record["rhs_sha256"] = digest(output_path)
        record["telemetry_before"] = before
        record["telemetry_after"] = after
        record["stdout"] = f"raw/transfer_{lane}.stdout"
        record["stderr"] = f"raw/transfer_{lane}.stderr"
        (raw / f"transfer_{lane}.stdout").write_text(completed.stdout)
        (raw / f"transfer_{lane}.stderr").write_text(completed.stderr)
        transfer_records[lane] = record

    transfer_analysis = {
        lane: stats(record["samples_milliseconds"])
        for lane, record in transfer_records.items()
    }
    transfer_analysis["median_ratio_opensbli_over_gradflow"] = (
        transfer_analysis["opensbli"]["median"]
        / transfer_analysis["gradflow"]["median"]
    )

    package = packages / f"u4c_weno5_f64_n{SIZE}.pt2"
    build_record = evidence / "aot_build.json"
    aot_build, elapsed = execute(
        [
            sys.executable,
            str(AOT_BUILDER),
            "--size",
            str(SIZE),
            "--input",
            str(state_path),
            "--package",
            str(package),
            "--record",
            str(build_record),
        ],
        cwd=ROOT,
        env=grad_env,
        commands=commands,
    )
    preparation["aot_builder_process_seconds"] = elapsed
    (raw / "aot_build.stdout").write_text(aot_build.stdout)
    (raw / "aot_build.stderr").write_text(aot_build.stderr)
    aot_record = json.loads(build_record.read_text())

    aot_admission: dict[str, Any]
    if aot_build.returncode == 0 and aot_record["status"] == "complete":
        aot_output = evidence / "aot_qualification_rhs.bin"
        qualified, elapsed = execute(
            [
                sys.executable,
                str(AOT_WORKER),
                "--size",
                str(SIZE),
                "--input",
                str(state_path),
                "--package",
                str(package),
                "--output",
                str(aot_output),
            ],
            cwd=ROOT,
            env=grad_env,
            commands=commands,
        )
        (raw / "aot_qualification.stdout").write_text(qualified.stdout)
        (raw / "aot_qualification.stderr").write_text(qualified.stderr)
        require(qualified, "C3 AOT qualification")
        aot_candidate = np.fromfile(aot_output, dtype=np.float64)
        aot_admission = {
            "status": "qualified",
            "process_seconds": elapsed,
            "comparison": comparison(aot_candidate, canonical),
            "rhs_sha256": digest(aot_output),
            "worker": parse_json(qualified.stdout),
        }
        if not aot_admission["comparison"]["passed"]:
            raise RuntimeError("C3 AOT correctness gate failed")
    else:
        aot_admission = {
            "status": "not_implemented",
            "reason": "AOTInductor package build failed",
        }

    launch_records: dict[str, list[dict[str, Any]]] = {
        "opensbli": [],
        "gradflow_aot": [],
    }
    for repetition in range(3):
        lane_env = environment | {
            "U4C_MODE": "launch",
            "U4C_STATE_PATH": str(state_path),
            "OPS_BLOCK_SIZE_X": "256",
        }
        completed, elapsed = execute(
            ["./opensbli_cuda"],
            cwd=application,
            env=lane_env,
            commands=commands,
        )
        require(completed, f"C3 OpenSBLI launch {repetition}")
        checksum = external_checksum(completed.stdout)
        record = {
            "repetition": repetition,
            "parent_launch_to_answer_seconds": elapsed,
            "finite_checksum": bool(np.isfinite(checksum)),
            "checksum_float64": checksum,
        }
        launch_records["opensbli"].append(record)
        (raw / f"launch_opensbli_{repetition}.stdout").write_text(completed.stdout)
        (raw / f"launch_opensbli_{repetition}.stderr").write_text(completed.stderr)

    if aot_admission["status"] == "qualified":
        for repetition in range(3):
            completed, elapsed = execute(
                [
                    sys.executable,
                    str(AOT_WORKER),
                    "--size",
                    str(SIZE),
                    "--input",
                    str(state_path),
                    "--package",
                    str(package),
                ],
                cwd=ROOT,
                env=grad_env,
                commands=commands,
            )
            require(completed, f"C3 GradFlow AOT launch {repetition}")
            worker = parse_json(completed.stdout)
            record = {
                "repetition": repetition,
                "parent_launch_to_answer_seconds": elapsed,
                "worker": worker,
            }
            launch_records["gradflow_aot"].append(record)
            (raw / f"launch_gradflow_aot_{repetition}.stdout").write_text(
                completed.stdout
            )
            (raw / f"launch_gradflow_aot_{repetition}.stderr").write_text(
                completed.stderr
            )

    launch_analysis: dict[str, Any] = {}
    for lane, records in launch_records.items():
        if records:
            launch_analysis[lane] = stats(
                [record["parent_launch_to_answer_seconds"] for record in records]
            )
        else:
            launch_analysis[lane] = {"status": "not_implemented"}
    if launch_records["gradflow_aot"]:
        launch_analysis["median_ratio_opensbli_over_gradflow_aot"] = (
            launch_analysis["opensbli"]["median"]
            / launch_analysis["gradflow_aot"]["median"]
        )

    result = {
        "schema": "gradflow.academic_u4c.endpoints.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "size": SIZE,
        "reason_single_size": (
            "N=8192 was both the smallest and largest C2 correctness-admitted size"
        ),
        "bounds": {
            "maximum_normalized": MAXIMUM_LIMIT,
            "rms_normalized": RMS_LIMIT,
        },
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "preparation": preparation,
        "transfer": {
            "order": transfer_order,
            "warmups": WARMUPS,
            "samples": SAMPLES,
            "records": transfer_records,
            "analysis": transfer_analysis,
        },
        "aot_build": aot_record,
        "aot_package_retained_outside_repository": str(package),
        "aot_admission": aot_admission,
        "prepared_launch": {
            "repetitions": 3,
            "records": launch_records,
            "analysis": launch_analysis,
        },
        "artifacts": {
            "opensbli_executable_sha256": digest(application / "opensbli_cuda"),
            "generated_source_sha256": {
                name: digest(application / name)
                for name in (
                    "opensbli.cpp",
                    "opensbli_ops.cpp",
                    "opensbliblock00_kernels.h",
                )
            },
            "frozen_state_sha256": digest(state_path),
            "canonical_rhs_sha256": digest(canonical_path),
        },
    }
    (evidence / "endpoints.json").write_text(json.dumps(result, indent=2) + "\n")
    (evidence / "COMMANDS.txt").write_text("\n".join(commands) + "\n")
    write_checksums(evidence)
    print(json.dumps({"transfer": transfer_analysis, "launch": launch_analysis}, indent=2))


if __name__ == "__main__":
    main()
