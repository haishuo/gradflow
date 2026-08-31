#!/usr/bin/env python3
"""Run the frozen Academic A4 second-machine contract on a SLURM compute node."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


TAG = "academic-v0.1.0-rc2"
TAG_COMMIT = "c5e8ab81ef5b33a2138b2db33afc538398b6f57f"
ORDERS = (5, 11, 15)
DTYPES = ("float32", "float64")
DEVICES = ("cpu", "cuda")
WORKERS = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def parse_last_json(text: str) -> Any | None:
    for line in reversed(text.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def run_command(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    raw: Path,
    environment: dict[str, str] | None = None,
    timeout: int = 7200,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        error = None
    except subprocess.TimeoutExpired as exc:
        returncode = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        error = f"timeout after {timeout} seconds"
    duration = time.perf_counter() - started
    stdout_path = raw / f"{name}.stdout"
    stderr_path = raw / f"{name}.stderr"
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    return {
        "name": name,
        "command": [str(item) for item in command],
        "command_shell_display": shlex.join(str(item) for item in command),
        "returncode": returncode,
        "duration_seconds": duration,
        "timeout_seconds": timeout,
        "error": error,
        "stdout": str(stdout_path.relative_to(raw.parent)),
        "stderr": str(stderr_path.relative_to(raw.parent)),
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "last_json": parse_last_json(stdout),
    }


def capture_text(command: list[str], cwd: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=False, timeout=60
        )
    except FileNotFoundError as error:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def environment_record(repo: Path, execution_context: str) -> dict[str, Any]:
    import torch

    commands: dict[str, list[str]] = {
        "lscpu": ["lscpu"],
        "nvidia_smi": ["nvidia-smi", "-q"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
    }
    if execution_context == "unity_slurm":
        commands["slurm_job"] = [
            "scontrol",
            "show",
            "job",
            os.environ.get("SLURM_JOB_ID", ""),
        ]
    return {
        "captured_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_commit": getattr(torch.version, "git_version", None),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "gpu_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None
        ),
        "execution_context": execution_context,
        "slurm": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("SLURM_")
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "git_commit": capture_text(["git", "rev-parse", "HEAD"], repo)["stdout"].strip(),
        "git_status": capture_text(["git", "status", "--porcelain=v1"], repo)["stdout"],
        "commands": {name: capture_text(command, repo) for name, command in commands.items()},
    }


def lane_median(record: dict[str, Any], lane: str) -> float | None:
    device = record["device"]
    if device == "cpu":
        candidates = []
        for threads in ("1", "6"):
            timing = record["cpu"][threads]["resident_timing"]["lanes"].get(lane)
            if timing is not None:
                candidates.append(timing["median"])
        return min(candidates) if candidates else None
    timing = record["cuda"]["resident_timing"]["lanes"].get(lane)
    return timing["median"] if timing is not None else None


def analyze_a2(workers: list[dict[str, Any]]) -> dict[str, Any]:
    cells = []
    graph_failures = []
    admission_failures = []
    for order in ORDERS:
        for dtype in DTYPES:
            cell: dict[str, Any] = {"order": order, "dtype": dtype, "devices": {}}
            for device in DEVICES:
                selected = [
                    worker
                    for worker in workers
                    if worker["order"] == order
                    and worker["dtype"] == dtype
                    and worker["device"] == device
                ]
                records = [worker["record"] for worker in selected if worker["record"]]
                for index, record in enumerate(records):
                    compiled = record["correctness"]["compiled"]
                    graph = compiled["graph"]
                    if graph["unique_graphs"] != 1 or graph["graph_break_count"] != 0:
                        graph_failures.append(
                            {"order": order, "dtype": dtype, "device": device, "worker": index}
                        )
                    for lane in ("eager", "compiled"):
                        if not record["correctness"][lane]["admitted"]:
                            admission_failures.append(
                                {
                                    "order": order,
                                    "dtype": dtype,
                                    "device": device,
                                    "worker": index,
                                    "lane": lane,
                                }
                            )
                    if device == "cpu":
                        for threads, thread_record in record["cpu"].items():
                            for lane in ("eager", "compiled"):
                                correctness = thread_record["correctness"].get(lane)
                                if correctness is not None and not correctness["admitted"]:
                                    admission_failures.append(
                                        {
                                            "order": order,
                                            "dtype": dtype,
                                            "device": device,
                                            "worker": index,
                                            "lane": lane,
                                            "threads": int(threads),
                                        }
                                    )
                lane_summary = {}
                for lane in ("eager", "compiled"):
                    values = [lane_median(record, lane) for record in records]
                    values = [value for value in values if value is not None]
                    lane_summary[lane] = {
                        "worker_medians_ms": values,
                        "median_of_worker_medians_ms": statistics.median(values)
                        if values
                        else None,
                    }
                cell["devices"][device] = {
                    "workers_expected": WORKERS,
                    "workers_parsed": len(records),
                    "lanes": lane_summary,
                }
            cpu_values = [
                cell["devices"]["cpu"]["lanes"][lane]["median_of_worker_medians_ms"]
                for lane in ("eager", "compiled")
            ]
            cuda_values = [
                cell["devices"]["cuda"]["lanes"][lane]["median_of_worker_medians_ms"]
                for lane in ("eager", "compiled")
            ]
            cpu_values = [value for value in cpu_values if value is not None]
            cuda_values = [value for value in cuda_values if value is not None]
            if cpu_values and cuda_values:
                cell["fastest_cuda_over_fastest_cpu"] = min(cuda_values) / min(cpu_values)
            else:
                cell["fastest_cuda_over_fastest_cpu"] = None
            cells.append(cell)
    useful_binary32 = any(
        cell["dtype"] == "float32"
        and cell["fastest_cuda_over_fastest_cpu"] is not None
        and cell["fastest_cuda_over_fastest_cpu"] < 0.95
        for cell in cells
    )
    return {
        "cells": cells,
        "graph_failures": graph_failures,
        "admission_failures": admission_failures,
        "all_expected_workers_parsed": (
            len(workers) == len(ORDERS) * len(DTYPES) * len(DEVICES) * WORKERS
            and all(worker["record"] is not None for worker in workers)
        ),
        "all_compiled_graphs_one_with_zero_breaks": not graph_failures,
        "materially_useful_binary32_cuda_observed": useful_binary32,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execution-context",
        choices=("unity_slurm", "standalone"),
        default="unity_slurm",
    )
    parser.add_argument(
        "--workspace-contract",
        default="/work/pi_zchen2_umassd_edu/hshu",
    )
    parser.add_argument(
        "--protocol",
        default="docs/ACADEMIC_A4_PROTOCOL.md",
    )
    arguments = parser.parse_args()
    repo = arguments.repo.resolve()
    output = arguments.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()

    document: dict[str, Any] = {
        "schema": "gradflow-academic-a4-second-machine-v1",
        "created_utc": utc_now(),
        "status": "running",
        "source_tag": TAG,
        "source_commit": TAG_COMMIT,
        "protocol": arguments.protocol,
        "workspace_contract": arguments.workspace_contract,
        "environment": environment_record(repo, arguments.execution_context),
        "sentinels": [],
        "a2_workers": [],
    }
    write_json(output / "second_machine.json", document)

    if document["environment"]["git_commit"] != TAG_COMMIT:
        raise SystemExit("replication checkout is not the frozen rc2 commit")
    if document["environment"]["git_status"]:
        raise SystemExit("replication checkout is dirty")

    python = sys.executable
    sentinel_commands = (
        ("pytest", [python, "-m", "pytest"]),
        (
            "verify_a1",
            [
                python,
                "experiments/academic_a1/verify_a1.py",
                "experiments/academic_a1/evidence/a1_20260830",
            ],
        ),
        (
            "verify_a2",
            [
                python,
                "experiments/academic_a2/verify_a2.py",
                "experiments/academic_a2/evidence/a2_20260830",
            ],
        ),
        (
            "verify_a3",
            [
                python,
                "experiments/academic_a3/verify_a3.py",
                "experiments/academic_a3/evidence/a3_20260830",
            ],
        ),
        (
            "verify_u5",
            [
                python,
                "experiments/academic_u5/verify_u5.py",
                "experiments/academic_u5/evidence/u5_20260831",
            ],
        ),
        (
            "verify_a4_rc2",
            [
                python,
                "experiments/academic_a4/verify_a4_rc2.py",
                "experiments/academic_a4/evidence/a4_rc2_20260831",
                "--ref",
                TAG,
            ],
        ),
    )
    for name, command in sentinel_commands:
        document["sentinels"].append(
            run_command(name=name, command=command, cwd=repo, raw=raw, timeout=3600)
        )
        write_json(output / "second_machine.json", document)

    a1_output = output / "a1_numerical_limits.json"
    document["a1"] = run_command(
        name="a1_numerical_limits",
        command=[
            python,
            "experiments/academic_a1/run_numerical_limits.py",
            "--output",
            str(a1_output),
        ],
        cwd=repo,
        raw=raw,
        timeout=7200,
    )
    document["a1"]["output_sha256"] = sha256(a1_output) if a1_output.is_file() else None
    write_json(output / "second_machine.json", document)

    a3_output = output / "a3_campaign.json"
    document["a3"] = run_command(
        name="a3_campaign",
        command=[python, "experiments/academic_a3/run_campaign.py", "--output", str(a3_output)],
        cwd=repo,
        raw=raw,
        timeout=7200,
    )
    a3_record = json.loads(a3_output.read_text()) if a3_output.is_file() else None
    document["a3"]["record"] = a3_record
    document["a3"]["output_sha256"] = sha256(a3_output) if a3_output.is_file() else None
    write_json(output / "second_machine.json", document)

    base_environment = os.environ.copy()
    base_environment["PYTHONPATH"] = os.pathsep.join((str(repo / "src"), str(repo)))
    for order in ORDERS:
        for dtype in DTYPES:
            for device in DEVICES:
                for repetition in range(WORKERS):
                    name = f"a2_o{order}_{dtype}_{device}_w{repetition}"
                    cache = tempfile.mkdtemp(prefix=f"{name}-", dir=os.environ.get("TMPDIR"))
                    environment = base_environment.copy()
                    environment["TORCHINDUCTOR_CACHE_DIR"] = cache
                    result = run_command(
                        name=name,
                        command=[
                            python,
                            "experiments/academic_a2/benchmark_worker.py",
                            "--subject",
                            "scalar",
                            "--order",
                            str(order),
                            "--dtype",
                            dtype,
                            "--dimensions",
                            "3",
                            "--size",
                            "64",
                            "--device",
                            device,
                        ],
                        cwd=repo,
                        raw=raw,
                        environment=environment,
                        timeout=3600,
                    )
                    document["a2_workers"].append(
                        {
                            "order": order,
                            "dtype": dtype,
                            "device": device,
                            "repetition": repetition,
                            "execution": result,
                            "record": result["last_json"],
                        }
                    )
                    write_json(output / "second_machine.json", document)

    document["a2_analysis"] = analyze_a2(document["a2_workers"])
    sentinel_pass = all(item["returncode"] == 0 for item in document["sentinels"])
    a1_pass = document["a1"]["returncode"] == 0
    a3_pass = bool(
        a3_record
        and a3_record.get("complete")
        and a3_record.get("derivative_gate", {}).get("registered_window_passed")
        and a3_record.get("inverse_gate", {}).get("passed")
        and all(
            a3_record.get("benchmarks", {})
            .get(device, {})
            .get("record", {})
            .get("compiled", {})
            .get("admitted")
            for device in DEVICES
        )
    )
    a2 = document["a2_analysis"]
    core_pass = (
        sentinel_pass
        and a1_pass
        and a3_pass
        and a2["all_expected_workers_parsed"]
        and a2["all_compiled_graphs_one_with_zero_breaks"]
        and a2["materially_useful_binary32_cuda_observed"]
    )
    if core_pass and not a2["admission_failures"]:
        status = "pass"
    elif core_pass:
        status = "pass_with_limitations"
    else:
        status = "fail_needs_investigation"
    document["qualification"] = {
        "sentinels_passed": sentinel_pass,
        "a1_completed": a1_pass,
        "a3_agreement_passed": a3_pass,
        "a2_worker_surface_complete": a2["all_expected_workers_parsed"],
        "a2_graph_contract_passed": a2["all_compiled_graphs_one_with_zero_breaks"],
        "binary32_cuda_materially_useful": a2["materially_useful_binary32_cuda_observed"],
        "admission_failures": a2["admission_failures"],
    }
    document["status"] = status
    document["completed_utc"] = utc_now()
    write_json(output / "second_machine.json", document)

    checksums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha256(path)}  {path.relative_to(output)}")
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(json.dumps({"status": status, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
