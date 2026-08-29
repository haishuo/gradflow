#!/usr/bin/env python3
"""Run the conditionally admitted Phase-6F prepared process-entry bakeoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/FD_FV_PHASE_6F_PROTOCOL.md"
PROTOCOL_COMMIT = "c3a19eb"
WORKER = Path(__file__).with_name("phase6f_worker.py")
QUALIFICATION = (
    ROOT / "experiments/fd_fv_euler/results/phase_6f_qualification_20260829/qualification.json"
)
QUALIFICATION_VERIFY = Path(__file__).with_name("verify_phase6f_qualification.py")
PHASE6D = ROOT / "experiments/fd_fv_euler/results/phase_6d_20260829/benchmark.json"
PHASE6D_VERIFY = Path(__file__).with_name("verify_phase6d.py")
LANE_A = ROOT / "experiments/fd_fv_euler/results/phase_6e_20260829"
HOST_RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6e_aot_20260829"
TENSOR_RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6e_device_r1_20260829"
ENDPOINTS = ("cuda_jit", "aot_host", "aot_tensor")
PROBLEMS = ("sod", "shu_osher")
METHODS = ("fd", "fv")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()


def verify(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        (sys.executable, str(path)), cwd=ROOT, capture_output=True, text=True
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0,
    }


def package_path(endpoint: str, problem: str, method: str) -> Path | None:
    if endpoint == "cuda_jit":
        return None
    base = HOST_RESULTS if endpoint == "aot_host" else TENSOR_RESULTS
    prefix = "host" if endpoint == "aot_host" else "device"
    record = json.loads(
        (base / "build_records" / f"{prefix}_{problem}_{method}.json").read_text()
    )
    path = Path(record["package_path"])
    if sha256(path) != record["package_sha256"]:
        raise RuntimeError(f"package identity changed: {path}")
    return path


def authority_paths(problem: str, method: str) -> tuple[Path, Path]:
    stem = f"{problem}_{method}_cpu_eager_r0"
    return LANE_A / "arrays" / f"{stem}.npy", LANE_A / "raw" / f"{stem}.json"


def worker_environment(cache: Path) -> dict[str, str]:
    result = os.environ.copy()
    result["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    result["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    return result


def decision(ratios: list[float]) -> str:
    if len(ratios) != 3:
        return "unresolved"
    if all(value < 1.0 / 1.05 for value in ratios):
        return "confirmed_numerator_win"
    if all(value > 1.05 for value in ratios):
        return "confirmed_denominator_win"
    if all(1.0 / 1.05 <= value <= 1.05 for value in ratios):
        return "practical_equivalence_5_percent"
    return "unresolved"


def selected_cpu_records() -> dict[tuple[str, str], list[dict[str, Any]]]:
    phase6d = json.loads(PHASE6D.read_text())
    result = {}
    for problem in PROBLEMS:
        mode = "eager" if problem == "sod" else "compiled"
        for method in METHODS:
            selected = sorted(
                (
                    item
                    for item in phase6d["shock_records"]
                    if item["problem"] == problem
                    and item["method"] == method
                    and item["cells"] == 800
                    and item["device"] == "cpu"
                    and item["mode"] == mode
                    and item["eligible"]
                ),
                key=lambda item: item["replicate"],
            )
            if len(selected) != 3:
                raise RuntimeError(f"missing Phase 6D CPU comparator: {problem}/{method}")
            result[(problem, method)] = selected
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError("refusing existing Phase 6F performance output")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6F performance requires a clean tree")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"), cwd=ROOT
    ).returncode:
        raise RuntimeError("frozen Phase 6F protocol is not an ancestor")
    prerequisites = [verify(QUALIFICATION_VERIFY), verify(PHASE6D_VERIFY)]
    if not all(item["passed"] for item in prerequisites):
        raise RuntimeError("Phase 6F performance prerequisites failed")
    qualification = json.loads(QUALIFICATION.read_text())
    if not qualification["lane_status"]["performance_admitted"]:
        raise RuntimeError("Phase 6F qualification did not admit performance")
    cache_archive = Path(qualification["prepared_cache"]["archive"])
    if sha256(cache_archive) != qualification["prepared_cache"]["archive_sha256"]:
        raise RuntimeError("prepared cache archive identity changed")

    output.mkdir(parents=True)
    (output / "raw").mkdir()
    (output / "arrays").mkdir()
    records = []
    for endpoint in ENDPOINTS:
        for problem in PROBLEMS:
            for method in METHODS:
                package = package_path(endpoint, problem, method)
                authority_array, authority_record = authority_paths(problem, method)
                for replicate in range(3):
                    stem = f"{endpoint}_{problem}_{method}_r{replicate}"
                    record_path = output / "raw" / f"{stem}.json"
                    array_path = output / "arrays" / f"{stem}.npy"
                    with tempfile.TemporaryDirectory(
                        prefix=f"gradflow-phase6f-timing-{stem}-"
                    ) as temp:
                        cache = Path(temp) / "cache"
                        cache.mkdir()
                        with tarfile.open(cache_archive, "r:gz") as archive:
                            archive.extractall(cache, filter="data")
                        command = [
                            sys.executable, str(WORKER), "--action", "solve",
                            "--endpoint", endpoint, "--problem", problem,
                            "--method", method, "--replicate", str(replicate),
                            "--authority-array", str(authority_array),
                            "--authority-record", str(authority_record),
                            "--array-output", str(array_path),
                        ]
                        if package is not None:
                            command.extend(("--package", str(package)))
                        command.extend(("--output", str(record_path)))
                        started = time.perf_counter_ns()
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            env=worker_environment(cache),
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        wall = (time.perf_counter_ns() - started) * 1.0e-9
                    record = (
                        json.loads(record_path.read_text())
                        if record_path.exists()
                        else {
                            "status": "failed",
                            "eligible": False,
                            "error_type": "WorkerProcessFailure",
                            "error": "worker produced no record",
                        }
                    )
                    record.update(
                        {
                            "endpoint": endpoint,
                            "problem": problem,
                            "method": method,
                            "replicate": replicate,
                            "process_launch_to_exit_seconds": wall,
                            "worker_returncode": completed.returncode,
                            "worker_stdout": completed.stdout,
                            "worker_stderr": completed.stderr,
                            "prepared_cache_restoration_timed": False,
                        }
                    )
                    record["eligible"] = bool(
                        record.get("eligible")
                        and completed.returncode == 0
                        and math.isfinite(wall)
                        and wall > 0.0
                    )
                    record_path.write_text(
                        json.dumps(record, indent=2, sort_keys=True) + "\n"
                    )
                    records.append(record)
                    print(
                        f"timed {stem}: {wall:.6f}s eligible={record['eligible']}",
                        flush=True,
                    )

    cpu = selected_cpu_records()
    summaries = []
    for problem in PROBLEMS:
        for method in METHODS:
            cuda = {
                endpoint: sorted(
                    (
                        item
                        for item in records
                        if item["endpoint"] == endpoint
                        and item["problem"] == problem
                        and item["method"] == method
                        and item["eligible"]
                    ),
                    key=lambda item: item["replicate"],
                )
                for endpoint in ENDPOINTS
            }
            cpu_selected = cpu[(problem, method)]
            durations = {
                "cpu": [item["process_launch_to_exit_seconds"] for item in cpu_selected],
                **{
                    endpoint: [item["process_launch_to_exit_seconds"] for item in items]
                    for endpoint, items in cuda.items()
                },
            }
            comparisons = {}
            for numerator in ENDPOINTS:
                for denominator in ("cpu", *[x for x in ENDPOINTS if x != numerator]):
                    ratios = [
                        left / right
                        for left, right in zip(durations[numerator], durations[denominator])
                    ]
                    comparisons[f"{numerator}_over_{denominator}"] = {
                        "paired_ratios": ratios,
                        "median_ratio": (
                            statistics.median(durations[numerator])
                            / statistics.median(durations[denominator])
                            if len(durations[numerator]) == len(durations[denominator]) == 3
                            else None
                        ),
                        "decision": decision(ratios),
                    }
            summaries.append(
                {
                    "problem": problem,
                    "method": method,
                    "cpu_endpoint": "cpu_eager" if problem == "sod" else "cpu_compiled",
                    "durations_seconds": durations,
                    "comparisons": comparisons,
                }
            )

    package_preparation = []
    for endpoint, base, prefix in (
        ("aot_host", HOST_RESULTS, "host"),
        ("aot_tensor", TENSOR_RESULTS, "device"),
    ):
        for problem in PROBLEMS:
            for method in METHODS:
                record = json.loads(
                    (base / "build_records" / f"{prefix}_{problem}_{method}.json").read_text()
                )
                package_preparation.append(
                    {
                        "endpoint": endpoint,
                        "problem": problem,
                        "method": method,
                        "export_seconds": record.get("export_seconds"),
                        "package_seconds": record.get("package_seconds"),
                        "package_bytes": record.get("package_bytes"),
                        "package_sha256": record.get("package_sha256"),
                    }
                )
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_euler_phase_6f_performance",
        "measurement_date": "2026-08-29",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": False,
        "prerequisites": prerequisites,
        "qualification_sha256": sha256(QUALIFICATION),
        "phase6d_sha256": sha256(PHASE6D),
        "prepared_cache_archive_sha256": sha256(cache_archive),
        "matrix": {
            "problems": list(PROBLEMS),
            "methods": list(METHODS),
            "endpoints": list(ENDPOINTS),
            "cells": 800,
            "replicates": 3,
            "workers": len(records),
            "dtype": "float64",
        },
        "records": records,
        "summaries": summaries,
        "package_preparation": package_preparation,
        "runtime_cache_preparation": qualification["preparation"],
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PROTOCOL, WORKER, Path(__file__))
        },
        "all_workers_eligible": len(records) == 36 and all(
            item["eligible"] for item in records
        ),
        "performance_measurements_collected": True,
        "production_sources_modified": False,
        "dveb_modified": False,
        "publication_claim": False,
    }
    aggregate = output / "benchmark.json"
    aggregate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    files = [
        aggregate,
        *sorted((output / "raw").glob("*.json")),
        *sorted((output / "arrays").glob("*.npy")),
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )
    print(f"Phase 6F performance workers eligible={payload['all_workers_eligible']}")


if __name__ == "__main__":
    main()
