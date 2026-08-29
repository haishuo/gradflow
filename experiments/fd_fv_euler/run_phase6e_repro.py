#!/usr/bin/env python3
"""Run the frozen Phase-6E retained-array reproducibility qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch

from experiments.fd_fv_euler.run_phase6c import admission as phase6c_admission


PROTOCOL_COMMIT = "af90466"
PHASE6D_TIMING_COMMIT = "7952c9fabbca5114994d457b563cc907c477db4e"
PROTOCOL = ROOT / "docs/FD_FV_PHASE_6E_PROTOCOL.md"
PHASE6D_VERIFY = ROOT / "experiments/fd_fv_euler/verify_phase6d.py"
PHASE6C_RECORD = (
    ROOT / "experiments/fd_fv_euler/results/phase_6c_20260829/benchmark.json"
)
WORKER = Path(__file__).with_name("phase6e_repro_worker.py")
PRODUCTION_SOURCES = (
    ROOT / "src/gradflow/euler1d.py",
    ROOT / "src/gradflow/euler1d_fv.py",
    ROOT / "experiments/fd_fv_euler/phase6b_problem.py",
    ROOT / "experiments/fd_fv_euler/phase6c_problem.py",
)
METHODS = ("fd", "fv")
PROBLEMS = ("sod", "shu_osher")
CUDA_REPLICATES = 5
ROUNDING_FACTOR = 128.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()


def git_blob_sha256(commit: str, path: Path) -> str:
    blob = subprocess.check_output(
        ("git", "show", f"{commit}:{path.relative_to(ROOT)}"), cwd=ROOT
    )
    return hashlib.sha256(blob).hexdigest()


def comparison(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    steps: int,
    reference_name: str,
    actual_name: str,
) -> dict[str, Any]:
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return {
            "reference": reference_name,
            "actual": actual_name,
            "shape_dtype_match": False,
            "passed": False,
        }
    difference = np.abs(actual - reference)
    flat_index = int(np.argmax(difference))
    index = tuple(int(value) for value in np.unravel_index(flat_index, difference.shape))
    epsilon = float(np.finfo(reference.dtype).eps)
    steps = max(1, steps)
    scale = max(1.0, float(np.max(np.abs(reference))))
    relative_bound = ROUNDING_FACTOR * epsilon * steps
    absolute_bound = relative_bound * scale
    reference_l1 = max(float(np.sum(np.abs(reference))), float(np.finfo(reference.dtype).tiny))
    reference_l2 = max(float(np.linalg.norm(reference.ravel())), float(np.finfo(reference.dtype).tiny))
    normalized_l1 = float(np.sum(difference)) / reference_l1
    normalized_l2 = float(np.linalg.norm(difference.ravel())) / reference_l2
    maximum = float(difference[index])
    passed = (
        maximum <= absolute_bound
        and normalized_l1 <= relative_bound
        and normalized_l2 <= relative_bound
    )
    return {
        "reference": reference_name,
        "actual": actual_name,
        "shape_dtype_match": True,
        "exact_equal": bool(np.array_equal(reference, actual)),
        "equal_elements": int(np.count_nonzero(reference == actual)),
        "total_elements": int(reference.size),
        "maximum_absolute_difference": maximum,
        "mean_absolute_difference": float(np.mean(difference)),
        "rms_difference": float(np.sqrt(np.mean(np.square(difference)))),
        "normalized_l1_difference": normalized_l1,
        "normalized_l2_difference": normalized_l2,
        "maximum_location": list(index),
        "reference_at_maximum": float(reference[index]),
        "actual_at_maximum": float(actual[index]),
        "steps": steps,
        "epsilon": epsilon,
        "reference_scale": scale,
        "absolute_bound": absolute_bound,
        "relative_bound": relative_bound,
        "passed": passed,
    }


def admission() -> dict[str, Any]:
    phase6d = subprocess.run(
        (sys.executable, str(PHASE6D_VERIFY)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    fresh = phase6c_admission()
    production_hashes = {
        str(path.relative_to(ROOT)): {
            "current": sha256(path),
            "phase6d_timing_source": git_blob_sha256(PHASE6D_TIMING_COMMIT, path),
        }
        for path in PRODUCTION_SOURCES
    }
    production_unchanged = all(
        item["current"] == item["phase6d_timing_source"]
        for item in production_hashes.values()
    )
    passed = (
        phase6d.returncode == 0
        and fresh["passed"]
        and fresh["cuda_status"] == "admitted"
        and production_unchanged
    )
    return {
        "phase6d_verification_stdout": phase6d.stdout.strip(),
        "phase6d_verification_passed": phase6d.returncode == 0,
        "fresh_cuda": fresh,
        "production_source_hashes": production_hashes,
        "production_sources_match_phase6d_timing_source": production_unchanged,
        "passed": passed,
    }


def compare_records(
    arrays: Path,
    reference: dict[str, Any],
    actual: dict[str, Any],
    *,
    comparison_name: str,
) -> dict[str, Any]:
    identity = {
        "comparison": comparison_name,
        "problem": reference["problem"],
        "method": reference["method"],
        "reference": reference.get("array_file"),
        "actual": actual.get("array_file"),
    }
    if not reference.get("eligible") or not actual.get("eligible"):
        return {
            **identity,
            "comparison_available": False,
            "reason": "ineligible_worker",
            "passed": False,
        }
    reference_array = np.load(
        arrays / reference["array_file"], allow_pickle=False
    )
    actual_array = np.load(arrays / actual["array_file"], allow_pickle=False)
    check = comparison(
        reference_array,
        actual_array,
        steps=max(
            reference["diagnostics"]["steps"], actual["diagnostics"]["steps"]
        ),
        reference_name=reference["array_file"],
        actual_name=actual["array_file"],
    )
    check.update(identity)
    check["comparison_available"] = True
    check["step_count_match"] = (
        reference["diagnostics"]["steps"] == actual["diagnostics"]["steps"]
    )
    check["passed"] = bool(check["passed"] and check["step_count_match"])
    return check


def worker_environment(cache: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    environment["TORCHINDUCTOR_CACHE_DIR"] = cache
    return environment


def run_worker(
    *,
    problem: str,
    method: str,
    device: str,
    mode: str,
    replicate: int,
    record_path: Path,
    array_path: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gradflow-phase6e-repro-") as cache:
        started = time.perf_counter_ns()
        completed = subprocess.run(
            (
                sys.executable,
                str(WORKER),
                "--problem",
                problem,
                "--method",
                method,
                "--device",
                device,
                "--mode",
                mode,
                "--replicate",
                str(replicate),
                "--output",
                str(record_path),
                "--array-output",
                str(array_path),
            ),
            cwd=ROOT,
            env=worker_environment(cache),
            capture_output=True,
            text=True,
            check=False,
        )
        process_seconds = (time.perf_counter_ns() - started) * 1.0e-9
    record = json.loads(record_path.read_text()) if record_path.exists() else {
        "status": "failed",
        "eligible": False,
        "error_type": "WorkerProcessFailure",
        "error": "worker produced no record",
    }
    record.update(
        {
            "worker_returncode": completed.returncode,
            "worker_stdout": completed.stdout,
            "worker_stderr": completed.stderr,
            "process_launch_to_exit_seconds_diagnostic": process_seconds,
            "command": [str(WORKER), problem, method, device, mode, str(replicate)],
        }
    )
    record["eligible"] = bool(
        record.get("eligible")
        and completed.returncode == 0
        and array_path.is_file()
        and record.get("array_file_sha256") == sha256(array_path)
    )
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_model": "AMD Ryzen 5 7600X 6-Core Processor",
        "visible_logical_cpus": os.cpu_count(),
        "cuda_device": properties.name,
        "cuda_runtime": torch.version.cuda,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "fp64_context": "consumer_RTX_rate_restricted",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6E requires a clean committed source tree")
    admitted = admission()
    if not admitted["passed"]:
        raise RuntimeError("Phase 6E reproducibility admission failed")
    source_commit = git("rev-parse", "HEAD")
    raw = output / "raw"
    arrays = output / "arrays"
    raw.mkdir(parents=True)
    arrays.mkdir()
    records = []
    for problem in PROBLEMS:
        for method in METHODS:
            configurations = [("cpu", "eager", 0)] + [
                ("cuda", "compiled", replicate)
                for replicate in range(CUDA_REPLICATES)
            ]
            for device, mode, replicate in configurations:
                stem = f"{problem}_{method}_{device}_{mode}_r{replicate}"
                record = run_worker(
                    problem=problem,
                    method=method,
                    device=device,
                    mode=mode,
                    replicate=replicate,
                    record_path=raw / f"{stem}.json",
                    array_path=arrays / f"{stem}.npy",
                )
                records.append(record)
                print(
                    f"{stem}: {record.get('status')} eligible={record.get('eligible')}",
                    flush=True,
                )

    lookup = {
        (item["problem"], item["method"], item["device"], item["replicate"]): item
        for item in records
    }
    comparisons = []
    for problem in PROBLEMS:
        for method in METHODS:
            cpu_record = lookup[(problem, method, "cpu", 0)]
            cuda_items = [lookup[(problem, method, "cuda", replicate)] for replicate in range(CUDA_REPLICATES)]
            for item in cuda_items:
                comparisons.append(
                    compare_records(
                        arrays,
                        cpu_record,
                        item,
                        comparison_name="cpu_cuda",
                    )
                )
            for left_index in range(CUDA_REPLICATES):
                for right_index in range(left_index + 1, CUDA_REPLICATES):
                    left_record = cuda_items[left_index]
                    right_record = cuda_items[right_index]
                    comparisons.append(
                        compare_records(
                            arrays,
                            left_record,
                            right_record,
                            comparison_name="cuda_cuda",
                        )
                    )

    lane_passed = all(item["eligible"] for item in records) and all(
        item["passed"] for item in comparisons
    )
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_euler_phase_6e_lane_a",
        "measurement_date": "2026-08-29",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": source_commit,
        "source_dirty": False,
        "admission": admitted,
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PROTOCOL, WORKER, Path(__file__), *PRODUCTION_SOURCES)
        },
        "environment": environment(),
        "matrix": {
            "problems": list(PROBLEMS),
            "methods": list(METHODS),
            "cells": 800,
            "cpu_authorities_per_case": 1,
            "cuda_replicates_per_case": CUDA_REPLICATES,
            "rounding_factor": ROUNDING_FACTOR,
        },
        "records": records,
        "comparisons": comparisons,
        "all_workers_eligible": all(item["eligible"] for item in records),
        "all_comparisons_passed": all(item["passed"] for item in comparisons),
        "lane_a_passed": lane_passed,
        "performance_measurements_collected": False,
        "phase_6e_lanes_b_c_d_begun": False,
        "production_sources_modified": False,
        "dveb_modified": False,
        "publication_claim": False,
    }
    aggregate = output / "qualification.json"
    aggregate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    files = [aggregate, *sorted(raw.glob("*.json")), *sorted(arrays.glob("*.npy"))]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )
    print(f"Phase 6E Lane A passed={lane_passed}; wrote {aggregate}", flush=True)


if __name__ == "__main__":
    main()
