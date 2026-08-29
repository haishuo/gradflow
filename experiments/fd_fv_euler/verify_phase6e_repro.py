#!/usr/bin/env python3
"""Independently verify the preserved Phase-6E Lane-A qualification."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6e_20260829"
RECORD = RESULTS / "qualification.json"
EXPECTED_SOURCE_COMMIT = "96035dc34eeef286b3d7d21747a46ec74bb22aa5"
EXPECTED_PROTOCOL_COMMIT = "af90466"
PROBLEMS = ("sod", "shu_osher")
METHODS = ("fd", "fv")
CUDA_REPLICATES = 5
ROUNDING_FACTOR = 128.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)


def compare_nested(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            compare_nested(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare_nested(left, right)
    elif isinstance(expected, float):
        assert close(actual, expected), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def comparison(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    steps: int,
    reference_name: str,
    actual_name: str,
) -> dict[str, Any]:
    assert reference.shape == actual.shape == (3, 800)
    assert reference.dtype == actual.dtype == np.float64
    difference = np.abs(actual - reference)
    flat_index = int(np.argmax(difference))
    index = tuple(int(value) for value in np.unravel_index(flat_index, difference.shape))
    epsilon = float(np.finfo(reference.dtype).eps)
    steps = max(1, steps)
    scale = max(1.0, float(np.max(np.abs(reference))))
    relative_bound = ROUNDING_FACTOR * epsilon * steps
    absolute_bound = relative_bound * scale
    tiny = float(np.finfo(reference.dtype).tiny)
    normalized_l1 = float(np.sum(difference)) / max(
        float(np.sum(np.abs(reference))), tiny
    )
    normalized_l2 = float(np.linalg.norm(difference.ravel())) / max(
        float(np.linalg.norm(reference.ravel())), tiny
    )
    maximum = float(difference[index])
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
        "passed": (
            maximum <= absolute_bound
            and normalized_l1 <= relative_bound
            and normalized_l2 <= relative_bound
        ),
    }


def verify_files(payload: dict[str, Any]) -> None:
    listed = {}
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    files = [
        RECORD,
        *sorted((RESULTS / "raw").glob("*.json")),
        *sorted((RESULTS / "arrays").glob("*.npy")),
    ]
    assert listed == {
        str(path.relative_to(RESULTS)): sha256(path) for path in files
    }
    raw = {
        path.stem: json.loads(path.read_text())
        for path in sorted((RESULTS / "raw").glob("*.json"))
    }
    assert len(raw) == len(payload["records"]) == 24
    for record in payload["records"]:
        stem = (
            f"{record['problem']}_{record['method']}_{record['device']}_"
            f"{record['mode']}_r{record['replicate']}"
        )
        assert raw[stem] == record
        array_path = RESULTS / "arrays" / record["array_file"]
        array = np.load(array_path, allow_pickle=False)
        assert array.shape == (3, 800) and array.dtype == np.float64
        assert sha256(array_path) == record["array_file_sha256"]
        assert tensor_sha256(array) == record["terminal_sha256"]


def recompute_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (item["problem"], item["method"], item["device"], item["replicate"]): item
        for item in records
    }
    result = []
    for problem in PROBLEMS:
        for method in METHODS:
            cpu_record = lookup[(problem, method, "cpu", 0)]
            cuda_records = [
                lookup[(problem, method, "cuda", replicate)]
                for replicate in range(CUDA_REPLICATES)
            ]
            for actual_record in cuda_records:
                item = comparison_records(cpu_record, actual_record)
                item.update(
                    {"comparison": "cpu_cuda", "problem": problem, "method": method}
                )
                result.append(item)
            for left in range(CUDA_REPLICATES):
                for right in range(left + 1, CUDA_REPLICATES):
                    item = comparison_records(cuda_records[left], cuda_records[right])
                    item.update(
                        {"comparison": "cuda_cuda", "problem": problem, "method": method}
                    )
                    result.append(item)
    return result


def comparison_records(
    reference_record: dict[str, Any], actual_record: dict[str, Any]
) -> dict[str, Any]:
    reference = np.load(
        RESULTS / "arrays" / reference_record["array_file"], allow_pickle=False
    )
    actual = np.load(
        RESULTS / "arrays" / actual_record["array_file"], allow_pickle=False
    )
    item = comparison(
        reference,
        actual,
        steps=max(
            reference_record["diagnostics"]["steps"],
            actual_record["diagnostics"]["steps"],
        ),
        reference_name=reference_record["array_file"],
        actual_name=actual_record["array_file"],
    )
    item["comparison_available"] = True
    item["step_count_match"] = (
        reference_record["diagnostics"]["steps"]
        == actual_record["diagnostics"]["steps"]
    )
    item["passed"] = bool(item["passed"] and item["step_count_match"])
    return item


def main() -> None:
    payload = json.loads(RECORD.read_text())
    assert payload["phase"] == "fd_fv_euler_phase_6e_lane_a"
    assert payload["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert payload["protocol_commit"] == EXPECTED_PROTOCOL_COMMIT
    assert payload["source_dirty"] is False
    assert payload["matrix"] == {
        "problems": list(PROBLEMS),
        "methods": list(METHODS),
        "cells": 800,
        "cpu_authorities_per_case": 1,
        "cuda_replicates_per_case": CUDA_REPLICATES,
        "rounding_factor": ROUNDING_FACTOR,
    }
    for name, digest in payload["source_hashes"].items():
        assert sha256(ROOT / name) == digest
    phase6d = subprocess.run(
        (sys.executable, str(ROOT / "experiments/fd_fv_euler/verify_phase6d.py")),
        cwd=ROOT,
        check=False,
    )
    assert phase6d.returncode == 0
    assert payload["admission"]["passed"]
    assert payload["admission"][
        "production_sources_match_phase6d_timing_source"
    ]
    verify_files(payload)
    for record in payload["records"]:
        assert record["status"] == "completed" and record["worker_returncode"] == 0
        assert record["eligible"] and record["host_visible_answer"]
        assert record["dtype"] == "float64" and record["shape"] == [3, 800]
        assert record["diagnostics"]["completed"]
        assert all(record["gate_decisions"].values())
        assert record["diagnostics"]["minimum_density"] > 0.0
        assert record["diagnostics"]["minimum_pressure"] > 0.0
    recomputed = recompute_comparisons(payload["records"])
    assert len(recomputed) == len(payload["comparisons"]) == 60
    compare_nested(payload["comparisons"], recomputed)
    assert payload["all_workers_eligible"] is True
    assert payload["all_comparisons_passed"] is True
    assert payload["lane_a_passed"] is True
    assert payload["performance_measurements_collected"] is False
    assert payload["phase_6e_lanes_b_c_d_begun"] is False
    assert payload["production_sources_modified"] is False
    assert payload["dveb_modified"] is False
    assert payload["publication_claim"] is False
    print("FD/FV Euler Phase 6E Lane A verification passed")


if __name__ == "__main__":
    main()
