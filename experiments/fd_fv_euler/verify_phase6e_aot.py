#!/usr/bin/env python3
"""Verify the preserved initial Phase-6E AOT qualification attempt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6e_aot_20260829"
RECORD = RESULTS / "qualification.json"
EXPECTED_SOURCE_COMMIT = "000e5217d6d281a6ebb0bb9909adbbaa59228809"
EXPECTED_PROTOCOL_COMMIT = "af90466"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha256(commit: str, relative: str) -> str:
    blob = subprocess.check_output(
        ("git", "show", f"{commit}:{relative}"), cwd=ROOT
    )
    return hashlib.sha256(blob).hexdigest()


def tensor_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_files(payload: dict) -> None:
    listed = {}
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    files = [
        RECORD,
        *sorted((RESULTS / "build_records").glob("*.json")),
        *sorted((RESULTS / "qualification_records").glob("*.json")),
        *sorted((RESULTS / "arrays").glob("*.npy")),
    ]
    assert listed == {
        str(path.relative_to(RESULTS)): sha256(path) for path in files
    }
    build_raw = {
        path.stem: json.loads(path.read_text())
        for path in sorted((RESULTS / "build_records").glob("*.json"))
    }
    assert len(build_raw) == len(payload["build_records"]) == 8
    for record in payload["build_records"]:
        stem = f"{record['lane']}_{record['problem']}_{record['method']}"
        assert build_raw[stem] == record
    qualification_raw = {
        path.stem: json.loads(path.read_text())
        for path in sorted((RESULTS / "qualification_records").glob("*.json"))
    }
    assert len(qualification_raw) == 4
    for record in payload["qualification_records"]:
        if record["status"] != "completed":
            assert record["status"] == "not_run_build_failed"
            continue
        stem = f"{record['lane']}_{record['problem']}_{record['method']}"
        assert qualification_raw[stem] == record
        array_path = RESULTS / "arrays" / record["array_file"]
        array = np.load(array_path, allow_pickle=False)
        assert array.shape == (3, 800) and array.dtype == np.float64
        assert sha256(array_path) == record["array_file_sha256"]
        assert tensor_sha256(array) == record["terminal_sha256"]


def main() -> None:
    payload = json.loads(RECORD.read_text())
    assert payload["phase"] == "fd_fv_euler_phase_6e_aot_qualification"
    assert payload["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert payload["protocol_commit"] == EXPECTED_PROTOCOL_COMMIT
    assert payload["source_dirty"] is False
    for name, digest in payload["source_hashes"].items():
        assert git_blob_sha256(EXPECTED_SOURCE_COMMIT, name) == digest
    lane_a = subprocess.run(
        (sys.executable, str(ROOT / "experiments/fd_fv_euler/verify_phase6e_repro.py")),
        cwd=ROOT,
        check=False,
    )
    assert lane_a.returncode == 0
    assert payload["lane_a_verification_passed"] is True
    verify_files(payload)

    host_builds = [item for item in payload["build_records"] if item["lane"] == "host"]
    device_builds = [item for item in payload["build_records"] if item["lane"] == "device"]
    assert len(host_builds) == len(device_builds) == 4
    for record in host_builds:
        assert record["status"] == "completed" and record["worker_returncode"] == 0
        package = Path(record["package_path"])
        assert package.is_file()
        assert sha256(package) == record["package_sha256"]
        assert package.stat().st_size == record["package_bytes"]
        assert record["custom_operator_used"] is False
    for record in device_builds:
        assert record["status"] == "failed" and record["worker_returncode"] != 0
        assert record["error_type"] == "RuntimeError"
        assert "aliasing the input or the output" in record["error"]
        assert record["custom_operator_used"] is False

    completed = [
        item for item in payload["qualification_records"] if item["status"] == "completed"
    ]
    skipped = [
        item for item in payload["qualification_records"]
        if item["status"] == "not_run_build_failed"
    ]
    assert len(completed) == len(skipped) == 4
    for record in completed:
        assert record["lane"] == "host" and record["worker_returncode"] == 0
        assert record["one_step_passed"]
        assert max(record["one_step_maximum_absolute_differences"]) <= 5.0e-11
        assert record["authority_parity"]["passed"]
        assert record["oracle"]["passed"]
        assert record["diagnostics"]["completed"]
        assert not record["movement_probe"]["forbidden_movement_events"]
        assert record["package_inventory"]["shared_objects"]
        sources = record["runtime_compiler_sources_created"]
        assert len(sources) == 6 and all(name.endswith(".cpp") for name in sources)
        expected_eligible = (
            record["one_step_passed"]
            and record["authority_parity"]["passed"]
            and record["oracle"]["passed"]
            and record["diagnostics"]["completed"]
            and not record["movement_probe"]["forbidden_movement_events"]
            and not sources
        )
        assert expected_eligible is False
        assert record["eligible"] is False
    assert payload["lane_status"] == {
        "host": {
            "builds_completed": 4,
            "qualifications_eligible": 0,
            "passed": False,
            "performance_admitted": False,
        },
        "device": {
            "builds_completed": 0,
            "qualifications_eligible": 0,
            "passed": False,
            "performance_admitted": False,
        },
    }
    assert payload["performance_measurements_collected"] is False
    assert payload["production_sources_modified"] is False
    assert payload["dveb_modified"] is False
    assert payload["publication_claim"] is False
    print("FD/FV Euler Phase 6E initial AOT qualification verification passed")


if __name__ == "__main__":
    main()
