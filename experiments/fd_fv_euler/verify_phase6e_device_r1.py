#!/usr/bin/env python3
"""Verify the corrected Phase-6E full-loop AOT requalification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6e_device_r1_20260829"
RECORD = RESULTS / "qualification.json"
EXPECTED_SOURCE_COMMIT = "f908ea14ffce72c9d5b33ea2e7d606f17f104a7f"
EXPECTED_PROTOCOL_COMMIT = "af90466"
EXPECTED_AMENDMENT_COMMIT = "94e0fe4"


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
    qualification_raw = {
        path.stem: json.loads(path.read_text())
        for path in sorted((RESULTS / "qualification_records").glob("*.json"))
    }
    assert len(build_raw) == len(payload["build_records"]) == 4
    assert len(qualification_raw) == len(payload["qualification_records"]) == 4
    for record in payload["build_records"]:
        stem = f"device_{record['problem']}_{record['method']}"
        assert build_raw[stem] == record
    for record in payload["qualification_records"]:
        stem = f"device_{record['problem']}_{record['method']}"
        assert qualification_raw[stem] == record
        array_path = RESULTS / "arrays" / record["array_file"]
        array = np.load(array_path, allow_pickle=False)
        assert array.shape == (3, 800) and array.dtype == np.float64
        assert sha256(array_path) == record["array_file_sha256"]
        assert tensor_sha256(array) == record["terminal_sha256"]


def main() -> None:
    payload = json.loads(RECORD.read_text())
    assert payload["phase"] == "fd_fv_euler_phase_6e_device_r1_qualification"
    assert payload["series"] == "phase_6e_device_r1_20260829"
    assert payload["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert payload["protocol_commit"] == EXPECTED_PROTOCOL_COMMIT
    assert payload["amendment_commit"] == EXPECTED_AMENDMENT_COMMIT
    assert payload["source_dirty"] is False
    for name, digest in payload["source_hashes"].items():
        assert git_blob_sha256(EXPECTED_SOURCE_COMMIT, name) == digest
    for prerequisite in payload["prerequisites"]:
        assert prerequisite["returncode"] == 0 and prerequisite["passed"]
    lane_a = subprocess.run(
        (sys.executable, str(ROOT / "experiments/fd_fv_euler/verify_phase6e_repro.py")),
        cwd=ROOT,
        check=False,
    )
    initial_aot = subprocess.run(
        (sys.executable, str(ROOT / "experiments/fd_fv_euler/verify_phase6e_aot.py")),
        cwd=ROOT,
        check=False,
    )
    assert lane_a.returncode == initial_aot.returncode == 0
    verify_files(payload)

    for record in payload["build_records"]:
        assert record["status"] == "completed" and record["worker_returncode"] == 0
        assert record["lane"] == "device"
        assert record["series"] == "phase_6e_device_r1_20260829"
        package = Path(record["package_path"])
        assert package.is_file()
        assert sha256(package) == record["package_sha256"]
        assert package.stat().st_size == record["package_bytes"]
        assert record["custom_operator_used"] is False

    for record in payload["qualification_records"]:
        assert record["status"] == "completed" and record["worker_returncode"] == 0
        assert record["lane"] == "device"
        assert record["series"] == "phase_6e_device_r1_20260829"
        assert record["authority_parity"]["passed"]
        assert record["oracle"]["passed"]
        assert record["diagnostics"]["completed"]
        assert record["diagnostics"]["cfl_scalar_host_controlled"] is False
        assert record["diagnostics"]["minimum_density"] > 0.0
        assert record["diagnostics"]["minimum_pressure"] > 0.0
        forbidden = record["movement_probe"]["forbidden_movement_events"]
        assert any("local_scalar" in name for name in forbidden)
        assert any("DtoH" in name for name in forbidden)
        sources = record["runtime_compiler_sources_created"]
        assert len(sources) == 6 and all(name.endswith(".cpp") for name in sources)
        assert record["package_inventory"]["shared_objects"]
        expected_eligible = (
            record["authority_parity"]["passed"]
            and record["oracle"]["passed"]
            and record["diagnostics"]["completed"]
            and not forbidden
            and not sources
        )
        assert expected_eligible is False
        assert record["eligible"] is False
    assert payload["lane_status"] == {
        "builds_completed": 4,
        "qualifications_eligible": 0,
        "passed": False,
        "performance_admitted": False,
    }
    assert payload["performance_measurements_collected"] is False
    assert payload["production_sources_modified"] is False
    assert payload["dveb_modified"] is False
    assert payload["publication_claim"] is False
    print("FD/FV Euler Phase 6E corrected device-loop verification passed")


if __name__ == "__main__":
    main()
