from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/academic_a4/evidence/unity_20260831"


def test_unity_result_retains_negative_portability_boundary() -> None:
    record = json.loads((EVIDENCE / "second_machine.json").read_text())

    assert record["status"] == "fail_needs_investigation"
    assert record["source_commit"] == "c5e8ab81ef5b33a2138b2db33afc538398b6f57f"
    environment = record["environment"]
    assert environment["git_status"] == ""
    assert environment["hostname"] == "gypsum-gpu005"
    assert environment["gpu"] == "Tesla M40 24GB"
    assert environment["gpu_capability"] == [5, 2]
    assert environment["torch"] == "2.13.0+cu126"

    assert record["a1"]["returncode"] == 0
    a3 = record["a3"]["record"]
    assert a3["derivative_gate"]["registered_window_passed"]
    assert a3["inverse_gate"]["passed"]
    assert a3["benchmarks"]["cuda"]["record"]["eager"]["admitted"]
    compiled_cuda = a3["benchmarks"]["cuda"]["record"]["compiled"]
    assert not compiled_cuda["admitted"]
    assert "GPUTooOldForTriton" in compiled_cuda["error"]

    assert len(record["a2_workers"]) == 36
    failures = record["qualification"]["admission_failures"]
    assert len(failures) == 18
    assert {failure["device"] for failure in failures} == {"cuda"}
    assert {failure["lane"] for failure in failures} == {"compiled"}
    assert all(
        1.0 < cell["fastest_cuda_over_fastest_cpu"] < 2.0
        for cell in record["a2_analysis"]["cells"]
    )


def test_unity_controller_checksum_manifest_verifies() -> None:
    expected = {}
    for line in (EVIDENCE / "SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    assert expected

    for relative, digest in expected.items():
        path = EVIDENCE / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    imported = {}
    for line in (EVIDENCE / "IMPORT_SHA256SUMS").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        imported[relative.removeprefix("./")] = digest
    assert len(imported) == 109
    for relative, digest in imported.items():
        path = EVIDENCE / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_unity_result_documents_packet_infrastructure_failures() -> None:
    pytest_output = (EVIDENCE / "raw/pytest.stdout").read_text()
    assert "344 passed, 12 skipped" in pytest_output
    assert "academic-v0.1.0-rc1:.gitignore" in pytest_output
    assert "assert package.is_file()" in pytest_output
    results = (ROOT / "docs/ACADEMIC_A4_UNITY_RESULTS.md").read_text()
    assert "backend-support boundary" in results
    assert "does not close A4" in results
