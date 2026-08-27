#!/usr/bin/env python3
"""Verify the immutable FD/FV Phase-3R record and prospective decision."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = (
    ROOT / "experiments/fd_fv_qualification/results/phase_3r_20260827"
)
RECORD_PATH = RESULT_DIR / "resolution.json"
SOURCE_COMMIT = "bd07370b9f36954567687e8c0cbc5f1a27ae24d3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    record = json.loads(RECORD_PATH.read_text())
    assert record["schema_version"] == 1
    assert record["phase"] == "fd_fv_phase_3r"
    assert record["formulation_id"] == (
        "fv_dimensional_js5_global_lf_periodic_v1"
    )
    assert record["protocol_commit"] == "93ad80b"
    assert record["source_commit"] == SOURCE_COMMIT
    assert record["source_dirty"] is False
    assert record["performance_measurements_collected"] is False

    for relative, expected in record["source_hashes"].items():
        assert sha256(ROOT / relative) == expected
    for relative, expected in record["original_record_hashes"].items():
        assert sha256(ROOT / relative) == expected

    immutable = record["immutable_record_verification"]
    assert immutable["passed"] is True
    assert immutable["phase_3_manifest"]["passed"] is True
    assert all(
        verification["returncode"] == 0 and verification["passed"]
        for verification in immutable["verifiers"].values()
    )

    identity = record["canonical_source_identity"]
    assert identity["canonical_source_commit"] == (
        "1d920ea97ed7abec9e4e451b377343cf72316f4c"
    )
    assert identity["actual_sha256"] == identity["expected_sha256"]
    assert identity["passed"] is True

    noncritical = record["noncritical_design_order"]
    assert noncritical["field"] == "exp(x)"
    assert set(noncritical["sequences"]) == {
        "left_face",
        "right_face",
        "positive_rhs",
        "negative_rhs",
    }
    for sequence in noncritical["sequences"].values():
        assert sequence["monotone"] is True
        assert all(
            fine < coarse
            for coarse, fine in zip(
                sequence["l2_errors"], sequence["l2_errors"][1:]
            )
        )
        assert sequence["last_two_rates"] == sequence["rates"][-2:]
        assert all(rate >= 4.7 for rate in sequence["last_two_rates"])
        assert sequence["passed"] is True
    assert noncritical["passed"] is True

    critical = record["critical_point_characterization"]
    mixed = critical["mixed_fourier_reproduction"]
    assert mixed["maximum_absolute_reproduction_difference"] <= 1.0e-15
    assert mixed["passed"] is True
    aligned = critical["aligned_simple_critical_point"]
    assert aligned["critical_face"] == 0.25
    for bias in ("left", "right"):
        result = aligned[bias]
        assert result["finite"] is True
        assert all(math.isfinite(error) for error in result["absolute_errors"])
        assert all(
            fine < coarse
            for coarse, fine in zip(
                result["absolute_errors"], result["absolute_errors"][1:]
            )
        )
        assert result["monotone"] is True
        assert result["passed"] is True
    assert aligned["passed"] is True
    assert critical["passed"] is True

    movement = record["movement_evidence"]
    static = movement["static"]
    assert static["forbidden_calls"] == []
    assert static["forbidden_to_calls"] == []
    assert static["dtype_only_to_calls"] == [
        {"line": 78, "path": "src/gradflow/weno_js.py"}
    ]
    assert static["passed"] is True
    cpu = movement["cpu"]
    assert cpu["movement_events"] == []
    assert cpu["resident"] is True
    assert cpu["input_device"] == cpu["output_device"] == "cpu"
    assert cpu["input_dtype"] == cpu["output_dtype"] == "float64"
    assert cpu["aten_to_events"]
    for event in cpu["aten_to_events"]:
        assert event["cpu_memory_usage"] == 0
        assert event["self_cpu_memory_usage"] == 0
        assert event["device_memory_usage"] == 0
        assert event["self_device_memory_usage"] == 0
    assert cpu["passed"] is True
    cuda = movement["cuda"]
    assert cuda.get("passed", cuda.get("status") == "untested_unavailable")
    assert movement["passed"] is True

    assert record["failed_gates"] == []
    assert all(record["gate_decisions"].values())
    assert record["passed"] is True

    expected_hash, expected_name = (
        RESULT_DIR / "SHA256SUMS"
    ).read_text().strip().split("  ", 1)
    assert expected_name == "resolution.json"
    assert sha256(RECORD_PATH) == expected_hash

    print(
        "FD/FV Phase 3R verified: noncritical fifth-order behavior, preserved "
        "critical-point evidence, and zero observed CPU data movement; CUDA "
        "remains unavailable."
    )


if __name__ == "__main__":
    main()
