#!/usr/bin/env python3
"""Verify the immutable timing-free FD/FV Phase-4A admission record."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "experiments/fd_fv_bakeoff/results/phase_4a_20260827"
RECORD_PATH = RESULT_DIR / "qualification.json"
SOURCE_COMMIT = "7ff5708449d2b5e833a33cbf017a7ce98f5e272d"
EXPECTED_SIZES = {
    "1": [24, 36, 54, 81],
    "2": [12, 18, 27, 40],
    "3": [8, 12, 18, 27],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    record = json.loads(RECORD_PATH.read_text())
    assert record["schema_version"] == 1
    assert record["phase"] == "fd_fv_phase_4a"
    assert record["protocol_commit"] == "6dbd4d1"
    assert record["source_commit"] == SOURCE_COMMIT
    assert record["source_dirty"] is False
    assert record["performance_measurements_collected"] is False
    assert record["problem"]["sizes"] == EXPECTED_SIZES
    assert record["problem"]["dtype"] == "float64"

    for relative, expected in record["source_hashes"].items():
        assert sha256(ROOT / relative) == expected
    assert record["phase_3r"]["passed"] is True
    assert record["phase_3r"]["returncode"] == 0

    convergence = record["convergence"]
    assert set(convergence["methods"]) == {"fd", "fv"}
    for method in convergence["methods"].values():
        for dimension, result in method["dimensions"].items():
            assert list(result["sizes"]) == EXPECTED_SIZES[dimension]
            l1 = [run["l1_error"] for run in result["runs"]]
            l2 = [run["l2_error"] for run in result["runs"]]
            assert all(math.isfinite(value) for value in l1 + l2)
            assert all(fine < coarse for coarse, fine in zip(l1, l1[1:]))
            assert all(fine < coarse for coarse, fine in zip(l2, l2[1:]))
            assert max(result["l2_rates"]) >= 4.0
            assert all(run["finite"] for run in result["runs"])
            assert all(run["conservation_passed"] for run in result["runs"])
            assert result["monotone_l1"] is True
            assert result["monotone_l2"] is True
            assert result["passed"] is True
        assert method["passed"] is True
    assert convergence["passed"] is True

    compiler = record["compiler"]
    for method in compiler["methods"].values():
        for result in method["dimensions"].values():
            assert result["graph_count"] == 1
            assert result["graph_break_count"] == 0
            assert result["maximum_absolute_difference"] <= 2.0e-11
            assert result["shape_preserved"] is True
            assert result["dtype_preserved"] is True
            assert result["device_preserved"] is True
            assert result["passed"] is True
        assert method["passed"] is True
    assert compiler["passed"] is True

    cuda = record["cuda"]
    assert cuda.get("status") in {"passed", "untested_unavailable"}
    assert record["mps"]["status"] in {
        "not_executed",
        "untested_unavailable",
    }
    assert record["failed_gates"] == []
    assert all(record["gate_decisions"].values())
    assert record["passed"] is True

    expected_hash, expected_name = (
        RESULT_DIR / "SHA256SUMS"
    ).read_text().strip().split("  ", 1)
    assert expected_name == "qualification.json"
    assert sha256(RECORD_PATH) == expected_hash
    print(
        "FD/FV Phase 4A verified: both formulations passed 1-D/2-D/3-D "
        "convergence, conservation, and CPU compiler admission; CUDA remains "
        "unavailable and no timing was collected."
    )


if __name__ == "__main__":
    main()
