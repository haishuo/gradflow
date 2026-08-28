#!/usr/bin/env python3
"""Verify the deferred Forge CUDA correctness-gates supplement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "experiments/deferred_cuda_gates/results/qualification_20260828"
RECORD_PATH = RESULT_DIR / "qualification.json"
SOURCE_COMMIT = "a73c777e0cd956d6bac6a9f1e6a307b7cf0e51bf"
ORDERS = (5, 7, 9, 11, 13, 15)
REPRESENTATIVE_ORDERS = (5, 11, 15)
BOUNDARIES = ("periodic", "transmissive")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    payload = json.loads(RECORD_PATH.read_text())
    assert payload["schema_version"] == 1
    assert payload["phase"] == "deferred_cuda_correctness_gates"
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["source_dirty"] is False
    assert payload["protocol"] == "docs/DEFERRED_CUDA_GATES_PROTOCOL.md"
    assert payload["passed"] is True
    assert payload["failed_gates"] == []
    assert all(payload["gate_decisions"].values())
    assert payload["performance_measurements_collected"] is False
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected

    environment = payload["environment"]
    assert environment["device"] == "NVIDIA GeForce RTX 5070 Ti"
    assert environment["device_capability"] == [12, 0]
    assert environment["device_total_memory_bytes"] == 16609247232
    assert environment["multiprocessor_count"] == 70
    assert environment["cuda_runtime"] == "13.0"
    assert environment["cuda_driver"] == "580.173.02"
    assert environment["mps_available"] is False
    assert payload["mps"] == {
        "status": "untested_unavailable",
        "available": False,
    }

    predecessors = payload["predecessors"]
    assert predecessors["passed"] is True
    assert set(predecessors["records"]) == {
        "fd_fv_phase_3",
        "fd_fv_phase_3r",
        "euler_boundary_shock_phase_b",
    }
    for record in predecessors["records"].values():
        assert record["passed"] is True
        assert record["returncode"] == 0
        assert record["stderr"] == ""
        assert sha256(ROOT / record["record"]) == record["record_sha256"]

    scalar = payload["scalar_fv"]
    agreement = scalar["cpu_cuda_agreement"]
    assert agreement["passed"] is True
    assert set(agreement["cases"]) == {"float32", "float64"}
    for dtype, tolerance in (("float32", 2.0e-4), ("float64", 2.0e-11)):
        case = agreement["cases"][dtype]
        assert case["tolerance"] == tolerance
        assert case["passed"] is True
        assert case["finite"] is True
        assert case["resident"] is True
        for name, value in case.items():
            if name.endswith("maximum_absolute_difference"):
                assert value <= tolerance

    compilation = scalar["cuda_compilation"]
    assert compilation["passed"] is True
    assert set(compilation["cases"]) == {
        f"{dtype}_{call}"
        for dtype in ("float32", "float64")
        for call in ("rhs", "ssp_rk3_step")
    }
    for case in compilation["cases"].values():
        assert case["passed"] is True
        assert case["graph_count"] == 1
        assert case["graph_break_count"] == 0
        assert case["break_reasons"] == []
        assert case["finite"] is True
        assert case["resident"] is True
        assert case["compiled_eager_maximum_absolute_difference"] <= case[
            "tolerance"
        ]

    movement = scalar["cuda_movement"]
    assert movement["passed"] is True
    assert movement["resident"] is True
    assert movement["input_device"] == movement["output_device"] == "cuda:0"
    assert movement["dtype"] == "float64"
    assert movement["movement_events"] == []
    assert len(movement["aten_to_events"]) == 1
    to_event = movement["aten_to_events"][0]
    assert to_event["count"] == 18
    for field in (
        "cpu_memory_usage",
        "self_cpu_memory_usage",
        "device_memory_usage",
        "self_device_memory_usage",
    ):
        assert to_event[field] == 0

    euler = payload["euler1d"]
    euler_agreement = euler["cpu_cuda_agreement"]
    assert euler_agreement["passed"] is True
    assert len(euler_agreement["cases"]) == 24
    assert {
        (case["order"], case["dtype"], case["boundary"])
        for case in euler_agreement["cases"].values()
    } == {
        (order, dtype, boundary)
        for order in ORDERS
        for dtype in ("float32", "float64")
        for boundary in BOUNDARIES
    }
    for case in euler_agreement["cases"].values():
        assert case["passed"] is True
        assert case["finite"] is True
        assert case["resident"] is True
        assert case["maximum_absolute_difference"] <= case["tolerance"]

    euler_compilation = euler["cuda_compilation"]
    assert euler_compilation["passed"] is True
    assert len(euler_compilation["cases"]) == 6
    assert set(euler_compilation["cases"]) == {
        f"order{order}_float64_{boundary}"
        for order in REPRESENTATIVE_ORDERS
        for boundary in BOUNDARIES
    }
    for case in euler_compilation["cases"].values():
        assert case["passed"] is True
        assert case["graph_count"] == 1
        assert case["graph_break_count"] == 0
        assert case["break_reasons"] == []
        assert case["finite"] is True
        assert case["resident"] is True
        assert case["compiled_eager_maximum_absolute_difference"] <= 5.0e-11

    cfl = euler["cuda_cfl"]
    assert cfl["passed"] is True
    assert cfl["shape"] == []
    assert cfl["device"] == "cuda:0"
    assert cfl["dtype"] == "float64"
    assert cfl["finite"] is True
    assert cfl["positive"] is True
    assert cfl["resident"] is True

    expected, relative = (RESULT_DIR / "SHA256SUMS").read_text().strip().split(
        "  ", 1
    )
    assert relative == "qualification.json"
    assert sha256(RECORD_PATH) == expected
    print(
        "Deferred CUDA gates verified: scalar FV agreement/compilation/"
        "movement and 24 Euler agreement plus six compiler cases passed on "
        "Forge; no performance timing was collected."
    )


if __name__ == "__main__":
    main()
