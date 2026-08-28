#!/usr/bin/env python3
"""Verify the immutable nonlinear FD/FV Phase-5B qualification record."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_nonlinear/results/phase_5b_20260828"
RECORD = RESULTS / "qualification.json"
SOURCE_COMMIT = "7eb2ba2f8d8a181557bffcda3e49214d3bb6e0b5"
PROTOCOL_COMMIT = "0d7b427"
SIZES = (24, 36, 54, 81)
METHODS = {"fd", "fv"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rates(errors: list[float]) -> list[float]:
    return [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors, errors[1:], SIZES, SIZES[1:]
        )
    ]


def assert_rates(actual: list[float], errors: list[float]) -> None:
    expected = rates(errors)
    assert len(actual) == len(expected)
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-14)
        for left, right in zip(actual, expected)
    )


def verify_spatial(payload: dict) -> None:
    spatial = payload["spatial_convergence"]
    assert set(spatial["methods"]) == METHODS
    assert spatial["noncritical_exclusion_radius"] == 0.1
    assert spatial["passed"] is True
    for method in spatial["methods"].values():
        records = method["records"]
        assert [record["cells"] for record in records] == list(SIZES)
        l1 = [record["l1_error"] for record in records]
        l2 = [record["l2_error"] for record in records]
        noncritical = [record["noncritical_l1_error"] for record in records]
        assert all(
            fine < coarse
            for errors in (l1, l2, noncritical)
            for coarse, fine in zip(errors, errors[1:])
        )
        assert all(record["finite"] for record in records)
        assert_rates(method["l1_rates"], l1)
        assert_rates(method["l2_rates"], l2)
        assert_rates(method["noncritical_l1_rates"], noncritical)
        assert method["l1_rates"][-1] >= 3.0
        assert method["noncritical_l1_rates"][-1] >= 4.3
        assert method["passed"] is True


def verify_solves(payload: dict) -> None:
    solves = payload["complete_solve_convergence"]
    assert set(solves["methods"]) == METHODS
    assert solves["passed"] is True
    for method in solves["methods"].values():
        records = method["records"]
        assert [record["cells"] for record in records] == list(SIZES)
        l1 = [record["l1_error"] for record in records]
        l2 = [record["l2_error"] for record in records]
        assert all(
            fine < coarse
            for errors in (l1, l2)
            for coarse, fine in zip(errors, errors[1:])
        )
        assert_rates(method["l1_rates"], l1)
        assert_rates(method["l2_rates"], l2)
        assert method["l1_rates"][-1] >= 3.0
        assert l1[-1] <= 2.0e-5
        assert l2[-1] <= 2.0e-5
        assert all(record["finite"] for record in records)
        assert all(record["conservation_passed"] for record in records)
        assert all(
            record["mass_change"] <= record["conservation_bound"]
            for record in records
        )
        assert method["passed"] is True


def verify_compiler_and_device(payload: dict) -> None:
    compiler = payload["compiler_and_device"]
    assert compiler["passed"] is True
    assert set(compiler["cpu"]["cases"]) == {
        "fd_rhs",
        "fd_step",
        "fv_rhs",
        "fv_step",
    }
    for case in compiler["cpu"]["cases"].values():
        assert case["graph_count"] == 1
        assert case["graph_break_count"] == 0
        assert case["break_reasons"] == []
        assert case["compiled_eager_maximum_absolute_difference"] <= 2.0e-11
        assert case["finite"] and case["resident"] and case["passed"]
        assert case["compilation_duration_measured"] is False

    cuda = compiler["cuda"]
    assert cuda["host_inventory"] == "present"
    assert cuda["process_visible"] is True
    assert cuda["status"] == "admitted"
    assert cuda["passed"] is True
    assert set(cuda["agreement"]) == set(compiler["cpu"]["cases"])
    assert set(cuda["compiler_cases"]) == set(compiler["cpu"]["cases"])
    for case in cuda["agreement"].values():
        assert case["maximum_absolute_difference"] <= 2.0e-11
        assert case["finite"] and case["resident"] and case["passed"]
    for case in cuda["compiler_cases"].values():
        assert case["graph_count"] == 1
        assert case["graph_break_count"] == 0
        assert case["break_reasons"] == []
        assert case["compiled_eager_maximum_absolute_difference"] <= 2.0e-11
        assert case["finite"] and case["resident"] and case["passed"]
        assert case["compilation_duration_measured"] is False


def verify_transfers(payload: dict) -> None:
    transfers = payload["transfer_evidence"]
    assert transfers["static_forbidden_hits"] == []
    assert set(transfers["profiles"]) == {
        "cpu_fd",
        "cpu_fv",
        "cuda_fd",
        "cuda_fv",
    }
    for profile in transfers["profiles"].values():
        assert profile["movement_events"] == []
        assert profile["aten_to_zero_memory"] is True
        assert profile["resident"] is True
        assert profile["passed"] is True
        for event in profile["aten_to_events"]:
            for field in (
                "cpu_memory_usage",
                "self_cpu_memory_usage",
                "device_memory_usage",
                "self_device_memory_usage",
            ):
                assert event[field] in (0, None)
    assert transfers["passed"] is True


def main() -> None:
    payload = json.loads(RECORD.read_text())
    assert payload["schema_version"] == 1
    assert payload["phase"] == "fd_fv_nonlinear_phase_5b"
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["source_dirty"] is False
    assert payload["protocol_commit"] == PROTOCOL_COMMIT
    assert payload["protocol"] == "docs/FD_FV_PHASE_5B_PROTOCOL.md"
    assert payload["performance_measurements_collected"] is False
    assert payload["passed"] is True
    assert payload["failed_gates"] == []
    assert all(payload["gate_decisions"].values())
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected

    predecessor = payload["predecessor"]
    assert predecessor["passed"] is True
    assert predecessor["returncode"] == 0
    assert predecessor["stderr"] == ""
    assert predecessor["contract_sha256"] == sha256(
        ROOT
        / "experiments/fd_fv_nonlinear/results/phase_5a_20260828/contract.json"
    )
    assert predecessor["oracle_cases_sha256"] == sha256(
        ROOT
        / "experiments/fd_fv_nonlinear/results/phase_5a_20260828/oracle_cases.json"
    )

    projections = payload["projection_oracle"]
    assert len(projections["cases"]) == 8
    assert projections["oracle_forbidden_imports"] == []
    assert projections["oracle_independent"] is True
    assert all(case["hex_values_equal"] for case in projections["cases"].values())
    assert projections["passed"] is True

    constants = payload["constant_and_conservation"]
    assert set(constants["cases"]) == METHODS
    for case in constants["cases"].values():
        assert case["constant_rhs_maximum_absolute"] <= 5.0e-13
        assert case["constant_step_maximum_absolute"] <= 5.0e-13
        assert case["nonconstant_rhs_mass_residual"] <= case[
            "conservation_bound"
        ]
        assert case["passed"] is True
    assert constants["passed"] is True

    verify_spatial(payload)
    verify_solves(payload)

    gradients = payload["differentiation"]
    assert set(gradients["cases"]) == METHODS
    for case in gradients["cases"].values():
        assert case["finite"] is True
        assert case["maximum_absolute_difference"] <= 3.0e-6
        assert case["relative_l2_difference"] <= 3.0e-5
        assert case["passed"] is True
    assert gradients["passed"] is True

    verify_compiler_and_device(payload)
    verify_transfers(payload)

    environment = payload["environment"]
    assert environment["cuda_process_visible"] is True
    assert environment["cuda_runtime"] == "13.0"
    assert environment["cuda"]["device"] == "NVIDIA GeForce RTX 5070 Ti"
    assert environment["cuda"]["capability"] == [12, 0]
    assert environment["cuda"]["total_memory_bytes"] == 16609247232
    assert environment["cuda"]["multiprocessor_count"] == 70
    assert environment["mps_status"] == "host_confirmed_absent"

    expected, name = (RESULTS / "SHA256SUMS").read_text().strip().split("  ", 1)
    assert name == "qualification.json"
    assert sha256(RECORD) == expected
    print(
        "FD/FV nonlinear Phase 5B verified: both Burgers JS5 formulations "
        "passed oracle, convergence, conservation, differentiation, CPU/CUDA, "
        "compiler, and residency gates; no performance timing was collected."
    )


if __name__ == "__main__":
    main()
