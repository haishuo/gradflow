#!/usr/bin/env python3
"""Independently verify the timing-free nonlinear Phase-5CR resolution."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.fd_fv_nonlinear.performance_problem import METHOD_IDS
from experiments.fd_fv_nonlinear.run_phase5c import (
    ERROR_TARGETS,
    METHODS,
    MODES,
    aggregate_complete,
    classification,
    target_selections,
)


RESULTS = ROOT / "experiments/fd_fv_nonlinear/results/phase_5cr_20260828"
RECORD = RESULTS / "resolution.json"
INITIAL_RESULTS = ROOT / "experiments/fd_fv_nonlinear/results/phase_5c_20260828"
INITIAL_RECORD = INITIAL_RESULTS / "benchmark.json"
INITIAL_VERIFY = ROOT / "experiments/fd_fv_nonlinear/verify_phase5c_initial.py"
SOURCE_COMMIT = "aaa61040687b47b5b3d4bd690ba69d0b8f2f9220"
RECORD_SHA256 = "534ec63e6af0ec226ef9a5fb599a1d6000d1c05dff0aa9739ae5dc0f579f6ebb"
DIAGNOSTIC_KEYS = {
    (method, cells) for method in METHODS for cells in (81, 162)
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accumulated_bound(single: float, steps: int) -> float:
    return steps * (single - 2.0e-15) + 2.0e-15


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-27)


def complete_nonconservation_gates(record: dict) -> bool:
    eager = record["accuracy"]["eager"]
    compiled = record["accuracy"]["compiled"]
    return (
        record["status"] == "completed"
        and record["worker_returncode"] == 0
        and record["kind"] == "complete"
        and record["formulation_id"] == METHOD_IDS[record["method"]]
        and eager["finite"]
        and compiled["finite"]
        and all(
            math.isfinite(case[name])
            for case in (eager, compiled)
            for name in ("l1_error", "l2_error")
        )
        and record["accuracy"]["compiled_eager_maximum_absolute_difference"]
        <= 2.0e-11
        and record["accuracy"]["compiled_repeat_maximum_absolute_difference"]
        == 0.0
        and eager["dtype"] == compiled["dtype"] == "float64"
        and eager["shape"] == compiled["shape"] == [record["cells"]]
        and eager["device"].split(":")[0] == record["device"]
        and compiled["device"].split(":")[0] == record["device"]
    )


def cold_nonconservation_gates(record: dict) -> bool:
    return (
        record["status"] == "completed"
        and record["worker_returncode"] == 0
        and record["kind"] == "cold"
        and record["formulation_id"] == METHOD_IDS[record["method"]]
        and record["finite"]
        and math.isfinite(record["l1_error"])
        and math.isfinite(record["l2_error"])
        and record["host_visible_answer"]
    )


def verify_mass_measurement(record: dict) -> None:
    values = (
        record["device_reduction"],
        record["host_tensor_reduction"],
        record["host_fsum_difference"],
        record["host_separate_fsum_difference"],
    )
    disagreement = max(values) - min(values)
    close(record["maximum_reduction_disagreement"], disagreement)
    assert record["reduction_tolerance"] == 2.0e-16
    assert record["passed"] is (disagreement <= 2.0e-16)


def verify_diagnostics(records: list[dict]) -> dict[tuple[str, int], bool]:
    assert {(item["method"], item["cells"]) for item in records} == (
        DIAGNOSTIC_KEYS
    )
    status = {}
    for record in records:
        assert record["steps"] > 0
        assert float.fromhex(record["dt_hex"]) * record["steps"] == 0.1
        device_passes = []
        for device in ("cpu", "cuda"):
            case = record["devices"][device]
            rhs_passed = case["rhs_mass_residual"] <= case["rhs_mass_bound"]
            assert case["rhs_conservation_passed"] is rhs_passed
            mode_passes = []
            for mode in MODES:
                mode_case = case[mode]
                verify_mass_measurement(mode_case["one_step_mass"])
                verify_mass_measurement(mode_case["full_solve_mass"])
                expected = accumulated_bound(
                    mode_case["single_bound"], mode_case["steps"]
                )
                close(mode_case["accumulated_bound"], expected)
                mass = mode_case["full_solve_mass"]["host_fsum_difference"]
                close(mode_case["mass_per_step"], mass / mode_case["steps"])
                close(
                    mode_case["accumulated_bound_utilization"],
                    mass / expected,
                )
                close(
                    mode_case["per_step_bound_utilization"],
                    mass / mode_case["steps"] / mode_case["single_bound"],
                )
                passed = (
                    mode_case["one_step_mass"]["passed"]
                    and mode_case["full_solve_mass"]["passed"]
                    and mode_case["one_step_conservation_passed"]
                    and mass <= expected
                    and mass / mode_case["steps"]
                    <= mode_case["single_bound"]
                    and mode_case["finite"]
                    and math.isfinite(mode_case["l1_error"])
                    and math.isfinite(mode_case["l2_error"])
                )
                assert mode_case["passed"] is passed
                mode_passes.append(passed)
            device_passed = (
                rhs_passed
                and case["compiled_eager_maximum_absolute_difference"]
                <= 2.0e-11
                and all(mode_passes)
            )
            assert case["passed"] is device_passed
            device_passes.append(device_passed)
        parity_passes = []
        for mode in MODES:
            parity = record["cpu_cuda"][mode]
            assert parity["tolerance"] == 2.0e-11
            passed = parity["maximum_absolute_difference"] <= 2.0e-11
            assert parity["passed"] is passed
            parity_passes.append(passed)
        passed = all(device_passes) and all(parity_passes)
        assert record["passed"] is passed
        status[(record["method"], record["cells"])] = passed
    return status


def verify_complete(
    initial: list[dict], summaries: list[dict], diagnostics: dict[tuple, bool]
) -> list[dict]:
    assert len(summaries) == len(initial) == 60
    copies = deepcopy(initial)
    for source, summary, copy in zip(initial, summaries, copies):
        identity = (source["method"], source["device"], source["cells"])
        assert identity == (
            summary["method"],
            summary["device"],
            summary["cells"],
        )
        assert summary["replicate"] == source["replicate"]
        assert summary["original_eligible"] is source["eligible"]
        expected_modes = []
        for mode in MODES:
            item = summary["modes"][mode]
            mass = source["accuracy"][mode]["mass_change"]
            single = source["accuracy"]["eager"]["mass_bound"]
            expected = accumulated_bound(single, source["steps"])
            assert item["mass_change"] == mass
            assert item["single_bound"] == single
            close(item["accumulated_bound"], expected)
            close(item["mass_per_step"], mass / source["steps"])
            close(item["accumulated_bound_utilization"], mass / expected)
            close(
                item["per_step_bound_utilization"],
                mass / source["steps"] / single,
            )
            passed = mass <= expected and mass / source["steps"] <= single
            assert item["passed"] is passed
            expected_modes.append(passed)
        required = source["device"] == "cuda" and source["cells"] in (81, 162)
        assert summary["diagnostic_required"] is required
        diagnostic_passed = (
            diagnostics[(source["method"], source["cells"])]
            if required
            else True
        )
        assert summary["diagnostic_passed"] is (
            diagnostic_passed if required else True
        )
        eligible = (
            complete_nonconservation_gates(source)
            and all(expected_modes)
            and (diagnostic_passed if required else True)
        )
        assert summary["eligible_under_phase_5cr"] is eligible
        copy["eligible"] = eligible
        copy["eligible_under_phase_5cr"] = eligible
    return copies


def cold_targets(records: list[dict], summaries: list[dict]) -> dict:
    eligible = {
        (x["method"], x["device"], x["mode"], x["cells"]): x[
            "eligible_under_phase_5cr"
        ]
        for x in summaries
    }
    result = {}
    for device in ("cpu", "cuda"):
        targets = {}
        for target in ERROR_TARGETS:
            item = {}
            for method in METHODS:
                candidates = [
                    x
                    for x in records
                    if x["method"] == method
                    and x["device"] == device
                    and x["l2_error"] <= target
                    and eligible[(method, device, x["mode"], x["cells"])]
                ]
                if candidates:
                    chosen = min(
                        candidates,
                        key=lambda x: x["process_launch_to_exit_seconds"],
                    )
                    item[method] = {
                        "status": "reached",
                        "cells": chosen["cells"],
                        "mode": chosen["mode"],
                        "l2_error": chosen["l2_error"],
                        "seconds": chosen["process_launch_to_exit_seconds"],
                    }
                else:
                    item[method] = {"status": "not_reached"}
            if all(item[m]["status"] == "reached" for m in METHODS):
                ratio = item["fv"]["seconds"] / item["fd"]["seconds"]
                item["fv_over_fd_ratio"] = ratio
                item["classification"] = (
                    "unresolved_cold_pilot"
                    if 1.0 / 1.10 <= ratio <= 1.10
                    else classification(ratio)
                )
            targets[str(target)] = item
        result[f"cold_{device}"] = targets
    return result


def verify_cold(
    initial: list[dict], summaries: list[dict], diagnostics: dict[tuple, bool]
) -> None:
    assert len(summaries) == len(initial) == 24
    for source, summary in zip(initial, summaries):
        assert (
            summary["method"],
            summary["device"],
            summary["mode"],
            summary["cells"],
        ) == (
            source["method"],
            source["device"],
            source["mode"],
            source["cells"],
        )
        assert summary["original_eligible"] is source["eligible"]
        mass = source["mass_change"]
        single = source["mass_bound"]
        expected = accumulated_bound(single, source["steps"])
        assert summary["mass_change"] == mass
        assert summary["single_bound"] == single
        close(summary["accumulated_bound"], expected)
        close(summary["mass_per_step"], mass / source["steps"])
        required = source["device"] == "cuda" and source["cells"] in (81, 162)
        diagnostic_passed = (
            diagnostics[(source["method"], source["cells"])]
            if required
            else True
        )
        assert summary["diagnostic_required"] is required
        assert summary["diagnostic_passed"] is (
            diagnostic_passed if required else True
        )
        eligible = (
            cold_nonconservation_gates(source)
            and mass <= expected
            and mass / source["steps"] <= single
            and (diagnostic_passed if required else True)
        )
        assert summary["eligible_under_phase_5cr"] is eligible


def main() -> None:
    assert sha256(RECORD) == RECORD_SHA256
    assert (RESULTS / "SHA256SUMS").read_text() == (
        f"{RECORD_SHA256}  resolution.json\n"
    )
    payload = json.loads(RECORD.read_text())
    assert payload["schema_version"] == 1
    assert payload["phase"] == "fd_fv_nonlinear_phase_5cr"
    assert payload["protocol_commit"] == "5ab6950"
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["source_dirty"] is False
    assert payload["performance_measurements_collected"] is False
    assert payload["performance_samples_reused_unchanged"] is True
    assert payload["implementation_changed"] is False
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected

    predecessor = subprocess.run(
        (sys.executable, str(INITIAL_VERIFY)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert predecessor.returncode == 0, predecessor.stderr
    initial = json.loads(INITIAL_RECORD.read_text())
    assert payload["initial_phase_5c"]["record_sha256"] == sha256(INITIAL_RECORD)
    assert payload["initial_phase_5c"]["manifest_sha256"] == sha256(
        INITIAL_RESULTS / "SHA256SUMS"
    )
    assert payload["initial_phase_5c"]["original_complete_gate_passed"] is False
    assert payload["initial_phase_5c"]["original_cold_gate_passed"] is False

    diagnostics = verify_diagnostics(payload["fresh_diagnostics"])
    copies = verify_complete(
        initial["complete_records"],
        payload["complete_reclassification"],
        diagnostics,
    )
    aggregates = aggregate_complete(copies)
    assert aggregates == payload["resolved_complete_aggregates"]
    assert target_selections(aggregates) == payload["resolved_target_selections"]
    verify_cold(
        initial["cold_records"],
        payload["cold_reclassification"],
        diagnostics,
    )
    assert cold_targets(
        initial["cold_records"], payload["cold_reclassification"]
    ) == payload["resolved_cold_target_selections"]
    assert payload["preserved_step_device_crossovers"] == initial[
        "step_device_crossovers"
    ]
    assert all(payload["gate_decisions"].values())
    assert payload["failed_gates"] == []
    assert payload["passed"] is True
    print(
        "Phase 5CR verified: the immutable Phase 5C timings are unchanged, "
        "all mechanistic diagnostics pass, and all complete and cold cells "
        "are eligible under the prospectively accumulated roundoff bound."
    )


if __name__ == "__main__":
    main()
