#!/usr/bin/env python3
"""Independently verify the preserved FD/FV Euler Phase-6C campaign."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6c_20260829"
RECORD = RESULTS / "benchmark.json"
EXPECTED_SOURCE_COMMIT = "2ad7867367098ca7048360e6cea949ceb067e944"
EXPECTED_PROTOCOL_COMMIT = "86a379f"
COMPLETE_SIZES = (24, 36, 54, 81, 162)
STEP_SIZES = (32, 128, 512, 2048, 8192, 32768, 131072, 524288)
COLD_SIZES = (24, 81, 162)
ERROR_TARGETS = (5.0e-6, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9)
SHOCK_SIZES = (200, 800)
METHODS = ("fd", "fv")
DEVICES = ("cpu", "cuda")
MODES = ("eager", "compiled")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)


def quantile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def verify_statistics(record: dict[str, Any]) -> None:
    samples = record["samples_seconds"]
    assert samples and all(math.isfinite(x) and x > 0.0 for x in samples)
    ordered = sorted(samples)
    expected = {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
        "q1_seconds": quantile(ordered, 0.25),
        "q3_seconds": quantile(ordered, 0.75),
    }
    for name, value in expected.items():
        assert close(record[name], value), (name, record[name], value)


def verify_files(payload: dict[str, Any]) -> None:
    expected_lines = {}
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        expected_lines[name] = digest
    actual_files = [RECORD, *sorted((RESULTS / "raw").glob("*.json"))]
    actual_lines = {
        str(path.relative_to(RESULTS)): sha256(path) for path in actual_files
    }
    assert expected_lines == actual_lines
    raw_lookup = {
        path.name: json.loads(path.read_text())
        for path in sorted((RESULTS / "raw").glob("*.json"))
    }
    embedded = (
        payload["complete_records"]
        + payload["step_records"]
        + payload["cold_records"]
        + payload["shock_records"]
    )
    assert len(raw_lookup) == len(embedded)
    for record in embedded:
        command = record["command"]
        output_name = None
        if record["kind"] == "complete":
            output_name = (
                f"complete_{record['device']}_{record['method']}_"
                f"n{record['cells']}_r{record['replicate']}.json"
            )
        elif record["kind"] == "step":
            output_name = (
                f"step_{record['device']}_{record['method']}_"
                f"n{record['cells']}_r{record['replicate']}.json"
            )
        elif record["kind"] == "cold":
            output_name = (
                f"cold_{record['device']}_{record['method']}_{record['mode']}_"
                f"n{record['cells']}.json"
            )
        elif record["kind"] == "shock":
            output_name = (
                f"shock_{record['problem']}_{record['device']}_"
                f"{record['method']}_{record['mode']}_n{record['cells']}.json"
            )
        assert command and raw_lookup[output_name] == record


def verify_worker_records(payload: dict[str, Any]) -> None:
    for record in payload["complete_records"]:
        assert record["status"] == "completed" and record["worker_returncode"] == 0
        for mode in MODES:
            verify_statistics(record[mode]["resident_complete_solve"])
            if record["device"] == "cuda":
                verify_statistics(record[mode]["prepared_transfer_complete_solve"])
            hashes = record[mode]["terminal_hashes"]
            assert len(hashes) == 3 and len(set(hashes)) == 1
            if record["device"] == "cuda":
                transfer_hashes = record[mode]["transfer_terminal_hashes"]
                assert len(transfer_hashes) == 3 and len(set(transfer_hashes)) == 1
            check = record["accuracy"][mode]
            assert check["passed"] and check["conservation"]["passed"]
            assert check["finite"] and check["dtype"] == "float64"
            assert check["shape"] == [3, record["cells"]]
        accuracy = record["accuracy"]
        expected_eligible = (
            accuracy["eager"]["passed"]
            and accuracy["compiled"]["passed"]
            and accuracy["compiled_eager_maximum_absolute_difference"] <= 5.0e-11
            and accuracy["compiled_first_eager_maximum_absolute_difference"]
            <= 5.0e-11
            and accuracy["repeat_deterministic"]
            and accuracy["transfer_repeat_deterministic"]
            and accuracy["eager"]["solve_diagnostics"]["steps"]
            == accuracy["compiled"]["solve_diagnostics"]["steps"]
        )
        assert record["eligible"] == expected_eligible
    for record in payload["step_records"]:
        assert record["status"] == "completed" and record["worker_returncode"] == 0
        for mode in MODES:
            verify_statistics(record["modes"][mode]["resident_step"])
            if record["device"] == "cuda":
                verify_statistics(record["modes"][mode]["transfer_inclusive_step"])
            item = record["modes"][mode]
            assert item["finite"] and item["deterministic"]
            assert item["maximum_absolute_difference_from_eager"] <= 5.0e-11
            assert item["first_maximum_absolute_difference_from_eager"] <= 5.0e-11
            assert item["eligible"]
        assert record["eligible"]
    for key in ("cold_records", "shock_records"):
        for record in payload[key]:
            assert record["status"] == "completed"
            assert record["worker_returncode"] == 0
            assert math.isfinite(record["process_launch_to_exit_seconds"])
            assert record["process_launch_to_exit_seconds"] > 0.0
            assert record["eligible"]


def classification(ratio: float, band: float = 0.05) -> str:
    if ratio > 1.0 + band:
        return "fd_faster"
    if ratio < 1.0 / (1.0 + band):
        return "fv_faster"
    return "unresolved_within_band"


def aggregate_complete(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for method in METHODS:
        for device in DEVICES:
            for cells in COMPLETE_SIZES:
                selected = [
                    x
                    for x in records
                    if x["method"] == method
                    and x["device"] == device
                    and x["cells"] == cells
                ]
                for mode in MODES:
                    eligible = [x for x in selected if x["eligible"]]
                    medians = [
                        x[mode]["resident_complete_solve"]["median_seconds"]
                        for x in eligible
                    ]
                    transfer = [
                        x[mode]["prepared_transfer_complete_solve"]["median_seconds"]
                        for x in eligible
                        if device == "cuda"
                    ]
                    errors = [x["accuracy"][mode]["l2_error"] for x in eligible]
                    rss = [x["memory"]["peak_process_rss_bytes"] for x in eligible]
                    cuda_memory = [
                        x[mode]["cuda_memory"]["peak_allocated_bytes"]
                        for x in eligible
                        if device == "cuda"
                    ]
                    valid = len(medians) == 3 and max(errors) - min(errors) <= 2e-13
                    result.append(
                        {
                            "method": method,
                            "device": device,
                            "cells": cells,
                            "mode": mode,
                            "replicates": len(selected),
                            "eligible_replicates": len(medians),
                            "worker_median_seconds": medians,
                            "aggregate_median_seconds": statistics.median(medians)
                            if valid
                            else None,
                            "transfer_worker_median_seconds": transfer,
                            "transfer_aggregate_median_seconds": statistics.median(
                                transfer
                            )
                            if valid and device == "cuda"
                            else None,
                            "l2_error": statistics.median(errors) if errors else None,
                            "peak_process_rss_bytes": max(rss) if rss else None,
                            "peak_cuda_allocated_bytes": max(cuda_memory)
                            if cuda_memory
                            else None,
                            "eligible": valid,
                        }
                    )
    return result


def targets(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    boundaries = {
        "cpu_warm": ("cpu", "aggregate_median_seconds"),
        "cuda_state_resident_host_controlled": (
            "cuda",
            "aggregate_median_seconds",
        ),
        "cuda_prepared_transfer": (
            "cuda",
            "transfer_aggregate_median_seconds",
        ),
    }
    result = {}
    for boundary, (device, field) in boundaries.items():
        target_result = {}
        for target in ERROR_TARGETS:
            methods = {}
            for method in METHODS:
                candidates = [
                    x
                    for x in aggregates
                    if x["method"] == method
                    and x["device"] == device
                    and x["eligible"]
                    and x["l2_error"] <= target
                    and x[field] is not None
                ]
                if candidates:
                    chosen = min(candidates, key=lambda x: x[field])
                    methods[method] = {
                        "status": "reached",
                        "cells": chosen["cells"],
                        "mode": chosen["mode"],
                        "l2_error": chosen["l2_error"],
                        "median_seconds": chosen[field],
                        "peak_process_rss_bytes": chosen["peak_process_rss_bytes"],
                        "peak_cuda_allocated_bytes": chosen[
                            "peak_cuda_allocated_bytes"
                        ],
                    }
                else:
                    methods[method] = {"status": "not_reached"}
            if all(methods[x]["status"] == "reached" for x in METHODS):
                ratio = methods["fv"]["median_seconds"] / methods["fd"][
                    "median_seconds"
                ]
                methods["fv_over_fd_time_ratio"] = ratio
                methods["classification"] = classification(ratio)
            target_result[str(target)] = methods
        result[boundary] = target_result
    return result


def replication_sizes(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    baseline = {
        (x["method"], x["device"], x["cells"]): x
        for x in records
        if x["replicate"] == 0 and x["eligible"]
    }
    result = {}
    for method in METHODS:
        winner = None
        for cells in STEP_SIZES:
            cpu = baseline[(method, "cpu", cells)]["modes"]["compiled"][
                "resident_step"
            ]["median_seconds"]
            cuda = baseline[(method, "cuda", cells)]["modes"]["compiled"][
                "resident_step"
            ]["median_seconds"]
            if cuda / cpu < 1.0 / 1.05:
                winner = cells
                break
        if winner is None:
            result[method] = list(STEP_SIZES[-2:])
        else:
            index = STEP_SIZES.index(winner)
            result[method] = sorted({STEP_SIZES[max(0, index - 1)], winner})
    return result


def aggregate_steps(
    records: list[dict[str, Any]], replication: dict[str, list[int]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = []
    for method in METHODS:
        for device in DEVICES:
            for cells in STEP_SIZES:
                selected = [
                    x
                    for x in records
                    if x["method"] == method
                    and x["device"] == device
                    and x["cells"] == cells
                    and x["eligible"]
                ]
                expected = 3 if cells in replication[method] else 1
                for mode in MODES:
                    medians = [
                        x["modes"][mode]["resident_step"]["median_seconds"]
                        for x in selected
                    ]
                    transfer = [
                        x["modes"][mode]["transfer_inclusive_step"]["median_seconds"]
                        for x in selected
                        if device == "cuda"
                    ]
                    valid = len(medians) == expected
                    result.append(
                        {
                            "method": method,
                            "device": device,
                            "cells": cells,
                            "mode": mode,
                            "expected_replicates": expected,
                            "eligible_replicates": len(medians),
                            "worker_median_seconds": medians,
                            "aggregate_median_seconds": statistics.median(medians)
                            if valid
                            else None,
                            "transfer_worker_median_seconds": transfer,
                            "transfer_aggregate_median_seconds": statistics.median(
                                transfer
                            )
                            if valid and device == "cuda"
                            else None,
                            "eligible": valid,
                        }
                    )
    lookup = {
        (x["method"], x["device"], x["cells"], x["mode"]): x for x in result
    }
    crossovers = {}
    for method in METHODS:
        candidate = None
        for cells in STEP_SIZES:
            cpu = lookup[(method, "cpu", cells, "compiled")]
            cuda = lookup[(method, "cuda", cells, "compiled")]
            if (
                cpu["eligible"]
                and cuda["eligible"]
                and cuda["aggregate_median_seconds"]
                / cpu["aggregate_median_seconds"]
                < 1.0 / 1.05
            ):
                candidate = cells
                break
        ratios = []
        if candidate in replication[method]:
            cpu_values = lookup[(method, "cpu", candidate, "compiled")][
                "worker_median_seconds"
            ]
            cuda_values = lookup[(method, "cuda", candidate, "compiled")][
                "worker_median_seconds"
            ]
            ratios = [cuda / cpu for cpu, cuda in zip(cpu_values, cuda_values)]
        confirmed = len(ratios) == 3 and all(x < 1.0 / 1.05 for x in ratios)
        crossovers[method] = {
            "baseline_winning_cells": candidate,
            "replicated_sizes": replication[method],
            "cuda_over_cpu_worker_median_ratios": ratios,
            "confirmed": confirmed,
            "decision": f"confirmed_at_n{candidate}" if confirmed else "unresolved",
        }
    return result, crossovers


def equal_steps(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (x["method"], x["device"], x["cells"], x["mode"]): x for x in aggregates
    }
    result = []
    for device in DEVICES:
        for cells in STEP_SIZES:
            for mode in MODES:
                fd = lookup[("fd", device, cells, mode)]
                fv = lookup[("fv", device, cells, mode)]
                if fd["eligible"] and fv["eligible"]:
                    ratio = fv["aggregate_median_seconds"] / fd[
                        "aggregate_median_seconds"
                    ]
                    result.append(
                        {
                            "device": device,
                            "cells": cells,
                            "mode": mode,
                            "fv_over_fd_ratio": ratio,
                            "classification": classification(ratio),
                        }
                    )
    return result


def cold_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (x["method"], x["device"], x["mode"], x["cells"]): x
        for x in records
        if x["eligible"]
    }
    result = []
    for device in DEVICES:
        for mode in MODES:
            for cells in COLD_SIZES:
                fd = lookup.get(("fd", device, mode, cells))
                fv = lookup.get(("fv", device, mode, cells))
                if fd and fv:
                    ratio = fv["process_launch_to_exit_seconds"] / fd[
                        "process_launch_to_exit_seconds"
                    ]
                    result.append(
                        {
                            "device": device,
                            "mode": mode,
                            "cells": cells,
                            "fv_over_fd_ratio": ratio,
                            "classification": classification(ratio, 0.10).replace(
                                "within_band", "cold_pilot"
                            ),
                        }
                    )
    return result


def shock_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (x["problem"], x["method"], x["device"], x["mode"], x["cells"]): x
        for x in records
        if x["eligible"]
    }
    result = []
    for problem in ("sod", "shu_osher"):
        for device in DEVICES:
            for mode in MODES:
                for cells in SHOCK_SIZES:
                    fd = lookup.get((problem, "fd", device, mode, cells))
                    fv = lookup.get((problem, "fv", device, mode, cells))
                    if fd and fv:
                        result.append(
                            {
                                "problem": problem,
                                "device": device,
                                "mode": mode,
                                "cells": cells,
                                "fv_over_fd_ratio": fv[
                                    "process_launch_to_exit_seconds"
                                ]
                                / fd["process_launch_to_exit_seconds"],
                                "classification": "descriptive_single_observation",
                            }
                        )
    return result


def compare_exact_numbers(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            compare_exact_numbers(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare_exact_numbers(left, right)
    elif isinstance(expected, float):
        assert close(actual, expected), (actual, expected)
    else:
        assert actual == expected


def main() -> None:
    payload = json.loads(RECORD.read_text())
    assert payload["phase"] == "fd_fv_euler_phase_6c"
    assert payload["source_commit"] == EXPECTED_SOURCE_COMMIT
    assert payload["protocol_commit"] == EXPECTED_PROTOCOL_COMMIT
    assert payload["source_dirty"] is False
    assert payload["matrix"] == {
        "complete_sizes": list(COMPLETE_SIZES),
        "step_sizes": list(STEP_SIZES),
        "cold_sizes": list(COLD_SIZES),
        "error_targets": list(ERROR_TARGETS),
        "shock_sizes": list(SHOCK_SIZES),
    }
    for name, digest in payload["source_hashes"].items():
        assert sha256(ROOT / name) == digest
    verification = subprocess.run(
        (sys.executable, str(ROOT / "experiments/fd_fv_euler/verify_phase6b.py")),
        cwd=ROOT,
        check=False,
    )
    assert verification.returncode == 0
    assert payload["admission"]["passed"]
    assert payload["admission"]["cuda_status"] == "admitted"
    verify_files(payload)
    verify_worker_records(payload)
    complete = aggregate_complete(payload["complete_records"])
    compare_exact_numbers(payload["complete_aggregates"], complete)
    compare_exact_numbers(payload["target_selections"], targets(complete))
    replication = replication_sizes(payload["step_records"])
    assert payload["step_replication_sizes"] == replication
    steps, crossovers = aggregate_steps(payload["step_records"], replication)
    compare_exact_numbers(payload["step_aggregates"], steps)
    compare_exact_numbers(payload["step_device_crossovers"], crossovers)
    compare_exact_numbers(payload["equal_grid_step_comparisons"], equal_steps(steps))
    compare_exact_numbers(
        payload["cold_comparisons"], cold_comparisons(payload["cold_records"])
    )
    compare_exact_numbers(
        payload["shock_comparisons"], shock_comparisons(payload["shock_records"])
    )
    assert all(
        record["diagnostics"]["completed"]
        and all(record["gate_decisions"].values())
        and record["host_visible_answer"]
        for record in payload["shock_records"]
    )
    assert all(payload[name] for name in (
        "all_complete_cells_eligible",
        "all_step_cells_eligible",
        "all_cold_cells_eligible",
        "all_shock_cells_eligible",
    ))
    assert payload["prepared_aot"] == {"status": "not_implemented"}
    assert payload["performance_measurements_collected"] is True
    assert payload["dveb_modified"] is False
    assert payload["phase_6d_begun"] is False
    assert payload["publication_claim"] is False
    print("FD/FV Euler Phase 6C verification passed")


if __name__ == "__main__":
    main()
