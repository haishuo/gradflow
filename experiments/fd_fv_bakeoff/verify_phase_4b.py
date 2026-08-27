#!/usr/bin/env python3
"""Verify raw samples, derivations, hashes, and eligibility for Phase 4B."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "experiments/fd_fv_bakeoff/results/phase_4b_20260827"
RECORD_PATH = RESULT_DIR / "benchmark.json"
SOURCE_COMMIT = "5736a8d4f1673a5cb7a42914d0942e822c90ec4b"
SIZES = {
    1: (24, 36, 54, 81),
    2: (12, 18, 27, 40),
    3: (8, 12, 18, 27),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]


def classification(ratio: float) -> str:
    if ratio > 1.05:
        return "fd_faster"
    if ratio < 1.0 / 1.05:
        return "fv_faster"
    return "unresolved_within_5_percent"


def verify_statistics(record: dict, repetitions: int) -> None:
    samples = record["samples_seconds"]
    assert len(samples) == repetitions
    assert all(math.isfinite(value) and value > 0.0 for value in samples)
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
        assert math.isclose(record[name], value, rel_tol=0.0, abs_tol=1.0e-15)


def main() -> None:
    payload = json.loads(RECORD_PATH.read_text())
    assert payload["schema_version"] == 1
    assert payload["phase"] == "fd_fv_phase_4b"
    assert payload["protocol_commit"] == "6dbd4d1"
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["source_dirty"] is False
    assert payload["performance_measurements_collected"] is True
    assert payload["phase_4a"]["passed"] is True
    assert payload["phase_4a"]["performance_measurements_collected"] is False
    assert payload["phase_4a"]["source_commit"] == (
        "7ff5708449d2b5e833a33cbf017a7ce98f5e272d"
    )

    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected

    records = payload["warm_records"]
    expected_keys = {
        (method, dimension, cells)
        for method in ("fd", "fv")
        for dimension, sizes in SIZES.items()
        for cells in sizes
    }
    actual_keys = {
        (record["method"], record["dimension"], record["cells_per_axis"])
        for record in records
    }
    assert len(records) == 24
    assert actual_keys == expected_keys
    by_key = {}
    for record in records:
        key = (record["method"], record["dimension"], record["cells_per_axis"])
        by_key[key] = record
        assert record["status"] == "completed"
        assert record["worker_returncode"] == 0
        assert record["finite"] is True
        assert record["eligible"] is True
        assert record["accuracy"][
            "compiled_eager_maximum_absolute_difference"
        ] <= 2.0e-11
        assert record["conservation"]["eager_passed"] is True
        assert record["conservation"]["compiled_passed"] is True
        assert record["controls"]["torch_intraop_threads"] == 6
        assert record["controls"]["torch_interop_threads"] == 1
        assert record["logical_cells"] == (
            record["cells_per_axis"] ** record["dimension"]
        )
        assert record["persistent_state_bytes"] == record["logical_cells"] * 8
        assert record["memory"]["peak_process_rss_bytes"] > 0
        assert record["memory"]["compiler_cache_bytes"] > 0
        for mode in ("eager", "compiled"):
            verify_statistics(record[mode]["complete_solve"], 5)
            verify_statistics(record[mode]["ssp_rk3_step"], 30)
        assert record["compiled"]["first_complete_solve_seconds"] > 0.0

        raw_path = RESULT_DIR / "raw" / (
            f"cpu_{record['method']}_{record['dimension']}d_"
            f"n{record['cells_per_axis']}.json"
        )
        assert json.loads(raw_path.read_text()) == record

    assert payload["all_warm_cells_eligible"] is True
    comparisons = payload["comparisons"]
    assert len(comparisons) == 12
    for comparison in comparisons:
        dimension = comparison["dimension"]
        cells = comparison["cells_per_axis"]
        fd = by_key[("fd", dimension, cells)]
        fv = by_key[("fv", dimension, cells)]
        assert comparison["eligible"] is True
        for mode in ("eager", "compiled"):
            solve_ratio = (
                fv[mode]["complete_solve"]["median_seconds"]
                / fd[mode]["complete_solve"]["median_seconds"]
            )
            step_ratio = (
                fv[mode]["ssp_rk3_step"]["median_seconds"]
                / fd[mode]["ssp_rk3_step"]["median_seconds"]
            )
            actual = comparison["matched_modes"][mode]
            assert math.isclose(
                actual["fv_over_fd_complete_solve_ratio"],
                solve_ratio,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            assert actual["complete_solve_classification"] == classification(
                solve_ratio
            )
            assert math.isclose(
                actual["fv_over_fd_step_ratio"],
                step_ratio,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            assert actual["step_classification"] == classification(step_ratio)
        best = comparison["best_practical"]
        for method, record in (("fd", fd), ("fv", fv)):
            medians = {
                mode: record[mode]["complete_solve"]["median_seconds"]
                for mode in ("eager", "compiled")
            }
            selected = min(medians, key=medians.get)
            assert best[method]["mode"] == selected
            assert best[method]["median_seconds"] == medians[selected]
        best_ratio = best["fv"]["median_seconds"] / best["fd"]["median_seconds"]
        assert math.isclose(
            best["fv_over_fd_complete_solve_ratio"],
            best_ratio,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        assert best["classification"] == classification(best_ratio)

    cold = payload["cold_records"]
    assert len(cold) == 6
    assert {
        (record["method"], record["dimension"], record["cells_per_axis"])
        for record in cold
    } == {
        (method, dimension, sizes[-1])
        for method in ("fd", "fv")
        for dimension, sizes in SIZES.items()
    }
    for record in cold:
        assert record["status"] == "completed"
        assert record["worker_returncode"] == 0
        assert record["eligible"] is True
        assert record["host_visible_answer"] is True
        assert record["process_launch_to_exit_seconds"] > 0.0
        assert record["peak_process_rss_bytes"] > 0
        raw_path = RESULT_DIR / "raw" / (
            f"cold_cpu_{record['method']}_{record['dimension']}d_"
            f"n{record['cells_per_axis']}.json"
        )
        assert json.loads(raw_path.read_text()) == record
    assert payload["all_cold_cells_eligible"] is True
    assert payload["prepared_aot"]["status"] == "not_implemented"
    assert payload["cuda_measurements"]["status"] == (
        "not_collected_unavailable"
    )

    manifest_entries = {}
    for line in (RESULT_DIR / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert relative not in manifest_entries
        manifest_entries[relative] = expected
        assert sha256(RESULT_DIR / relative) == expected
    assert set(manifest_entries) == {
        "benchmark.json",
        *{
            f"raw/{path.name}" for path in (RESULT_DIR / "raw").glob("*.json")
        },
    }
    assert len(manifest_entries) == 31
    print(
        "FD/FV Phase 4B verified: 24 eligible warm cells, 6 eligible cold "
        "pilots, 1,680 recomputed timing samples, and no CUDA measurements."
    )


if __name__ == "__main__":
    main()
