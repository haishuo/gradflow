#!/usr/bin/env python3
"""Verify the immutable initial Phase-5C campaign, including failed gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.fd_fv_nonlinear.performance_problem import (
    quantile,
)
from experiments.fd_fv_nonlinear.run_phase5c import (
    COLD_SIZES,
    COMPLETE_SIZES,
    DEVICES,
    METHODS,
    MODES,
    STEP_SIZES,
    aggregate_complete,
    aggregate_steps,
    equal_grid_step_comparisons,
    replication_sizes,
    target_selections,
)


RESULTS = ROOT / "experiments/fd_fv_nonlinear/results/phase_5c_20260828"
RECORD = RESULTS / "benchmark.json"
SOURCE_COMMIT = "7b0a989a8cad1b02ebd5a67446e2336c4a25675a"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    for key, value in expected.items():
        assert math.isclose(record[key], value, rel_tol=0.0, abs_tol=1.0e-15)


def raw_equals(record: dict, name: str) -> None:
    assert json.loads((RESULTS / "raw" / name).read_text()) == record


def main() -> None:
    payload = json.loads(RECORD.read_text())
    assert payload["schema_version"] == 1
    assert payload["phase"] == "fd_fv_nonlinear_phase_5c"
    assert payload["protocol_commit"] == "1bc340c"
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["source_dirty"] is False
    assert payload["performance_measurements_collected"] is True
    assert payload["admission"]["passed"] is True
    assert payload["admission"]["cuda_status"] == "admitted"
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected

    complete = payload["complete_records"]
    assert len(complete) == 60
    assert {
        (record["method"], record["device"], record["cells"], record["replicate"])
        for record in complete
    } == {
        (method, device, cells, replicate)
        for method in METHODS
        for device in DEVICES
        for cells in COMPLETE_SIZES
        for replicate in range(3)
    }
    ineligible_complete = set()
    for record in complete:
        assert record["status"] == "completed"
        assert record["worker_returncode"] == 0
        assert record["steps"] > 0
        assert record["controls"]["complete_repetitions"] == 3
        for mode in MODES:
            verify_statistics(record[mode]["resident_complete_solve"], 3)
            if record["device"] == "cuda":
                verify_statistics(
                    record[mode]["prepared_transfer_complete_solve"], 3
                )
            else:
                assert record[mode]["prepared_transfer_complete_solve"] is None
        eager = record["accuracy"]["eager"]
        compiled = record["accuracy"]["compiled"]
        expected_eligible = (
            eager["finite"]
            and compiled["finite"]
            and eager["conservation_passed"]
            and compiled["conservation_passed"]
            and record["accuracy"][
                "compiled_eager_maximum_absolute_difference"
            ]
            <= 2.0e-11
            and record["accuracy"][
                "compiled_repeat_maximum_absolute_difference"
            ]
            == 0.0
            and eager["dtype"] == "float64"
            and compiled["dtype"] == "float64"
        )
        assert record["eligible"] is expected_eligible
        if not record["eligible"]:
            ineligible_complete.add(
                (
                    record["method"],
                    record["device"],
                    record["cells"],
                    record["replicate"],
                )
            )
        raw_equals(
            record,
            f"complete_{record['device']}_{record['method']}_"
            f"n{record['cells']}_r{record['replicate']}.json",
        )
    assert ineligible_complete == {
        (method, "cuda", cells, replicate)
        for method in METHODS
        for cells in (81, 162)
        for replicate in range(3)
    }
    assert payload["all_complete_cells_eligible"] is False
    assert aggregate_complete(complete) == payload["complete_aggregates"]
    assert target_selections(payload["complete_aggregates"]) == payload[
        "target_selections"
    ]

    steps = payload["step_records"]
    assert len(steps) == 48
    for record in steps:
        assert record["status"] == "completed"
        assert record["worker_returncode"] == 0
        assert record["eligible"] is True
        for mode in MODES:
            verify_statistics(record["modes"][mode]["resident_step"], 30)
            if record["device"] == "cuda":
                verify_statistics(
                    record["modes"][mode]["transfer_inclusive_step"], 20
                )
        raw_equals(
            record,
            f"step_{record['device']}_{record['method']}_"
            f"n{record['cells']}_r{record['replicate']}.json",
        )
    assert payload["all_step_cells_eligible"] is True
    expected_replication = replication_sizes(steps)
    assert expected_replication == payload["step_replication_sizes"]
    aggregates, crossovers = aggregate_steps(steps, expected_replication)
    assert aggregates == payload["step_aggregates"]
    assert crossovers == payload["step_device_crossovers"]
    assert equal_grid_step_comparisons(aggregates) == payload[
        "equal_grid_step_comparisons"
    ]

    cold = payload["cold_records"]
    assert len(cold) == 24
    assert {
        (record["method"], record["device"], record["mode"], record["cells"])
        for record in cold
    } == {
        (method, device, mode, cells)
        for method in METHODS
        for device in DEVICES
        for mode in MODES
        for cells in COLD_SIZES
    }
    ineligible_cold = set()
    for record in cold:
        assert record["status"] == "completed"
        assert record["worker_returncode"] == 0
        assert record["process_launch_to_exit_seconds"] > 0.0
        expected_eligible = (
            record["finite"]
            and record["conservation_passed"]
            and record["host_visible_answer"]
        )
        assert record["eligible"] is expected_eligible
        if not record["eligible"]:
            ineligible_cold.add(
                (
                    record["method"],
                    record["device"],
                    record["mode"],
                    record["cells"],
                )
            )
        raw_equals(
            record,
            f"cold_{record['device']}_{record['method']}_"
            f"{record['mode']}_n{record['cells']}.json",
        )
    assert ineligible_cold == {
        (method, "cuda", mode, cells)
        for method in METHODS
        for mode in MODES
        for cells in (81, 162)
    }
    assert payload["all_cold_cells_eligible"] is False
    assert payload["prepared_aot"] == {"status": "not_implemented"}

    manifest = {}
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert relative not in manifest
        manifest[relative] = expected
        assert sha256(RESULTS / relative) == expected
    assert len(manifest) == 133
    assert set(manifest) == {
        "benchmark.json",
        *{f"raw/{path.name}" for path in (RESULTS / "raw").glob("*.json")},
    }
    print(
        "Initial Phase 5C verified: 132 worker records and all timing samples "
        "are intact; 12 complete and 8 cold CUDA cells remain ineligible under "
        "the frozen full-solve conservation bound."
    )


if __name__ == "__main__":
    main()
