#!/usr/bin/env python3
"""Verify Phase-4R replication samples, derivations, identities, and hashes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "experiments/fd_fv_bakeoff/results/phase_4r_20260827"
RECORD_PATH = RESULT_DIR / "replication.json"
SOURCE_COMMIT = "75cb329f948736a2513dafdac143de1479b2ef83"
SIZES = (18, 21, 24, 27, 30, 33, 36, 40, 48)
METHODS = ("fd", "fv")
COMPILER_SUMMARY_FIELDS = (
    "generated_kernel_count",
    "generated_cpp_vec_kernel_count",
    "ir_nodes_pre_fusion",
    "num_bytes_accessed",
    "num_loop_reordering",
    "num_auto_chunking",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repetitions(cells: int) -> int:
    return 3 if cells == 27 else 2


def quantile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def classification(ratio: float) -> str:
    if ratio > 1.05:
        return "fd_faster"
    if ratio < 1.0 / 1.05:
        return "fv_faster"
    return "unresolved_within_5_percent"


def assert_close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-15)


def verify_statistics(record: dict, repetitions_: int) -> None:
    samples = record["samples_seconds"]
    assert len(samples) == repetitions_
    assert all(math.isfinite(value) and value > 0.0 for value in samples)
    ordered = sorted(samples)
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    expected = {
        "median_seconds": median,
        "mean_seconds": mean,
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
        "q1_seconds": quantile(ordered, 0.25),
        "q3_seconds": quantile(ordered, 0.75),
        "median_absolute_deviation_seconds": statistics.median(
            abs(value - median) for value in samples
        ),
        "coefficient_of_variation": statistics.pstdev(samples) / mean,
    }
    for name, value in expected.items():
        assert_close(record[name], value)


def main() -> None:
    payload = json.loads(RECORD_PATH.read_text())
    assert payload["schema_version"] == 1
    assert payload["phase"] == "fd_fv_phase_4r"
    assert payload["replication_date"] == "2026-08-27"
    assert payload["protocol_commit"] == "037e980"
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["source_dirty"] is False
    assert payload["performance_measurements_collected"] is True
    assert tuple(payload["sizes"]) == SIZES

    assert payload["prior_hashes"] == {
        "phase_4a": "6fd933cb44b1aa9350dd3c52cd7d446e182dce7411f93ba4cd8e6b3a0abe5362",
        "phase_4b": "056f61997b13faddd36f0d80dd541da8c0cb5da6ffda8a0951b48b38af25737a",
    }
    for verification in payload["prior_verification"].values():
        assert verification["passed"] is True
        assert verification["returncode"] == 0
        assert verification["stderr"] == ""
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected

    environment = payload["environment"]
    assert environment["primary_intraop_threads"] == 6
    assert environment["interop_threads"] == 1
    assert environment["visible_logical_cpus"] == 12
    assert environment["cuda_admission"]["available"] is False
    assert environment["cuda_admission"]["status"] == "untested_unavailable"

    records = payload["raw_records"]
    expected_keys = {
        (method, cells, replicate)
        for method in METHODS
        for cells in SIZES
        for replicate in range(repetitions(cells))
    }
    actual_keys = {
        (record["method"], record["cells_per_axis"], record["replicate"])
        for record in records
    }
    assert len(records) == 38
    assert actual_keys == expected_keys
    by_key = {}
    timing_sample_count = 0
    for record in records:
        key = (record["method"], record["cells_per_axis"], record["replicate"])
        by_key[key] = record
        assert record["status"] == "completed"
        assert record["worker_returncode"] == 0
        assert "Traceback" not in record["worker_stderr"]
        assert record["eligible"] is True
        assert record["device"] == "cpu"
        assert record["dimension"] == 3
        assert record["dtype"] == "float64"
        assert record["logical_cells"] == record["cells_per_axis"] ** 3
        assert record["formulation_id"] == {
            "fd": "fd_classical_js5_global_lf_periodic_v1",
            "fv": "fv_dimensional_js5_global_lf_periodic_v1",
        }[record["method"]]

        controls = record["controls"]
        assert controls["primary_intraop_threads"] == 6
        assert controls["interop_threads"] == 1
        assert controls["eager_warmups"] == 10
        assert controls["eager_repetitions"] == 30
        assert controls["compiled_warmups"] == 10
        assert controls["compiled_repetitions"] == 50
        assert controls["visible_logical_cpus"] == 12
        assert controls["process_affinity"] == list(range(12))

        correctness = record["correctness"]
        assert correctness["compiled_eager_maximum_absolute_difference"] <= 2.0e-11
        assert correctness["eager_repeat_maximum_absolute_difference"] == 0.0
        assert correctness["compiled_repeat_maximum_absolute_difference"] == 0.0
        for field in ("finite", "shape_preserved", "dtype_preserved", "device_preserved"):
            assert correctness[field] is True
        assert record["graph"]["graph_count"] == 1
        assert record["graph"]["graph_break_count"] == 0
        assert record["graph"]["break_reasons"] == []

        verify_statistics(record["timing"]["eager"], 30)
        verify_statistics(record["timing"]["compiled"], 50)
        timing_sample_count += 80
        assert record["timing"]["first_compiled_call_seconds"] > 0.0
        assert record["peak_process_rss_bytes"] > 0
        assert record["total_worker_process_seconds"] > 0.0

        metrics = record["compiler_metrics"]
        expected_metrics = {
            "fd": (39, 36, 228),
            "fv": (51, 47, 240),
        }[record["method"]]
        assert (
            metrics["generated_kernel_count"],
            metrics["generated_cpp_vec_kernel_count"],
            metrics["ir_nodes_pre_fusion"],
        ) == expected_metrics
        assert metrics["num_loop_reordering"] == 0
        assert metrics["num_auto_chunking"] == 0
        assert metrics["parallel_reduction_count"] == 0
        assert metrics["cpp_to_dtype_count"] == 0
        assert metrics["cpp_outer_loop_fused_inner_counts"] == []

        cache = record["cache_evidence"]
        assert cache["status"] == "recorded"
        assert cache["cpp_file_count"] == 7
        assert len(cache["files"]) == 7
        assert cache["cpp_total_bytes"] == sum(item["bytes"] for item in cache["files"])
        assert cache["cpp_total_lines"] == sum(item["lines"] for item in cache["files"])
        assert cache["total_cache_bytes"] > cache["cpp_total_bytes"]
        assert len(record["compiled_profile"]) > 0

        if record["cells_per_axis"] == 27:
            assert controls["thread_counts"] == [1, 2, 3, 6, 12]
            assert controls["thread_warmups"] == 5
            assert controls["thread_repetitions"] == 30
            assert set(record["thread_sweep"]) == {"1", "2", "3", "6", "12"}
            for modes in record["thread_sweep"].values():
                verify_statistics(modes["eager"], 30)
                verify_statistics(modes["compiled"], 30)
                timing_sample_count += 60
        else:
            assert record["thread_sweep"] is None
            assert controls["thread_counts"] is None
            assert controls["thread_warmups"] is None
            assert controls["thread_repetitions"] is None

        raw_path = RESULT_DIR / "raw" / (
            f"cpu_{record['method']}_3d_n{record['cells_per_axis']}_"
            f"r{record['replicate']}.json"
        )
        assert json.loads(raw_path.read_text()) == record

    assert payload["all_cpu_cells_eligible"] is True
    assert len(payload["size_summaries"]) == len(SIZES)
    for summary, cells in zip(payload["size_summaries"], SIZES):
        assert summary["cells_per_axis"] == cells
        assert summary["logical_cells"] == cells**3
        assert summary["replicates"] == repetitions(cells)
        assert summary["eligible"] is True
        selected = {
            method: [by_key[(method, cells, rep)] for rep in range(repetitions(cells))]
            for method in METHODS
        }
        medians = {
            method: [record["timing"]["compiled"]["median_seconds"] for record in rows]
            for method, rows in selected.items()
        }
        assert summary["fd_worker_medians_seconds"] == medians["fd"]
        assert summary["fv_worker_medians_seconds"] == medians["fv"]
        fd_process = statistics.median(medians["fd"])
        fv_process = statistics.median(medians["fv"])
        assert_close(summary["fd_process_median_seconds"], fd_process)
        assert_close(summary["fv_process_median_seconds"], fv_process)
        paired = [fv / fd for fd, fv in zip(medians["fd"], medians["fv"])]
        for actual, expected in zip(summary["paired_fv_over_fd_ratios"], paired):
            assert_close(actual, expected)
        ratio = fv_process / fd_process
        assert_close(summary["fv_over_fd_process_median_ratio"], ratio)
        assert summary["classification"] == classification(ratio)
        for method in METHODS:
            compiler = summary["compiler_evidence"][method]
            for field in COMPILER_SUMMARY_FIELDS:
                assert compiler[field] == [
                    record["compiler_metrics"][field] for record in selected[method]
                ]
            assert compiler["cpp_file_count"] == [7] * repetitions(cells)
            assert compiler["cpp_total_bytes"] == [
                record["cache_evidence"]["cpp_total_bytes"]
                for record in selected[method]
            ]

    n27 = next(item for item in payload["size_summaries"] if item["cells_per_axis"] == 27)
    expected_strong = (
        sum(value < 0.5 for value in n27["paired_fv_over_fd_ratios"]) >= 2
        and n27["fv_over_fd_process_median_ratio"] < 0.5
    )
    assert expected_strong is False
    assert payload["n27_strong_replication"] is expected_strong
    below_eight = [
        item["cells_per_axis"]
        for item in payload["size_summaries"]
        if item["fv_over_fd_process_median_ratio"] < 0.8
    ]
    assert payload["transition_below_0_8"] == {
        "sampled_sizes": below_eight,
        "first_sampled_size": below_eight[0],
        "last_sampled_size": below_eight[-1],
    }
    assert below_eight == [27]

    assert payload["cuda_replication"] == {
        "status": "untested_unavailable",
        "measurements_collected": False,
    }

    manifest_entries = {}
    for line in (RESULT_DIR / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert relative not in manifest_entries
        manifest_entries[relative] = expected
        assert sha256(RESULT_DIR / relative) == expected
    expected_manifest = {
        "replication.json",
        *{f"raw/{path.name}" for path in (RESULT_DIR / "raw").glob("*.json")},
    }
    assert set(manifest_entries) == expected_manifest
    assert len(manifest_entries) == 39
    print(
        "FD/FV Phase 4R verified: 38 eligible CPU workers, "
        f"{timing_sample_count:,} recomputed timing samples, strong N=27 "
        "replication failed, and CUDA remained untested unavailable."
    )


if __name__ == "__main__":
    main()
