#!/usr/bin/env python3
"""Verify the Phase-4R Forge CUDA supplement and all event samples."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "experiments/fd_fv_bakeoff/results/phase_4r_cuda_20260828"
RECORD_PATH = RESULT_DIR / "replication_cuda.json"
SOURCE_COMMIT = "ba646aa757d2fee63a9c7369ed106571c3f699b9"
CPU_RECORD_SHA256 = (
    "17d52fd8bf851d0fad87497857c5c30e6e3e378426a7ddeb57c2414d59d20fff"
)
SIZES = (18, 27, 40, 64)
METHODS = ("fd", "fv")
REPLICATES = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def verify_statistics(record: dict) -> None:
    samples = record["samples_seconds"]
    assert len(samples) == 50
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
    assert record["peak_allocated_bytes"] > 0
    assert record["peak_reserved_bytes"] >= record["peak_allocated_bytes"]


def main() -> None:
    payload = json.loads(RECORD_PATH.read_text())
    assert payload["schema_version"] == 1
    assert payload["phase"] == "fd_fv_phase_4r_cuda_supplement"
    assert payload["protocol_commit"] == "037e980"
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["source_dirty"] is False
    assert payload["cpu_replication_sha256"] == CPU_RECORD_SHA256
    assert tuple(payload["sizes"]) == SIZES
    assert payload["replicates"] == REPLICATES
    assert payload["performance_measurements_collected"] is True
    assert payload["timing_scope"] == (
        "device_resident_cuda_events_excludes_transfers_and_compilation"
    )
    assert payload["cpu_verification"]["passed"] is True
    assert payload["cpu_verification"]["returncode"] == 0
    assert payload["cpu_verification"]["stderr"] == ""
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected

    environment = payload["environment"]["cuda"]
    assert environment["device"] == "NVIDIA GeForce RTX 5070 Ti"
    assert environment["device_capability"] == [12, 0]
    assert environment["device_total_memory_bytes"] == 16609247232
    assert environment["multiprocessor_count"] == 70
    assert environment["cuda_runtime"] == "13.0"
    assert environment["cuda_driver"] == "580.173.02"

    admission = payload["fresh_admission"]
    assert json.loads((RESULT_DIR / "admission.json").read_text()) == admission
    assert admission["available"] is True
    assert admission["status"] == "passed"
    assert admission["passed"] is True
    assert admission["performance_measurements_collected"] is False
    assert admission["environment"] == environment
    assert len(admission["cases"]) == 6
    assert {
        (case["method"], case["dimension"], case["cells_per_axis"])
        for case in admission["cases"]
    } == {
        (method, dimension, cells)
        for method in METHODS
        for dimension, cells in ((1, 81), (2, 40), (3, 27))
    }
    for case in admission["cases"]:
        assert case["passed"] is True
        assert case["resident"] is True
        assert case["finite"] is True
        assert case["graph_count"] == 1
        assert case["graph_break_count"] == 0
        assert case["cpu_eager_gpu_eager_maximum_absolute_difference"] <= 2.0e-11
        assert case["compiled_eager_maximum_absolute_difference"] <= 2.0e-11

    records = payload["raw_records"]
    expected_keys = {
        (method, cells, replicate)
        for method in METHODS
        for cells in SIZES
        for replicate in range(REPLICATES)
    }
    actual_keys = {
        (record["method"], record["cells_per_axis"], record["replicate"])
        for record in records
    }
    assert len(records) == 24
    assert actual_keys == expected_keys
    by_key = {}
    for record in records:
        key = (record["method"], record["cells_per_axis"], record["replicate"])
        by_key[key] = record
        assert record["status"] == "completed"
        assert record["worker_returncode"] == 0
        assert record["worker_stdout"] == ""
        assert record["worker_stderr"] == ""
        assert record["eligible"] is True
        assert record["dimension"] == 3
        assert record["logical_cells"] == record["cells_per_axis"] ** 3
        assert record["device"] == "cuda:0"
        assert record["dtype"] == "float64"
        assert record["resident_timed_region"] is True
        assert record["timing_method"] == "cuda_events"
        assert record["controls"] == {"warmups": 10, "repetitions": 50}
        assert record["environment"] == environment
        assert record["formulation_id"] == {
            "fd": "fd_classical_js5_global_lf_periodic_v1",
            "fv": "fv_dimensional_js5_global_lf_periodic_v1",
        }[record["method"]]

        graph = record["graph"]
        assert graph["graph_count"] == 1
        assert graph["graph_break_count"] == 0
        assert graph["break_reasons"] == []
        correctness = record["correctness"]
        assert correctness["cpu_eager_gpu_eager_maximum_absolute_difference"] <= 2.0e-11
        assert correctness["compiled_eager_maximum_absolute_difference"] <= 2.0e-11
        assert correctness["eager_repeat_maximum_absolute_difference"] == 0.0
        assert correctness["compiled_repeat_maximum_absolute_difference"] == 0.0
        for field in (
            "finite",
            "shape_preserved",
            "dtype_preserved",
            "device_preserved",
            "conservation_passed",
        ):
            assert correctness[field] is True
        assert correctness["conservation_mass_change"] <= correctness[
            "conservation_bound"
        ]

        verify_statistics(record["timing"]["eager"])
        verify_statistics(record["timing"]["compiled"])
        assert record["timing"]["first_compiled_call_seconds"] > 0.0
        assert record["peak_process_rss_bytes"] > 0
        assert record["total_worker_process_seconds"] > 0.0

        metrics = record["compiler_metrics"]
        assert (
            metrics["generated_kernel_count"],
            metrics["ir_nodes_pre_fusion"],
        ) == {"fd": (54, 229), "fv": (57, 187)}[record["method"]]
        assert metrics["generated_cpp_vec_kernel_count"] == 0
        assert metrics["num_loop_reordering"] == 0
        assert metrics["num_auto_chunking"] == 0
        assert metrics["parallel_reduction_count"] == 0
        assert metrics["cpp_to_dtype_count"] == 0

        cache = record["cache_evidence"]
        expected_cache = {"fd": (183, 11), "fv": (327, 19)}[record["method"]]
        assert cache["status"] == "recorded"
        assert (cache["file_count"], cache["text_file_count"]) == expected_cache
        assert len(cache["files"]) == cache["file_count"]
        assert cache["total_cache_bytes"] == sum(
            item["bytes"] for item in cache["files"]
        )
        assert cache["text_total_bytes"] == sum(
            item["bytes"] for item in cache["files"] if "text_counts" in item
        )

        raw_path = RESULT_DIR / "raw" / (
            f"cuda_{record['method']}_3d_n{record['cells_per_axis']}_"
            f"r{record['replicate']}.json"
        )
        assert json.loads(raw_path.read_text()) == record

    assert payload["all_cells_eligible"] is True
    assert len(payload["size_summaries"]) == len(SIZES)
    for summary, cells in zip(payload["size_summaries"], SIZES):
        assert summary["cells_per_axis"] == cells
        assert summary["logical_cells"] == cells**3
        assert summary["replicates"] == REPLICATES
        assert summary["eligible"] is True
        for mode in ("eager", "compiled"):
            actual = summary["modes"][mode]
            medians = {
                method: [
                    by_key[(method, cells, replicate)]["timing"][mode][
                        "median_seconds"
                    ]
                    for replicate in range(REPLICATES)
                ]
                for method in METHODS
            }
            assert actual["fd_worker_medians_seconds"] == medians["fd"]
            assert actual["fv_worker_medians_seconds"] == medians["fv"]
            fd_process = statistics.median(medians["fd"])
            fv_process = statistics.median(medians["fv"])
            assert_close(actual["fd_process_median_seconds"], fd_process)
            assert_close(actual["fv_process_median_seconds"], fv_process)
            paired = [fv / fd for fd, fv in zip(medians["fd"], medians["fv"])]
            for observed, expected in zip(actual["paired_fv_over_fd_ratios"], paired):
                assert_close(observed, expected)
            ratio = fv_process / fd_process
            assert_close(actual["fv_over_fd_process_median_ratio"], ratio)
            assert actual["classification"] == classification(ratio)

    manifest_entries = {}
    for line in (RESULT_DIR / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert relative not in manifest_entries
        manifest_entries[relative] = expected
        assert sha256(RESULT_DIR / relative) == expected
    expected_manifest = {
        "admission.json",
        "replication_cuda.json",
        *{f"raw/{path.name}" for path in (RESULT_DIR / "raw").glob("*.json")},
    }
    assert set(manifest_entries) == expected_manifest
    assert len(manifest_entries) == 26
    print(
        "FD/FV Phase 4R CUDA verified: fresh six-case admission and 24 "
        "eligible RTX 5070 Ti workers with 2,400 recomputed event samples."
    )


if __name__ == "__main__":
    main()
