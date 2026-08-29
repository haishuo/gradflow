#!/usr/bin/env python3
"""Independently verify the preserved FD/FV Euler Phase-6D campaign."""

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
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6d_20260829"
RECORD = RESULTS / "benchmark.json"
PHASE6C = ROOT / "experiments/fd_fv_euler/results/phase_6c_20260829/benchmark.json"
EXPECTED_TIMING_COMMIT = "7952c9fabbca5114994d457b563cc907c477db4e"
EXPECTED_AGGREGATION_COMMIT = "889fbc321f925c66fc12f2c863f8f8cb08a56c77"
EXPECTED_PROTOCOL_COMMIT = "0a919c5"
PRIMARY_SIZES = (2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768)
INTERACTION_SIZES = (4096, 8192, 32768)
PRIMARY_THREADS = (1, 6)
INTERMEDIATE_THREADS = (2, 4)
METHODS = ("fd", "fv")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha256(commit: str, relative: str) -> str:
    blob = subprocess.check_output(
        ("git", "show", f"{commit}:{relative}"), cwd=ROOT
    )
    return hashlib.sha256(blob).hexdigest()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)


def compare(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            compare(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare(left, right)
    elif isinstance(expected, float):
        assert close(actual, expected), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


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
    assert len(samples) == 30
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
        assert close(record[name], value), (name, record[name], value)


def raw_name(record: dict[str, Any]) -> str:
    if record["kind"] == "cpu_regime":
        return (
            f"cpu_{record['method']}_n{record['cells']}_"
            f"t{record['threads']}_r{record['replicate']}.json"
        )
    return (
        f"shock_{record['problem']}_{record['device']}_{record['method']}_"
        f"{record['mode']}_n{record['cells']}_r{record['replicate']}.json"
    )


def verify_files(payload: dict[str, Any]) -> None:
    listed = {}
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    files = [RECORD, *sorted((RESULTS / "raw").glob("*.json"))]
    assert listed == {
        str(path.relative_to(RESULTS)): sha256(path) for path in files
    }
    raw = {
        path.name: json.loads(path.read_text())
        for path in sorted((RESULTS / "raw").glob("*.json"))
    }
    embedded = payload["shock_records"] + payload["cpu_records"]
    assert len(raw) == len(embedded) == 92
    for record in embedded:
        assert raw[raw_name(record)] == record


def verify_workers(payload: dict[str, Any], phase6c: dict[str, Any]) -> None:
    reference = {
        (item["problem"], item["method"], item["device"], item["mode"]): item[
            "terminal_sha256"
        ]
        for item in phase6c["shock_records"]
        if item["cells"] == 800
    }
    for record in payload["shock_records"]:
        assert record["status"] == "completed" and record["worker_returncode"] == 0
        assert record["cells"] == 800 and record["host_visible_answer"]
        assert record["diagnostics"]["completed"]
        assert all(record["gate_decisions"].values())
        assert math.isfinite(record["process_launch_to_exit_seconds"])
        assert record["process_launch_to_exit_seconds"] > 0.0
        expected_hash = reference[
            (record["problem"], record["method"], record["device"], record["mode"])
        ]
        matches = record["terminal_sha256"] == expected_hash
        assert record["phase6c_terminal_sha256"] == expected_hash
        assert record["terminal_hash_matches_phase6c"] == matches
        assert record["eligible"] == matches

    for record in payload["cpu_records"]:
        assert record["status"] == "completed" and record["worker_returncode"] == 0
        assert record["device"] == "cpu" and record["dtype"] == "float64"
        assert record["shape"] == [3, record["cells"]]
        for mode in ("eager", "compiled"):
            verify_statistics(record[mode]["resident_step"])
            hashes = record[mode]["terminal_hashes"]
            assert len(hashes) == 30 and len(set(hashes)) == 1
        expected = (
            record["finite"]
            and record["deterministic"]
            and record["compiled_eager_maximum_absolute_difference"] <= 5.0e-11
            and record["compiled_first_eager_maximum_absolute_difference"]
            <= 5.0e-11
        )
        assert record["eligible"] == expected
        controls = record["controls"]
        assert controls["warmups"] == 10 and controls["repetitions"] == 30
        assert controls["torch_intraop_threads"] == record["threads"]
        assert controls["torch_interop_threads"] == 1


def expected_replicates(cells: int, threads: int) -> int:
    return 3 if cells in INTERACTION_SIZES and threads in PRIMARY_THREADS else 1


def cpu_aggregates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    configurations = sorted(
        {(item["method"], item["cells"], item["threads"]) for item in records}
    )
    for method, cells, threads in configurations:
        selected = sorted(
            (
                item
                for item in records
                if item["method"] == method
                and item["cells"] == cells
                and item["threads"] == threads
                and item["eligible"]
            ),
            key=lambda item: item["replicate"],
        )
        expected = expected_replicates(cells, threads)
        signatures = [
            {
                "compiler_metrics": item["compiler_metrics"],
                "cpp_totals": item["generated_cpp_inventory"]["totals"],
                "cpp_hashes": [
                    source["sha256"]
                    for source in item["generated_cpp_inventory"]["files"]
                ],
            }
            for item in selected
        ]
        structural = bool(signatures) and all(
            item["compiler_metrics"] == signatures[0]["compiler_metrics"]
            and item["cpp_totals"] == signatures[0]["cpp_totals"]
            for item in signatures[1:]
        )
        hash_reproducible = bool(signatures) and all(
            item["cpp_hashes"] == signatures[0]["cpp_hashes"]
            for item in signatures[1:]
        )
        entry = {
            "method": method,
            "cells": cells,
            "threads": threads,
            "expected_replicates": expected,
            "eligible_replicates": len(selected),
            "structural_reproducible": structural,
            "generated_hash_reproducible": hash_reproducible,
            "compiler_metrics": signatures[0]["compiler_metrics"] if signatures else None,
            "cpp_totals": signatures[0]["cpp_totals"] if signatures else None,
            "eligible": len(selected) == expected,
        }
        for mode in ("eager", "compiled"):
            medians = [item[mode]["resident_step"]["median_seconds"] for item in selected]
            entry[f"{mode}_worker_medians_seconds"] = medians
            entry[f"{mode}_aggregate_median_seconds"] = (
                statistics.median(medians) if len(medians) == expected else None
            )
        result.append(entry)
    return result


def shock_replication(records: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = {
        "sod": {"cpu": ("cpu", "eager"), "cuda": ("cuda", "compiled")},
        "shu_osher": {
            "cpu": ("cpu", "compiled"),
            "cuda": ("cuda", "compiled"),
        },
    }
    result = {}
    for problem, choices in endpoints.items():
        result[problem] = {}
        for method in METHODS:
            chosen = {}
            for side in ("cpu", "cuda"):
                chosen[side] = sorted(
                    (
                        item
                        for item in records
                        if item["problem"] == problem
                        and item["method"] == method
                        and (item["device"], item["mode"]) == choices[side]
                    ),
                    key=lambda item: item["replicate"],
                )
            ratios = [
                cuda["process_launch_to_exit_seconds"]
                / cpu["process_launch_to_exit_seconds"]
                for cpu, cuda in zip(chosen["cpu"], chosen["cuda"])
                if cpu["eligible"] and cuda["eligible"]
            ]
            confirmed = len(ratios) == 3 and all(value < 1.0 / 1.05 for value in ratios)
            cpu_seconds = [item["process_launch_to_exit_seconds"] for item in chosen["cpu"]]
            cuda_seconds = [item["process_launch_to_exit_seconds"] for item in chosen["cuda"]]
            result[problem][method] = {
                "cpu_endpoint": {"device": choices["cpu"][0], "mode": choices["cpu"][1]},
                "cuda_endpoint": {
                    "device": choices["cuda"][0],
                    "mode": choices["cuda"][1],
                },
                "cpu_seconds": cpu_seconds,
                "cuda_seconds": cuda_seconds,
                "cuda_over_cpu_paired_ratios": ratios,
                "aggregate_cuda_over_cpu_ratio": statistics.median(cuda_seconds)
                / statistics.median(cpu_seconds),
                "confirmed": confirmed,
                "decision": "confirmed_cuda_win" if confirmed else "unresolved",
            }
    return result


def signature(item: dict[str, Any]) -> list[int]:
    metrics = item["compiler_metrics"]
    cpp = item["cpp_totals"]
    return [
        metrics["generated_kernel_count"],
        metrics["generated_cpp_vec_kernel_count"],
        metrics["ir_nodes_pre_fusion"],
        cpp["cpp_files"],
        cpp["openmp_pragmas"],
        cpp["parallel_for_markers"],
        cpp["vectorized_markers"],
        cpp["gcc_ivdep_pragmas"],
    ]


def causal_summary(
    aggregates: list[dict[str, Any]], phase6c: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    lookup = {(item["method"], item["cells"], item["threads"]): item for item in aggregates}
    steps = {
        (item["method"], item["device"], item["cells"]): item
        for item in phase6c["step_aggregates"]
        if item["mode"] == "compiled" and item["eligible"]
    }
    points = {}
    for cells in PRIMARY_SIZES:
        point = {}
        for threads in PRIMARY_THREADS:
            fd = lookup[("fd", cells, threads)]
            fv = lookup[("fv", cells, threads)]
            fd_bytes = fd["compiler_metrics"]["num_bytes_accessed"]
            fv_bytes = fv["compiler_metrics"]["num_bytes_accessed"]
            point[f"threads_{threads}"] = {
                "compiled_fv_over_fd": fv["compiled_aggregate_median_seconds"]
                / fd["compiled_aggregate_median_seconds"],
                "eager_fv_over_fd": fv["eager_aggregate_median_seconds"]
                / fd["eager_aggregate_median_seconds"],
                "compiled_over_eager": {
                    method: lookup[(method, cells, threads)]["compiled_aggregate_median_seconds"]
                    / lookup[(method, cells, threads)]["eager_aggregate_median_seconds"]
                    for method in METHODS
                },
                "fv_over_fd_ir_nodes": fv["compiler_metrics"]["ir_nodes_pre_fusion"]
                / fd["compiler_metrics"]["ir_nodes_pre_fusion"],
                "fv_over_fd_estimated_bytes": fv_bytes / fd_bytes if fd_bytes > 0 else None,
                "fd_signature": signature(fd),
                "fv_signature": signature(fv),
            }
        point["thread_interaction_factor"] = (
            point["threads_6"]["compiled_fv_over_fd"]
            / point["threads_1"]["compiled_fv_over_fd"]
        )
        point["compiled_thread_speedup_t1_over_t6"] = {
            method: lookup[(method, cells, 1)]["compiled_aggregate_median_seconds"]
            / lookup[(method, cells, 6)]["compiled_aggregate_median_seconds"]
            for method in METHODS
        }
        if cells in (8192, 32768):
            point["phase6c_cuda_fv_over_fd"] = (
                steps[("fv", "cuda", cells)]["aggregate_median_seconds"]
                / steps[("fd", "cuda", cells)]["aggregate_median_seconds"]
            )
        points[str(cells)] = point

    sizes = [cells for cells in PRIMARY_SIZES if cells >= 8192]
    thread_flags = [points[str(cells)]["thread_interaction_factor"] >= 1.5 for cells in sizes]
    consecutive = any(left and right for left, right in zip(thread_flags, thread_flags[1:]))
    replicated = True
    for cells in (8192, 32768):
        factors = []
        by_key = {
            (item["method"], item["threads"], item["replicate"]): item
            for item in records
            if item["cells"] == cells and item["threads"] in PRIMARY_THREADS
        }
        for replicate in range(3):
            ratio1 = by_key[("fv", 1, replicate)]["compiled"]["resident_step"]["median_seconds"] / by_key[("fd", 1, replicate)]["compiled"]["resident_step"]["median_seconds"]
            ratio6 = by_key[("fv", 6, replicate)]["compiled"]["resident_step"]["median_seconds"] / by_key[("fd", 6, replicate)]["compiled"]["resident_step"]["median_seconds"]
            factors.append(ratio6 / ratio1)
        points[str(cells)]["replicated_thread_interaction_factors"] = factors
        replicated = replicated and all(value >= 1.5 for value in factors)
    thread_supported = consecutive and replicated
    traffic_flags = []
    for cells in sizes:
        item = points[str(cells)]["threads_6"]
        byte_ratio = item["fv_over_fd_estimated_bytes"]
        traffic_flags.append(
            (byte_ratio is not None and byte_ratio >= 1.5)
            or item["fv_over_fd_ir_nodes"] >= 1.5
        )
    traffic_supported = any(
        left and right for left, right in zip(traffic_flags, traffic_flags[1:])
    ) and all(points[str(cells)]["phase6c_cuda_fv_over_fd"] < 1.25 for cells in (8192, 32768))
    first_slow = next(
        (cells for cells in PRIMARY_SIZES if points[str(cells)]["threads_6"]["compiled_fv_over_fd"] > 2.0),
        None,
    )
    codegen_supported = False
    if first_slow is not None and PRIMARY_SIZES.index(first_slow) > 0:
        previous = PRIMARY_SIZES[PRIMARY_SIZES.index(first_slow) - 1]
        codegen_supported = (
            points[str(first_slow)]["threads_6"]["fv_signature"]
            != points[str(previous)]["threads_6"]["fv_signature"]
            and points[str(first_slow)]["threads_6"]["fd_signature"]
            == points[str(previous)]["threads_6"]["fd_signature"]
        )
    mechanisms = {
        "thread_interaction_supported": thread_supported,
        "traffic_expansion_supported": traffic_supported,
        "codegen_transition_supported": codegen_supported,
    }
    count = sum(mechanisms.values())
    return {
        "points": points,
        "first_six_thread_fv_over_fd_above_2": first_slow,
        **mechanisms,
        "supported_mechanism_count": count,
        "unresolved_mixture": count != 1,
        "classification": next((name for name, value in mechanisms.items() if value), "unresolved_mixture") if count == 1 else "unresolved_mixture",
        "estimated_bytes_metric_available": all(
            points[str(cells)][f"threads_{threads}"]["fv_over_fd_estimated_bytes"] is not None
            for cells in PRIMARY_SIZES
            for threads in PRIMARY_THREADS
        ),
    }


def main() -> None:
    payload = json.loads(RECORD.read_text())
    phase6c = json.loads(PHASE6C.read_text())
    assert payload["phase"] == "fd_fv_euler_phase_6d"
    assert payload["source_commit"] == EXPECTED_TIMING_COMMIT
    assert payload["aggregation_commit"] == EXPECTED_AGGREGATION_COMMIT
    assert payload["protocol_commit"] == EXPECTED_PROTOCOL_COMMIT
    assert payload["aggregation_reused_frozen_raw_records"] is True
    assert payload["source_dirty"] is False
    assert payload["matrix"] == {
        "shock_cells": 800,
        "shock_replicates": 3,
        "primary_sizes": list(PRIMARY_SIZES),
        "interaction_sizes": list(INTERACTION_SIZES),
        "primary_threads": list(PRIMARY_THREADS),
        "intermediate_threads": list(INTERMEDIATE_THREADS),
    }
    for name, digest in payload["timing_source_hashes"].items():
        assert git_blob_sha256(EXPECTED_TIMING_COMMIT, name) == digest
    for name, digest in payload["aggregation_source_hashes"].items():
        assert sha256(ROOT / name) == digest
    verification = subprocess.run(
        (sys.executable, str(ROOT / "experiments/fd_fv_euler/verify_phase6c.py")),
        cwd=ROOT,
        check=False,
    )
    assert verification.returncode == 0
    assert payload["admission"]["passed"]
    verify_files(payload)
    verify_workers(payload, phase6c)
    aggregates = cpu_aggregates(payload["cpu_records"])
    compare(payload["cpu_aggregates"], aggregates)
    compare(payload["shock_replication"], shock_replication(payload["shock_records"]))
    compare(payload["causal_summaries"], causal_summary(aggregates, phase6c, payload["cpu_records"]))
    assert payload["all_cpu_cells_eligible"] is True
    assert payload["all_shock_cells_eligible"] is False
    assert payload["performance_measurements_collected"] is True
    assert payload["production_sources_modified"] is False
    assert payload["phase_6e_begun"] is False
    assert payload["dveb_modified"] is False
    assert payload["publication_claim"] is False
    print("FD/FV Euler Phase 6D verification passed")


if __name__ == "__main__":
    main()
