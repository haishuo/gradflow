#!/usr/bin/env python3
"""Orchestrate the frozen Euler Phase-6D replication and causal campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from experiments.fd_fv_euler.run_phase6c import admission as phase6c_admission


PROTOCOL_COMMIT = "0a919c5"
PROTOCOL = ROOT / "docs/FD_FV_PHASE_6D_PROTOCOL.md"
PHASE6C_VERIFY = ROOT / "experiments/fd_fv_euler/verify_phase6c.py"
PHASE6C_RECORD = (
    ROOT / "experiments/fd_fv_euler/results/phase_6c_20260829/benchmark.json"
)
CPU_WORKER = Path(__file__).with_name("phase6d_cpu_worker.py")
SHOCK_WORKER = Path(__file__).with_name("phase6c_shock_worker.py")
PRIMARY_SIZES = (2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768)
INTERACTION_SIZES = (4096, 8192, 32768)
PRIMARY_THREADS = (1, 6)
INTERMEDIATE_THREADS = (2, 4)
METHODS = ("fd", "fv")
TIMING_SOURCE_COMMIT = "7952c9f"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha256(commit: str, path: Path) -> str:
    relative = path.relative_to(ROOT)
    blob = subprocess.check_output(
        ("git", "show", f"{commit}:{relative}"), cwd=ROOT
    )
    return hashlib.sha256(blob).hexdigest()


def positive_metric_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return a compiler-metric ratio, or None when the metric is unavailable."""
    if denominator <= 0 or numerator < 0:
        return None
    return numerator / denominator


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def worker_environment(cache: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    environment["TORCHINDUCTOR_CACHE_DIR"] = cache
    return environment


def run_worker(
    script: Path,
    arguments: tuple[str, ...],
    output: Path,
    *,
    measure_process: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gradflow-phase6d-") as cache:
        started = time.perf_counter_ns()
        completed = subprocess.run(
            (sys.executable, str(script), *arguments, "--output", str(output)),
            cwd=ROOT,
            env=worker_environment(cache),
            text=True,
            capture_output=True,
            check=False,
        )
        process_seconds = (time.perf_counter_ns() - started) * 1.0e-9
    if output.exists():
        record = json.loads(output.read_text())
    else:
        record = {
            "status": "failed",
            "eligible": False,
            "error_type": "WorkerProcessFailure",
            "error": "worker produced no output",
        }
    record["worker_returncode"] = completed.returncode
    record["worker_stdout"] = completed.stdout
    record["worker_stderr"] = completed.stderr
    record["command"] = [str(script), *arguments]
    if measure_process:
        record["process_launch_to_exit_seconds"] = process_seconds
        if not math.isfinite(process_seconds) or process_seconds <= 0.0:
            record["eligible"] = False
    return record


def admission() -> dict[str, Any]:
    verification = subprocess.run(
        (sys.executable, str(PHASE6C_VERIFY)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.returncode:
        raise RuntimeError("Phase 6C verification failed before timing")
    fresh = phase6c_admission()
    cpu_threads_supported = all(value <= (os.cpu_count() or 0) for value in (1, 2, 4, 6))
    passed = (
        fresh["passed"]
        and fresh["cuda_status"] == "admitted"
        and cpu_threads_supported
    )
    return {
        "phase_6c_verification_stdout": verification.stdout.strip(),
        "phase_6c_record_sha256": sha256(PHASE6C_RECORD),
        "fresh_cuda": fresh,
        "cpu_thread_counts": [1, 2, 4, 6],
        "cpu_thread_counts_supported": cpu_threads_supported,
        "passed": passed,
    }


def phase6c_shock_lookup() -> dict[tuple, dict[str, Any]]:
    payload = json.loads(PHASE6C_RECORD.read_text())
    return {
        (
            record["problem"],
            record["method"],
            record["device"],
            record["mode"],
        ): record
        for record in payload["shock_records"]
        if record["cells"] == 800
    }


def aggregate_shocks(records: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = {
        "sod": {"cpu": ("cpu", "eager"), "cuda": ("cuda", "compiled")},
        "shu_osher": {
            "cpu": ("cpu", "compiled"),
            "cuda": ("cuda", "compiled"),
        },
    }
    result: dict[str, Any] = {}
    for problem, choices in endpoints.items():
        result[problem] = {}
        for method in METHODS:
            cpu = sorted(
                (
                    record
                    for record in records
                    if record["problem"] == problem
                    and record["method"] == method
                    and (record["device"], record["mode"]) == choices["cpu"]
                ),
                key=lambda item: item["replicate"],
            )
            cuda = sorted(
                (
                    record
                    for record in records
                    if record["problem"] == problem
                    and record["method"] == method
                    and (record["device"], record["mode"]) == choices["cuda"]
                ),
                key=lambda item: item["replicate"],
            )
            ratios = [
                right["process_launch_to_exit_seconds"]
                / left["process_launch_to_exit_seconds"]
                for left, right in zip(cpu, cuda)
                if left["eligible"] and right["eligible"]
            ]
            confirmed = len(ratios) == 3 and all(x < 1.0 / 1.05 for x in ratios)
            result[problem][method] = {
                "cpu_endpoint": {"device": choices["cpu"][0], "mode": choices["cpu"][1]},
                "cuda_endpoint": {
                    "device": choices["cuda"][0],
                    "mode": choices["cuda"][1],
                },
                "cpu_seconds": [x["process_launch_to_exit_seconds"] for x in cpu],
                "cuda_seconds": [x["process_launch_to_exit_seconds"] for x in cuda],
                "cuda_over_cpu_paired_ratios": ratios,
                "aggregate_cuda_over_cpu_ratio": (
                    statistics.median(x["process_launch_to_exit_seconds"] for x in cuda)
                    / statistics.median(x["process_launch_to_exit_seconds"] for x in cpu)
                    if len(cpu) == len(cuda) == 3
                    else None
                ),
                "confirmed": confirmed,
                "decision": "confirmed_cuda_win" if confirmed else "unresolved",
            }
    return result


def expected_replicates(cells: int, threads: int) -> int:
    return 3 if cells in INTERACTION_SIZES and threads in PRIMARY_THREADS else 1


def aggregate_cpu(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configurations = sorted(
        {(x["method"], x["cells"], x["threads"]) for x in records}
    )
    result = []
    for method, cells, threads in configurations:
        selected = sorted(
            (
                x
                for x in records
                if x["method"] == method
                and x["cells"] == cells
                and x["threads"] == threads
                and x["eligible"]
            ),
            key=lambda item: item["replicate"],
        )
        expected = expected_replicates(cells, threads)
        signatures = [
            {
                "compiler_metrics": x["compiler_metrics"],
                "cpp_totals": x["generated_cpp_inventory"]["totals"],
                "cpp_hashes": [
                    item["sha256"] for item in x["generated_cpp_inventory"]["files"]
                ],
            }
            for x in selected
        ]
        structural_reproducible = bool(signatures) and all(
            item["compiler_metrics"] == signatures[0]["compiler_metrics"]
            and item["cpp_totals"] == signatures[0]["cpp_totals"]
            for item in signatures[1:]
        )
        generated_hash_reproducible = bool(signatures) and all(
            item["cpp_hashes"] == signatures[0]["cpp_hashes"]
            for item in signatures[1:]
        )
        entry: dict[str, Any] = {
            "method": method,
            "cells": cells,
            "threads": threads,
            "expected_replicates": expected,
            "eligible_replicates": len(selected),
            "structural_reproducible": structural_reproducible,
            "generated_hash_reproducible": generated_hash_reproducible,
            "compiler_metrics": signatures[0]["compiler_metrics"] if signatures else None,
            "cpp_totals": signatures[0]["cpp_totals"] if signatures else None,
            "eligible": len(selected) == expected,
        }
        for mode in ("eager", "compiled"):
            medians = [x[mode]["resident_step"]["median_seconds"] for x in selected]
            entry[f"{mode}_worker_medians_seconds"] = medians
            entry[f"{mode}_aggregate_median_seconds"] = (
                statistics.median(medians) if len(medians) == expected else None
            )
        result.append(entry)
    return result


def signature(record: dict[str, Any]) -> tuple:
    metrics = record["compiler_metrics"]
    cpp = record["cpp_totals"]
    return (
        metrics["generated_kernel_count"],
        metrics["generated_cpp_vec_kernel_count"],
        metrics["ir_nodes_pre_fusion"],
        cpp["cpp_files"],
        cpp["openmp_pragmas"],
        cpp["parallel_for_markers"],
        cpp["vectorized_markers"],
        cpp["gcc_ivdep_pragmas"],
    )


def causal_summaries(
    aggregates: list[dict[str, Any]],
    phase6c: dict[str, Any],
    cpu_records: list[dict[str, Any]],
) -> dict[str, Any]:
    lookup = {(x["method"], x["cells"], x["threads"]): x for x in aggregates}
    phase6c_steps = {
        (x["method"], x["device"], x["cells"]): x
        for x in phase6c["step_aggregates"]
        if x["mode"] == "compiled" and x["eligible"]
    }
    points: dict[str, Any] = {}
    for cells in PRIMARY_SIZES:
        item: dict[str, Any] = {}
        for threads in PRIMARY_THREADS:
            fd = lookup[("fd", cells, threads)]
            fv = lookup[("fv", cells, threads)]
            item[f"threads_{threads}"] = {
                "compiled_fv_over_fd": fv["compiled_aggregate_median_seconds"]
                / fd["compiled_aggregate_median_seconds"],
                "eager_fv_over_fd": fv["eager_aggregate_median_seconds"]
                / fd["eager_aggregate_median_seconds"],
                "compiled_over_eager": {
                    "fd": fd["compiled_aggregate_median_seconds"]
                    / fd["eager_aggregate_median_seconds"],
                    "fv": fv["compiled_aggregate_median_seconds"]
                    / fv["eager_aggregate_median_seconds"],
                },
                "fv_over_fd_ir_nodes": positive_metric_ratio(
                    fv["compiler_metrics"]["ir_nodes_pre_fusion"],
                    fd["compiler_metrics"]["ir_nodes_pre_fusion"],
                ),
                "fv_over_fd_estimated_bytes": positive_metric_ratio(
                    fv["compiler_metrics"]["num_bytes_accessed"],
                    fd["compiler_metrics"]["num_bytes_accessed"],
                ),
                "fd_signature": list(signature(fd)),
                "fv_signature": list(signature(fv)),
            }
        item["thread_interaction_factor"] = (
            item["threads_6"]["compiled_fv_over_fd"]
            / item["threads_1"]["compiled_fv_over_fd"]
        )
        item["compiled_thread_speedup_t1_over_t6"] = {
            method: lookup[(method, cells, 1)]["compiled_aggregate_median_seconds"]
            / lookup[(method, cells, 6)]["compiled_aggregate_median_seconds"]
            for method in METHODS
        }
        if cells in (8192, 32768):
            fd_cuda = phase6c_steps[("fd", "cuda", cells)][
                "aggregate_median_seconds"
            ]
            fv_cuda = phase6c_steps[("fv", "cuda", cells)][
                "aggregate_median_seconds"
            ]
            item["phase6c_cuda_fv_over_fd"] = fv_cuda / fd_cuda
        points[str(cells)] = item

    eligible_sizes = [x for x in PRIMARY_SIZES if x >= 8192]
    thread_flags = [
        points[str(cells)]["thread_interaction_factor"] >= 1.5
        for cells in eligible_sizes
    ]
    consecutive_thread = any(
        left and right for left, right in zip(thread_flags, thread_flags[1:])
    )
    replicated_thread = True
    for cells in (8192, 32768):
        ratios = []
        for replicate in range(3):
            by_key = {
                (x["method"], x["threads"], x["replicate"]): x
                for x in cpu_records
                if x["cells"] == cells and x["threads"] in PRIMARY_THREADS
            }
            ratio1 = by_key[("fv", 1, replicate)]["compiled"]["resident_step"][
                "median_seconds"
            ] / by_key[("fd", 1, replicate)]["compiled"]["resident_step"][
                "median_seconds"
            ]
            ratio6 = by_key[("fv", 6, replicate)]["compiled"]["resident_step"][
                "median_seconds"
            ] / by_key[("fd", 6, replicate)]["compiled"]["resident_step"][
                "median_seconds"
            ]
            ratios.append(ratio6 / ratio1)
        points[str(cells)]["replicated_thread_interaction_factors"] = ratios
        replicated_thread = replicated_thread and all(x >= 1.5 for x in ratios)
    thread_supported = consecutive_thread and replicated_thread

    traffic_flags = []
    for cells in eligible_sizes:
        item = points[str(cells)]["threads_6"]
        byte_ratio = item["fv_over_fd_estimated_bytes"]
        ir_ratio = item["fv_over_fd_ir_nodes"]
        traffic_flags.append(
            (byte_ratio is not None and byte_ratio >= 1.5)
            or (ir_ratio is not None and ir_ratio >= 1.5)
        )
    traffic_consecutive = any(
        left and right for left, right in zip(traffic_flags, traffic_flags[1:])
    )
    cuda_consistent = all(
        points[str(cells)]["phase6c_cuda_fv_over_fd"] < 1.25
        for cells in (8192, 32768)
    )
    traffic_supported = traffic_consecutive and cuda_consistent

    first_slow = next(
        (
            cells
            for cells in PRIMARY_SIZES
            if points[str(cells)]["threads_6"]["compiled_fv_over_fd"] > 2.0
        ),
        None,
    )
    codegen_supported = False
    if first_slow is not None and PRIMARY_SIZES.index(first_slow) > 0:
        previous = PRIMARY_SIZES[PRIMARY_SIZES.index(first_slow) - 1]
        fv_changed = (
            points[str(first_slow)]["threads_6"]["fv_signature"]
            != points[str(previous)]["threads_6"]["fv_signature"]
        )
        fd_changed = (
            points[str(first_slow)]["threads_6"]["fd_signature"]
            != points[str(previous)]["threads_6"]["fd_signature"]
        )
        codegen_supported = fv_changed and not fd_changed
    mechanisms = {
        "thread_interaction_supported": thread_supported,
        "traffic_expansion_supported": traffic_supported,
        "codegen_transition_supported": codegen_supported,
    }
    supported_count = sum(mechanisms.values())
    return {
        "points": points,
        "first_six_thread_fv_over_fd_above_2": first_slow,
        **mechanisms,
        "supported_mechanism_count": supported_count,
        "unresolved_mixture": supported_count != 1,
        "classification": (
            next(name for name, value in mechanisms.items() if value)
            if supported_count == 1
            else "unresolved_mixture"
        ),
        "estimated_bytes_metric_available": all(
            points[str(cells)][f"threads_{threads}"][
                "fv_over_fd_estimated_bytes"
            ]
            is not None
            for cells in PRIMARY_SIZES
            for threads in PRIMARY_THREADS
        ),
    }


def environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_model": "AMD Ryzen 5 7600X 6-Core Processor",
        "visible_logical_cpus": os.cpu_count(),
        "cuda_device": properties.name,
        "cuda_runtime": torch.version.cuda,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "fp64_context": "consumer_RTX_rate_restricted",
    }


def write_aggregate(
    output: Path,
    admitted: dict[str, Any],
    shock_records: list[dict[str, Any]],
    cpu_records: list[dict[str, Any]],
    *,
    timing_source_commit: str,
    aggregation_commit: str,
) -> None:
    cpu_aggregates = aggregate_cpu(cpu_records)
    phase6c = json.loads(PHASE6C_RECORD.read_text())
    timing_paths = (PROTOCOL, CPU_WORKER, SHOCK_WORKER, Path(__file__))
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_euler_phase_6d",
        "measurement_date": "2026-08-29",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": timing_source_commit,
        "source_dirty": False,
        "aggregation_commit": aggregation_commit,
        "aggregation_reused_frozen_raw_records": timing_source_commit
        != aggregation_commit,
        "aggregation_correction": (
            "Treat zero TorchInductor estimated-byte counters as unavailable; "
            "no timed worker was rerun."
            if timing_source_commit != aggregation_commit
            else None
        ),
        "admission": admitted,
        "timing_source_hashes": {
            str(path.relative_to(ROOT)): git_blob_sha256(timing_source_commit, path)
            for path in timing_paths
        },
        "aggregation_source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in timing_paths
        },
        "environment": environment(),
        "matrix": {
            "shock_cells": 800,
            "shock_replicates": 3,
            "primary_sizes": list(PRIMARY_SIZES),
            "interaction_sizes": list(INTERACTION_SIZES),
            "primary_threads": list(PRIMARY_THREADS),
            "intermediate_threads": list(INTERMEDIATE_THREADS),
        },
        "shock_records": shock_records,
        "shock_replication": aggregate_shocks(shock_records),
        "cpu_records": cpu_records,
        "cpu_aggregates": cpu_aggregates,
        "causal_summaries": causal_summaries(
            cpu_aggregates, phase6c, cpu_records
        ),
        "all_shock_cells_eligible": all(x["eligible"] for x in shock_records),
        "all_cpu_cells_eligible": all(x["eligible"] for x in cpu_records),
        "performance_measurements_collected": True,
        "production_sources_modified": False,
        "phase_6e_begun": False,
        "dveb_modified": False,
        "publication_claim": False,
    }
    aggregate_path = output / "benchmark.json"
    aggregate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    files = [aggregate_path, *sorted((output / "raw").glob("*.json"))]
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output)}\n" for path in files
        )
    )
    print(f"wrote Phase 6D aggregate to {aggregate_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aggregate-existing", action="store_true")
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if output.exists() and not arguments.aggregate_existing:
        raise FileExistsError(f"refusing existing output directory: {output}")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6D requires a clean committed source tree")
    admitted = admission()
    if not admitted["passed"]:
        raise RuntimeError("Phase 6D admission failed before timing")
    aggregation_commit = git("rev-parse", "HEAD")
    if arguments.aggregate_existing:
        raw = output / "raw"
        if not raw.is_dir() or (output / "benchmark.json").exists():
            raise RuntimeError(
                "existing aggregation requires raw/ and no benchmark.json"
            )
        shock_paths = sorted(raw.glob("shock_*.json"))
        cpu_paths = sorted(raw.glob("cpu_*.json"))
        if len(shock_paths) != 24 or len(cpu_paths) != 68:
            raise RuntimeError(
                f"incomplete raw campaign: {len(shock_paths)} shock, "
                f"{len(cpu_paths)} CPU records"
            )
        shock_records = [json.loads(path.read_text()) for path in shock_paths]
        cpu_records = [json.loads(path.read_text()) for path in cpu_paths]
        write_aggregate(
            output,
            admitted,
            shock_records,
            cpu_records,
            timing_source_commit=git("rev-parse", TIMING_SOURCE_COMMIT),
            aggregation_commit=aggregation_commit,
        )
        return
    source_commit = aggregation_commit
    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()

    reference = phase6c_shock_lookup()
    shock_records = []
    endpoints = {
        "sod": (("cpu", "eager"), ("cuda", "compiled")),
        "shu_osher": (("cpu", "compiled"), ("cuda", "compiled")),
    }
    for replicate in range(3):
        for problem, choices in endpoints.items():
            for device, mode in choices:
                for method in METHODS:
                    path = raw / (
                        f"shock_{problem}_{device}_{method}_{mode}_n800_"
                        f"r{replicate}.json"
                    )
                    record = run_worker(
                        SHOCK_WORKER,
                        (
                            "--problem",
                            problem,
                            "--method",
                            method,
                            "--device",
                            device,
                            "--mode",
                            mode,
                            "--cells",
                            "800",
                        ),
                        path,
                        measure_process=True,
                    )
                    record["replicate"] = replicate
                    expected_hash = reference[(problem, method, device, mode)][
                        "terminal_sha256"
                    ]
                    record["phase6c_terminal_sha256"] = expected_hash
                    record["terminal_hash_matches_phase6c"] = (
                        record.get("terminal_sha256") == expected_hash
                    )
                    record["eligible"] = bool(
                        record.get("eligible")
                        and record["terminal_hash_matches_phase6c"]
                    )
                    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                    shock_records.append(record)
                    print(
                        f"shock r{replicate} {problem} {device} {method} {mode}: "
                        f"{record.get('status')}",
                        flush=True,
                    )

    cpu_records = []
    baseline = [
        (cells, threads)
        for cells in PRIMARY_SIZES
        for threads in PRIMARY_THREADS
    ] + [
        (cells, threads)
        for cells in INTERACTION_SIZES
        for threads in INTERMEDIATE_THREADS
    ]
    for cells, threads in baseline:
        for method in METHODS:
            path = raw / f"cpu_{method}_n{cells}_t{threads}_r0.json"
            record = run_worker(
                CPU_WORKER,
                (
                    "--method",
                    method,
                    "--cells",
                    str(cells),
                    "--threads",
                    str(threads),
                    "--replicate",
                    "0",
                ),
                path,
                measure_process=False,
            )
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            cpu_records.append(record)
            print(
                f"cpu baseline {method} N={cells} threads={threads}: "
                f"{record.get('status')}",
                flush=True,
            )
    for cells in INTERACTION_SIZES:
        for threads in PRIMARY_THREADS:
            for replicate in (1, 2):
                for method in METHODS:
                    path = raw / (
                        f"cpu_{method}_n{cells}_t{threads}_r{replicate}.json"
                    )
                    record = run_worker(
                        CPU_WORKER,
                        (
                            "--method",
                            method,
                            "--cells",
                            str(cells),
                            "--threads",
                            str(threads),
                            "--replicate",
                            str(replicate),
                        ),
                        path,
                        measure_process=False,
                    )
                    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                    cpu_records.append(record)
                    print(
                        f"cpu replicate r{replicate} {method} N={cells} "
                        f"threads={threads}: {record.get('status')}",
                        flush=True,
                    )

    write_aggregate(
        output,
        admitted,
        shock_records,
        cpu_records,
        timing_source_commit=source_commit,
        aggregation_commit=aggregation_commit,
    )


if __name__ == "__main__":
    main()
