#!/usr/bin/env python3
"""Orchestrate the frozen isolated-worker nonlinear Phase-5C campaign."""

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

from gradflow import (
    burgers_fd_weno5_rhs,
    burgers_fv_weno5_rhs,
    ssp_rk3_step,
)
from experiments.fd_fv_nonlinear.burgers_oracle import LF_ALPHA
from experiments.fd_fv_nonlinear.performance_problem import state
from experiments.infrastructure.device_admission import classify_device_admission


PROTOCOL_COMMIT = "1bc340c"
PROTOCOL = ROOT / "docs/FD_FV_PHASE_5C_PROTOCOL.md"
PHASE5B_RECORD = (
    ROOT
    / "experiments/fd_fv_nonlinear/results/phase_5b_20260828"
    / "qualification.json"
)
PHASE5B_VERIFY = ROOT / "experiments/fd_fv_nonlinear/verify_phase_5b.py"
WORKER = Path(__file__).with_name("phase5c_worker.py")
COLD_WORKER = Path(__file__).with_name("phase5c_cold_worker.py")
PROBLEM = Path(__file__).with_name("performance_problem.py")
COMPLETE_SIZES = (24, 36, 54, 81, 162)
STEP_SIZES = (32, 128, 512, 2048, 8192, 32768, 131072, 524288)
COLD_SIZES = (24, 81, 162)
ERROR_TARGETS = (2.0e-5, 3.0e-6, 5.0e-7, 1.0e-7, 5.0e-8)
METHODS = ("fd", "fv")
DEVICES = ("cpu", "cuda")
MODES = ("eager", "compiled")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    measure_process: bool = False,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gradflow-phase5c-") as cache:
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
            "error_type": "WorkerProcessFailure",
            "error": "worker produced no output",
        }
    record["worker_returncode"] = completed.returncode
    record["worker_stdout"] = completed.stdout
    record["worker_stderr"] = completed.stderr
    record["command"] = [str(script), *arguments]
    if measure_process:
        record["process_launch_to_exit_seconds"] = process_seconds
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def admission() -> dict[str, Any]:
    verification = subprocess.run(
        (sys.executable, str(PHASE5B_VERIFY)),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if verification.returncode:
        raise RuntimeError("Phase 5B verification failed before timing")
    if not torch.cuda.is_available():
        status = classify_device_admission(
            process_visible=False,
            host_inventory="present",
            admission="not_run",
        )
        raise RuntimeError(f"CUDA admission requires device visibility: {status}")
    cases: dict[str, Any] = {}
    cpu = state("fd", 37)
    cuda = cpu.cuda()
    for method, rhs in (
        ("fd", burgers_fd_weno5_rhs),
        ("fv", burgers_fv_weno5_rhs),
    ):
        cpu_state = state(method, 37)
        cuda_state = cpu_state.cuda()

        def rhs_call(values: torch.Tensor) -> torch.Tensor:
            return rhs(values, 1.0 / 37.0, LF_ALPHA)

        def step_call(values: torch.Tensor) -> torch.Tensor:
            return ssp_rk3_step(values, 1.0e-3, rhs_call)

        for name, call in (("rhs", rhs_call), ("step", step_call)):
            cpu_expected = call(cpu_state)
            cuda_eager = call(cuda_state)
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(call)(cuda_state)
            torch._dynamo.reset()
            compiled = torch.compile(call, fullgraph=True, dynamic=False)
            cuda_compiled = compiled(cuda_state)
            torch.cuda.synchronize()
            cpu_cuda = float(torch.max(torch.abs(cuda_eager.cpu() - cpu_expected)))
            compiled_eager = float(torch.max(torch.abs(cuda_compiled - cuda_eager)))
            key = f"{method}_{name}"
            cases[key] = {
                "graph_count": explanation.graph_count,
                "graph_break_count": explanation.graph_break_count,
                "cpu_cuda_maximum_absolute_difference": cpu_cuda,
                "compiled_eager_maximum_absolute_difference": compiled_eager,
                "finite": bool(torch.isfinite(cuda_compiled).all()),
                "resident": cuda_compiled.device == cuda_state.device,
                "passed": explanation.graph_count == 1
                and explanation.graph_break_count == 0
                and cpu_cuda <= 2.0e-11
                and compiled_eager <= 2.0e-11
                and bool(torch.isfinite(cuda_compiled).all())
                and cuda_compiled.device == cuda_state.device,
            }
    passed = all(case["passed"] for case in cases.values())
    return {
        "phase_5b_verification_stdout": verification.stdout.strip(),
        "phase_5b_record_sha256": sha256(PHASE5B_RECORD),
        "cuda_host_inventory": "present",
        "cuda_process_visible": True,
        "cuda_status": classify_device_admission(
            process_visible=True,
            host_inventory="present",
            admission="passed" if passed else "failed",
        ),
        "cases": cases,
        "passed": passed,
    }


def aggregate_complete(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = []
    for method in METHODS:
        for device in DEVICES:
            for cells in COMPLETE_SIZES:
                selected = [
                    record
                    for record in records
                    if record.get("method") == method
                    and record.get("device") == device
                    and record.get("cells") == cells
                ]
                for mode in MODES:
                    medians = [
                        record[mode]["resident_complete_solve"]["median_seconds"]
                        for record in selected
                        if record.get("eligible")
                    ]
                    transfer_medians = [
                        record[mode]["prepared_transfer_complete_solve"][
                            "median_seconds"
                        ]
                        for record in selected
                        if record.get("eligible") and device == "cuda"
                    ]
                    accuracy = selected[0]["accuracy"][mode] if selected else {}
                    memory_values = [
                        record["memory"]["peak_process_rss_bytes"]
                        for record in selected
                        if record.get("eligible")
                    ]
                    cuda_memory = [
                        record[mode]["cuda_memory"]["peak_allocated_bytes"]
                        for record in selected
                        if record.get("eligible") and device == "cuda"
                    ]
                    eligible = len(medians) == 3
                    aggregates.append(
                        {
                            "method": method,
                            "device": device,
                            "cells": cells,
                            "mode": mode,
                            "replicates": len(selected),
                            "eligible_replicates": len(medians),
                            "worker_median_seconds": medians,
                            "aggregate_median_seconds": (
                                statistics.median(medians) if eligible else None
                            ),
                            "transfer_worker_median_seconds": transfer_medians,
                            "transfer_aggregate_median_seconds": (
                                statistics.median(transfer_medians)
                                if len(transfer_medians) == 3
                                else None
                            ),
                            "l1_error": accuracy.get("l1_error"),
                            "l2_error": accuracy.get("l2_error"),
                            "peak_process_rss_bytes": (
                                max(memory_values) if memory_values else None
                            ),
                            "peak_cuda_allocated_bytes": (
                                max(cuda_memory) if cuda_memory else None
                            ),
                            "eligible": eligible,
                        }
                    )
    return aggregates


def target_selections(
    aggregates: list[dict[str, Any]],
) -> dict[str, Any]:
    boundaries = {
        "cpu_warm": ("cpu", "aggregate_median_seconds"),
        "cuda_resident_warm": ("cuda", "aggregate_median_seconds"),
        "cuda_prepared_transfer": (
            "cuda",
            "transfer_aggregate_median_seconds",
        ),
    }
    selections: dict[str, Any] = {}
    for boundary, (device, timing_field) in boundaries.items():
        boundary_record: dict[str, Any] = {}
        for target in ERROR_TARGETS:
            target_record: dict[str, Any] = {}
            for method in METHODS:
                candidates = [
                    record
                    for record in aggregates
                    if record["method"] == method
                    and record["device"] == device
                    and record["eligible"]
                    and record["l2_error"] <= target
                    and record[timing_field] is not None
                ]
                if candidates:
                    selected = min(candidates, key=lambda item: item[timing_field])
                    target_record[method] = {
                        "status": "reached",
                        "cells": selected["cells"],
                        "mode": selected["mode"],
                        "l2_error": selected["l2_error"],
                        "median_seconds": selected[timing_field],
                        "peak_process_rss_bytes": selected[
                            "peak_process_rss_bytes"
                        ],
                        "peak_cuda_allocated_bytes": selected[
                            "peak_cuda_allocated_bytes"
                        ],
                    }
                else:
                    target_record[method] = {"status": "not_reached"}
            if all(target_record[m]["status"] == "reached" for m in METHODS):
                ratio = (
                    target_record["fv"]["median_seconds"]
                    / target_record["fd"]["median_seconds"]
                )
                target_record["fv_over_fd_time_ratio"] = ratio
                target_record["classification"] = classification(ratio)
            boundary_record[str(target)] = target_record
        selections[boundary] = boundary_record
    return selections


def classification(ratio: float) -> str:
    if ratio > 1.05:
        return "fd_faster"
    if ratio < 1.0 / 1.05:
        return "fv_faster"
    return "unresolved_within_5_percent"


def baseline_step_map(records: list[dict[str, Any]]) -> dict[tuple, dict]:
    return {
        (record["method"], record["device"], record["cells"]): record
        for record in records
        if record.get("replicate") == 0 and record.get("eligible")
    }


def replication_sizes(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    baseline = baseline_step_map(records)
    result: dict[str, list[int]] = {}
    for method in METHODS:
        winning = None
        for cells in STEP_SIZES:
            cpu = baseline[(method, "cpu", cells)]["modes"]["compiled"][
                "resident_step"
            ]["median_seconds"]
            cuda = baseline[(method, "cuda", cells)]["modes"]["compiled"][
                "resident_step"
            ]["median_seconds"]
            if cuda / cpu < 1.0 / 1.05:
                winning = cells
                break
        if winning is None:
            result[method] = list(STEP_SIZES[-2:])
        else:
            index = STEP_SIZES.index(winning)
            previous = STEP_SIZES[max(0, index - 1)]
            result[method] = sorted({previous, winning})
    return result


def aggregate_steps(
    records: list[dict[str, Any]], replication: dict[str, list[int]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregates = []
    for method in METHODS:
        for device in DEVICES:
            for cells in STEP_SIZES:
                selected = [
                    record
                    for record in records
                    if record.get("method") == method
                    and record.get("device") == device
                    and record.get("cells") == cells
                    and record.get("eligible")
                ]
                expected = 3 if cells in replication[method] else 1
                for mode in MODES:
                    medians = [
                        record["modes"][mode]["resident_step"]["median_seconds"]
                        for record in selected
                    ]
                    transfer = [
                        record["modes"][mode]["transfer_inclusive_step"][
                            "median_seconds"
                        ]
                        for record in selected
                        if device == "cuda"
                    ]
                    aggregates.append(
                        {
                            "method": method,
                            "device": device,
                            "cells": cells,
                            "mode": mode,
                            "expected_replicates": expected,
                            "eligible_replicates": len(medians),
                            "worker_median_seconds": medians,
                            "aggregate_median_seconds": (
                                statistics.median(medians)
                                if len(medians) == expected
                                else None
                            ),
                            "transfer_worker_median_seconds": transfer,
                            "transfer_aggregate_median_seconds": (
                                statistics.median(transfer)
                                if len(transfer) == expected
                                else None
                            ),
                            "eligible": len(medians) == expected,
                        }
                    )
    by_key = {
        (record["method"], record["device"], record["cells"], record["mode"]): record
        for record in aggregates
    }
    crossovers: dict[str, Any] = {}
    for method in METHODS:
        baseline_winner = None
        for cells in STEP_SIZES:
            cpu = by_key[(method, "cpu", cells, "compiled")]
            cuda = by_key[(method, "cuda", cells, "compiled")]
            if (
                cpu["aggregate_median_seconds"] is not None
                and cuda["aggregate_median_seconds"]
                / cpu["aggregate_median_seconds"]
                < 1.0 / 1.05
            ):
                baseline_winner = cells
                break
        ratios = []
        confirmed = False
        if baseline_winner is not None and baseline_winner in replication[method]:
            cpu_values = by_key[(method, "cpu", baseline_winner, "compiled")][
                "worker_median_seconds"
            ]
            cuda_values = by_key[(method, "cuda", baseline_winner, "compiled")][
                "worker_median_seconds"
            ]
            ratios = [cuda / cpu for cpu, cuda in zip(cpu_values, cuda_values)]
            confirmed = len(ratios) == 3 and all(
                ratio < 1.0 / 1.05 for ratio in ratios
            )
        crossovers[method] = {
            "baseline_winning_cells": baseline_winner,
            "replicated_sizes": replication[method],
            "cuda_over_cpu_worker_median_ratios": ratios,
            "confirmed": confirmed,
            "decision": (
                f"confirmed_at_n{baseline_winner}"
                if confirmed
                else "unresolved"
            ),
        }
    return aggregates, crossovers


def equal_grid_step_comparisons(
    aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (record["method"], record["device"], record["cells"], record["mode"]): record
        for record in aggregates
    }
    comparisons = []
    for device in DEVICES:
        for cells in STEP_SIZES:
            for mode in MODES:
                fd = by_key[("fd", device, cells, mode)]
                fv = by_key[("fv", device, cells, mode)]
                if not fd["eligible"] or not fv["eligible"]:
                    continue
                ratio = (
                    fv["aggregate_median_seconds"]
                    / fd["aggregate_median_seconds"]
                )
                comparisons.append(
                    {
                        "device": device,
                        "cells": cells,
                        "mode": mode,
                        "fv_over_fd_ratio": ratio,
                        "classification": classification(ratio),
                    }
                )
    return comparisons


def environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,uuid",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cpu_model": "AMD Ryzen 5 7600X 6-Core Processor",
        "cpu_intraop_threads": 6,
        "cpu_interop_threads": 1,
        "visible_logical_cpus": os.cpu_count(),
        "cuda": {
            "device": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "runtime": torch.version.cuda,
            "driver_uuid_query": query.stdout.strip(),
            "driver_query_returncode": query.returncode,
            "fp64_context": "consumer_RTX_rate_restricted",
        },
        "mps_status": "host_confirmed_absent",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    if source_dirty:
        raise RuntimeError("Phase 5C requires a clean committed source tree")
    admitted = admission()
    if not admitted["passed"] or admitted["cuda_status"] != "admitted":
        raise RuntimeError("fresh Phase 5C CUDA admission failed; no timing run")

    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    complete_records = []
    for replicate in range(3):
        for method in METHODS:
            for device in DEVICES:
                for cells in COMPLETE_SIZES:
                    path = raw / (
                        f"complete_{device}_{method}_n{cells}_r{replicate}.json"
                    )
                    record = run_worker(
                        WORKER,
                        (
                            "--kind",
                            "complete",
                            "--method",
                            method,
                            "--device",
                            device,
                            "--cells",
                            str(cells),
                            "--replicate",
                            str(replicate),
                        ),
                        path,
                    )
                    complete_records.append(record)
                    print(
                        f"complete r{replicate} {device} {method} N={cells}: "
                        f"{record.get('status')}",
                        flush=True,
                    )

    step_records = []
    for method in METHODS:
        for device in DEVICES:
            for cells in STEP_SIZES:
                path = raw / f"step_{device}_{method}_n{cells}_r0.json"
                record = run_worker(
                    WORKER,
                    (
                        "--kind",
                        "step",
                        "--method",
                        method,
                        "--device",
                        device,
                        "--cells",
                        str(cells),
                        "--replicate",
                        "0",
                    ),
                    path,
                )
                step_records.append(record)
                print(
                    f"step baseline {device} {method} N={cells}: "
                    f"{record.get('status')}",
                    flush=True,
                )

    replication = replication_sizes(step_records)
    for method in METHODS:
        for cells in replication[method]:
            for device in DEVICES:
                for replicate in (1, 2):
                    path = raw / (
                        f"step_{device}_{method}_n{cells}_r{replicate}.json"
                    )
                    record = run_worker(
                        WORKER,
                        (
                            "--kind",
                            "step",
                            "--method",
                            method,
                            "--device",
                            device,
                            "--cells",
                            str(cells),
                            "--replicate",
                            str(replicate),
                        ),
                        path,
                    )
                    step_records.append(record)
                    print(
                        f"step replicate r{replicate} {device} {method} "
                        f"N={cells}: {record.get('status')}",
                        flush=True,
                    )

    cold_records = []
    for method in METHODS:
        for device in DEVICES:
            for mode in MODES:
                for cells in COLD_SIZES:
                    path = raw / f"cold_{device}_{method}_{mode}_n{cells}.json"
                    record = run_worker(
                        COLD_WORKER,
                        (
                            "--method",
                            method,
                            "--device",
                            device,
                            "--mode",
                            mode,
                            "--cells",
                            str(cells),
                        ),
                        path,
                        measure_process=True,
                    )
                    cold_records.append(record)
                    print(
                        f"cold {device} {method} {mode} N={cells}: "
                        f"{record.get('status')}",
                        flush=True,
                    )

    complete_aggregates = aggregate_complete(complete_records)
    step_aggregates, crossovers = aggregate_steps(step_records, replication)
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_nonlinear_phase_5c",
        "measurement_date": "2026-08-28",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "admission": admitted,
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PROTOCOL, PROBLEM, WORKER, COLD_WORKER, Path(__file__))
        },
        "environment": environment(),
        "matrix": {
            "complete_sizes": list(COMPLETE_SIZES),
            "step_sizes": list(STEP_SIZES),
            "cold_sizes": list(COLD_SIZES),
            "error_targets": list(ERROR_TARGETS),
        },
        "complete_records": complete_records,
        "complete_aggregates": complete_aggregates,
        "target_selections": target_selections(complete_aggregates),
        "step_records": step_records,
        "step_replication_sizes": replication,
        "step_aggregates": step_aggregates,
        "step_device_crossovers": crossovers,
        "equal_grid_step_comparisons": equal_grid_step_comparisons(
            step_aggregates
        ),
        "cold_records": cold_records,
        "prepared_aot": {"status": "not_implemented"},
        "all_complete_cells_eligible": all(
            record["eligible"] for record in complete_records
        ),
        "all_step_cells_eligible": all(record["eligible"] for record in step_records),
        "all_cold_cells_eligible": all(record["eligible"] for record in cold_records),
        "performance_measurements_collected": True,
    }
    aggregate_path = output / "benchmark.json"
    aggregate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    files = [aggregate_path, *sorted(raw.glob("*.json"))]
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output)}\n" for path in files
        )
    )
    print(f"wrote Phase 5C aggregate to {aggregate_path}", flush=True)


if __name__ == "__main__":
    main()
