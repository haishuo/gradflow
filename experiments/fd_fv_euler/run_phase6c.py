#!/usr/bin/env python3
"""Orchestrate the frozen isolated-worker Euler Phase-6C campaign."""

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

from experiments.fd_fv_euler.phase6c_problem import (
    fixed_step_function,
    smooth_initial,
    stage_function,
)
from experiments.infrastructure.device_admission import classify_device_admission


PROTOCOL_COMMIT = "86a379f"
PROTOCOL = ROOT / "docs/FD_FV_PHASE_6C_PROTOCOL.md"
PHASE6B_RECORD = (
    ROOT
    / "experiments/fd_fv_euler/results/phase_6b_20260828/qualification.json"
)
PHASE6B_VERIFY = ROOT / "experiments/fd_fv_euler/verify_phase6b.py"
PROBLEM = Path(__file__).with_name("phase6c_problem.py")
WORKER = Path(__file__).with_name("phase6c_worker.py")
COLD_WORKER = Path(__file__).with_name("phase6c_cold_worker.py")
SHOCK_WORKER = Path(__file__).with_name("phase6c_shock_worker.py")
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
    with tempfile.TemporaryDirectory(prefix="gradflow-phase6c-") as cache:
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
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def admission() -> dict[str, Any]:
    verification = subprocess.run(
        (sys.executable, str(PHASE6B_VERIFY)),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if verification.returncode:
        raise RuntimeError("Phase 6B verification failed before timing")
    if not torch.cuda.is_available():
        status = classify_device_admission(
            process_visible=False,
            host_inventory="present",
            admission="not_run",
        )
        raise RuntimeError(f"CUDA admission requires visibility: {status}")
    cases: dict[str, Any] = {}
    for method in METHODS:
        cpu_state = smooth_initial(method, 37)
        cuda_state = cpu_state.cuda()
        stages = stage_function(method, 37, "periodic")
        dt_cpu = cpu_state.new_tensor(2.0e-4)
        dt_cuda = cuda_state.new_tensor(2.0e-4)
        cpu = stages(cpu_state, dt_cpu)[-1]
        cuda_eager = stages(cuda_state, dt_cuda)[-1]
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(stages)(cuda_state, dt_cuda)
        torch._dynamo.reset()
        compiled = torch.compile(stages, fullgraph=True, dynamic=False)
        cuda_compiled = compiled(cuda_state, dt_cuda)[-1]
        torch.cuda.synchronize()
        cpu_cuda = float(torch.max(torch.abs(cuda_eager.cpu() - cpu)))
        compiled_eager = float(
            torch.max(torch.abs(cuda_compiled - cuda_eager))
        )
        cases[method] = {
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "cpu_cuda_maximum_absolute_difference": cpu_cuda,
            "compiled_eager_maximum_absolute_difference": compiled_eager,
            "finite": bool(torch.isfinite(cuda_compiled).all()),
            "resident": cuda_compiled.device == cuda_state.device,
            "passed": explanation.graph_count == 1
            and explanation.graph_break_count == 0
            and cpu_cuda <= 5.0e-11
            and compiled_eager <= 5.0e-11
            and bool(torch.isfinite(cuda_compiled).all())
            and cuda_compiled.device == cuda_state.device,
        }
    passed = all(case["passed"] for case in cases.values())
    return {
        "phase_6b_verification_stdout": verification.stdout.strip(),
        "phase_6b_record_sha256": sha256(PHASE6B_RECORD),
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


def classification(ratio: float, band: float = 0.05) -> str:
    if ratio > 1.0 + band:
        return "fd_faster"
    if ratio < 1.0 / (1.0 + band):
        return "fv_faster"
    return "unresolved_within_band"


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
                    eligible_records = [
                        record for record in selected if record.get("eligible")
                    ]
                    medians = [
                        record[mode]["resident_complete_solve"]["median_seconds"]
                        for record in eligible_records
                    ]
                    transfer = [
                        record[mode]["prepared_transfer_complete_solve"][
                            "median_seconds"
                        ]
                        for record in eligible_records
                        if device == "cuda"
                    ]
                    errors = [
                        record["accuracy"][mode]["l2_error"]
                        for record in eligible_records
                    ]
                    memory = [
                        record["memory"]["peak_process_rss_bytes"]
                        for record in eligible_records
                    ]
                    cuda_memory = [
                        record[mode]["cuda_memory"]["peak_allocated_bytes"]
                        for record in eligible_records
                        if device == "cuda"
                    ]
                    eligible = len(medians) == 3 and max(errors, default=0.0) - min(
                        errors, default=0.0
                    ) <= 2.0e-13
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
                            "transfer_worker_median_seconds": transfer,
                            "transfer_aggregate_median_seconds": (
                                statistics.median(transfer)
                                if eligible and device == "cuda"
                                else None
                            ),
                            "l2_error": statistics.median(errors) if errors else None,
                            "peak_process_rss_bytes": max(memory) if memory else None,
                            "peak_cuda_allocated_bytes": (
                                max(cuda_memory) if cuda_memory else None
                            ),
                            "eligible": eligible,
                        }
                    )
    return aggregates


def target_selections(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
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
    selections: dict[str, Any] = {}
    for boundary, (device, timing_field) in boundaries.items():
        by_target: dict[str, Any] = {}
        for target in ERROR_TARGETS:
            methods: dict[str, Any] = {}
            for method in METHODS:
                candidates = [
                    item
                    for item in aggregates
                    if item["method"] == method
                    and item["device"] == device
                    and item["eligible"]
                    and item["l2_error"] <= target
                    and item[timing_field] is not None
                ]
                if candidates:
                    selected = min(candidates, key=lambda item: item[timing_field])
                    methods[method] = {
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
                    methods[method] = {"status": "not_reached"}
            if all(methods[name]["status"] == "reached" for name in METHODS):
                ratio = methods["fv"]["median_seconds"] / methods["fd"][
                    "median_seconds"
                ]
                methods["fv_over_fd_time_ratio"] = ratio
                methods["classification"] = classification(ratio)
            by_target[str(target)] = methods
        selections[boundary] = by_target
    return selections


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
            cpu_record = baseline.get((method, "cpu", cells))
            cuda_record = baseline.get((method, "cuda", cells))
            if not cpu_record or not cuda_record:
                continue
            cpu = cpu_record["modes"]["compiled"]["resident_step"][
                "median_seconds"
            ]
            cuda = cuda_record["modes"]["compiled"]["resident_step"][
                "median_seconds"
            ]
            if cuda / cpu < 1.0 / 1.05:
                winning = cells
                break
        if winning is None:
            result[method] = list(STEP_SIZES[-2:])
        else:
            index = STEP_SIZES.index(winning)
            result[method] = sorted(
                {STEP_SIZES[max(0, index - 1)], winning}
            )
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
                    eligible = len(medians) == expected
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
                                statistics.median(medians) if eligible else None
                            ),
                            "transfer_worker_median_seconds": transfer,
                            "transfer_aggregate_median_seconds": (
                                statistics.median(transfer)
                                if eligible and device == "cuda"
                                else None
                            ),
                            "eligible": eligible,
                        }
                    )
    lookup = {
        (x["method"], x["device"], x["cells"], x["mode"]): x
        for x in aggregates
    }
    crossovers: dict[str, Any] = {}
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
    return aggregates, crossovers


def equal_grid_steps(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (x["method"], x["device"], x["cells"], x["mode"]): x
        for x in aggregates
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
        if x.get("eligible")
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
        (
            x["problem"],
            x["method"],
            x["device"],
            x["mode"],
            x["cells"],
        ): x
        for x in records
        if x.get("eligible")
    }
    result = []
    for problem in ("sod", "shu_osher"):
        for device in DEVICES:
            for mode in MODES:
                for cells in SHOCK_SIZES:
                    fd = lookup.get((problem, "fd", device, mode, cells))
                    fv = lookup.get((problem, "fv", device, mode, cells))
                    if fd and fv:
                        ratio = fv["process_launch_to_exit_seconds"] / fd[
                            "process_launch_to_exit_seconds"
                        ]
                        result.append(
                            {
                                "problem": problem,
                                "device": device,
                                "mode": mode,
                                "cells": cells,
                                "fv_over_fd_ratio": ratio,
                                "classification": "descriptive_single_observation",
                            }
                        )
    return result


def environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=driver_version,uuid",
            "--format=csv,noheader",
        ),
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
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6C requires a clean committed source tree")
    source_commit = git("rev-parse", "HEAD")
    admitted = admission()
    if not admitted["passed"] or admitted["cuda_status"] != "admitted":
        raise RuntimeError("fresh Phase 6C CUDA admission failed; no timing run")

    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    complete_records = []
    for replicate in range(3):
        for cells in COMPLETE_SIZES:
            for device in DEVICES:
                for method in METHODS:
                    path = raw / f"complete_{device}_{method}_n{cells}_r{replicate}.json"
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
    for cells in STEP_SIZES:
        for device in DEVICES:
            for method in METHODS:
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
                    path = raw / f"step_{device}_{method}_n{cells}_r{replicate}.json"
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
                        f"step replicate r{replicate} {device} {method} N={cells}: "
                        f"{record.get('status')}",
                        flush=True,
                    )

    cold_records = []
    for cells in COLD_SIZES:
        for device in DEVICES:
            for mode in MODES:
                for method in METHODS:
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

    shock_records = []
    for problem in ("sod", "shu_osher"):
        for cells in SHOCK_SIZES:
            for device in DEVICES:
                for mode in MODES:
                    for method in METHODS:
                        path = raw / (
                            f"shock_{problem}_{device}_{method}_{mode}_n{cells}.json"
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
                                str(cells),
                            ),
                            path,
                            measure_process=True,
                        )
                        shock_records.append(record)
                        print(
                            f"shock {problem} {device} {method} {mode} "
                            f"N={cells}: {record.get('status')}",
                            flush=True,
                        )

    complete_aggregates = aggregate_complete(complete_records)
    step_aggregates, crossovers = aggregate_steps(step_records, replication)
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_euler_phase_6c",
        "measurement_date": "2026-08-29",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": source_commit,
        "source_dirty": False,
        "admission": admitted,
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                PROTOCOL,
                PROBLEM,
                WORKER,
                COLD_WORKER,
                SHOCK_WORKER,
                Path(__file__),
            )
        },
        "environment": environment(),
        "matrix": {
            "complete_sizes": list(COMPLETE_SIZES),
            "step_sizes": list(STEP_SIZES),
            "cold_sizes": list(COLD_SIZES),
            "error_targets": list(ERROR_TARGETS),
            "shock_sizes": list(SHOCK_SIZES),
        },
        "complete_records": complete_records,
        "complete_aggregates": complete_aggregates,
        "target_selections": target_selections(complete_aggregates),
        "step_records": step_records,
        "step_replication_sizes": replication,
        "step_aggregates": step_aggregates,
        "step_device_crossovers": crossovers,
        "equal_grid_step_comparisons": equal_grid_steps(step_aggregates),
        "cold_records": cold_records,
        "cold_comparisons": cold_comparisons(cold_records),
        "shock_records": shock_records,
        "shock_comparisons": shock_comparisons(shock_records),
        "prepared_aot": {"status": "not_implemented"},
        "all_complete_cells_eligible": all(
            record["eligible"] for record in complete_records
        ),
        "all_step_cells_eligible": all(
            record["eligible"] for record in step_records
        ),
        "all_cold_cells_eligible": all(
            record["eligible"] for record in cold_records
        ),
        "all_shock_cells_eligible": all(
            record["eligible"] for record in shock_records
        ),
        "performance_measurements_collected": True,
        "dveb_modified": False,
        "phase_6d_begun": False,
        "publication_claim": False,
    }
    aggregate_path = output / "benchmark.json"
    aggregate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    files = [aggregate_path, *sorted(raw.glob("*.json"))]
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{sha256(path)}  {path.relative_to(output)}\n" for path in files
        )
    )
    print(f"wrote Phase 6C aggregate to {aggregate_path}", flush=True)


if __name__ == "__main__":
    main()
