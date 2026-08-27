#!/usr/bin/env python3
"""Orchestrate frozen Phase-4R CPU replication and CUDA availability audit."""

from __future__ import annotations

import argparse
import hashlib
import json
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
WORKER = Path(__file__).with_name("replication_worker.py")
PROBLEM = Path(__file__).with_name("problem.py")
PHASE4A_RECORD = (
    ROOT / "experiments/fd_fv_bakeoff/results/phase_4a_20260827/qualification.json"
)
PHASE4B_RECORD = (
    ROOT / "experiments/fd_fv_bakeoff/results/phase_4b_20260827/benchmark.json"
)
VERIFY4A = Path(__file__).with_name("verify_phase_4a.py")
VERIFY4B = Path(__file__).with_name("verify_phase_4b.py")
PROTOCOL = ROOT / "docs/FD_FV_PHASE_4_REPLICATION_PROTOCOL.md"
SIZES = (18, 21, 24, 27, 30, 33, 36, 40, 48)
METHODS = ("fd", "fv")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def classification(ratio: float) -> str:
    if ratio > 1.05:
        return "fd_faster"
    if ratio < 1.0 / 1.05:
        return "fv_faster"
    return "unresolved_within_5_percent"


def repetitions(cells: int) -> int:
    return 3 if cells == 27 else 2


def run_worker(
    method: str,
    cells: int,
    replicate: int,
    output: Path,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    with tempfile.TemporaryDirectory(
        prefix=f"gradflow-phase4r-{method}-n{cells}-r{replicate}-"
    ) as cache:
        environment["TORCHINDUCTOR_CACHE_DIR"] = cache
        started = time.perf_counter_ns()
        completed = subprocess.run(
            (
                sys.executable,
                str(WORKER),
                "--method",
                method,
                "--cells",
                str(cells),
                "--replicate",
                str(replicate),
                "--output",
                str(output),
            ),
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        process_seconds = (time.perf_counter_ns() - started) * 1.0e-9
    if not output.exists():
        output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "method": method,
                    "cells_per_axis": cells,
                    "replicate": replicate,
                    "error_type": "WorkerProcessFailure",
                    "returncode": completed.returncode,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    record = json.loads(output.read_text())
    record["worker_returncode"] = completed.returncode
    record["worker_stdout"] = completed.stdout
    record["worker_stderr"] = completed.stderr
    record["total_worker_process_seconds"] = process_seconds
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def size_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for cells in SIZES:
        selected = {
            method: sorted(
                (
                    record
                    for record in records
                    if record["method"] == method
                    and record["cells_per_axis"] == cells
                ),
                key=lambda record: record["replicate"],
            )
            for method in METHODS
        }
        medians = {
            method: [
                record["timing"]["compiled"]["median_seconds"]
                for record in selected[method]
            ]
            for method in METHODS
        }
        process_medians = {
            method: statistics.median(values) for method, values in medians.items()
        }
        paired = [
            fv["timing"]["compiled"]["median_seconds"]
            / fd["timing"]["compiled"]["median_seconds"]
            for fd, fv in zip(selected["fd"], selected["fv"])
        ]
        ratio = process_medians["fv"] / process_medians["fd"]
        metrics = {}
        for method in METHODS:
            metrics[method] = {
                name: [record["compiler_metrics"].get(name) for record in selected[method]]
                for name in (
                    "generated_kernel_count",
                    "generated_cpp_vec_kernel_count",
                    "ir_nodes_pre_fusion",
                    "num_bytes_accessed",
                    "num_loop_reordering",
                    "num_auto_chunking",
                )
            }
            metrics[method]["cpp_file_count"] = [
                record["cache_evidence"]["cpp_file_count"]
                for record in selected[method]
            ]
            metrics[method]["cpp_total_bytes"] = [
                record["cache_evidence"]["cpp_total_bytes"]
                for record in selected[method]
            ]
        summaries.append(
            {
                "cells_per_axis": cells,
                "logical_cells": cells**3,
                "replicates": repetitions(cells),
                "eligible": all(
                    record["eligible"]
                    for method_records in selected.values()
                    for record in method_records
                ),
                "fd_worker_medians_seconds": medians["fd"],
                "fv_worker_medians_seconds": medians["fv"],
                "fd_process_median_seconds": process_medians["fd"],
                "fv_process_median_seconds": process_medians["fv"],
                "paired_fv_over_fd_ratios": paired,
                "fv_over_fd_process_median_ratio": ratio,
                "classification": classification(ratio),
                "compiler_evidence": metrics,
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")

    verifications = {}
    for name, script in (("phase_4a", VERIFY4A), ("phase_4b", VERIFY4B)):
        completed = subprocess.run(
            (sys.executable, str(script)),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        verifications[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "passed": completed.returncode == 0,
        }
    if not all(item["passed"] for item in verifications.values()):
        raise RuntimeError("Phase 4 records did not verify; refusing replication")
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    if source_dirty:
        raise RuntimeError("refusing Phase 4R from a dirty source tree")

    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    records = []
    for cells in SIZES:
        for replicate in range(repetitions(cells)):
            method_order = METHODS if replicate % 2 == 0 else tuple(reversed(METHODS))
            for method in method_order:
                path = raw / f"cpu_{method}_3d_n{cells}_r{replicate}.json"
                record = run_worker(method, cells, replicate, path)
                records.append(record)
                print(
                    f"{method} N={cells} replicate={replicate}: "
                    f"{record.get('status')}",
                    flush=True,
                )

    summaries = size_summaries(records)
    n27 = next(item for item in summaries if item["cells_per_axis"] == 27)
    below_eight = [
        item["cells_per_axis"]
        for item in summaries
        if item["fv_over_fd_process_median_ratio"] < 0.8
    ]
    strong_replication = (
        sum(ratio < 0.5 for ratio in n27["paired_fv_over_fd_ratios"]) >= 2
        and n27["fv_over_fd_process_median_ratio"] < 0.5
    )
    phase4a = json.loads(PHASE4A_RECORD.read_text())
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_phase_4r",
        "replication_date": "2026-08-27",
        "protocol_commit": "037e980",
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "prior_verification": verifications,
        "prior_hashes": {
            "phase_4a": sha256(PHASE4A_RECORD),
            "phase_4b": sha256(PHASE4B_RECORD),
        },
        "source_hashes": {
            "docs/FD_FV_PHASE_4_REPLICATION_PROTOCOL.md": sha256(PROTOCOL),
            "experiments/fd_fv_bakeoff/problem.py": sha256(PROBLEM),
            "experiments/fd_fv_bakeoff/replication_worker.py": sha256(WORKER),
            "experiments/fd_fv_bakeoff/replicate_phase_4r.py": sha256(
                Path(__file__)
            ),
            "src/gradflow/fv_weno5.py": sha256(
                ROOT / "src/gradflow/fv_weno5.py"
            ),
            "src/gradflow/weno_js.py": sha256(ROOT / "src/gradflow/weno_js.py"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "visible_logical_cpus": os.cpu_count(),
            "cpu_model": "AMD Ryzen 5 7600X 6-Core Processor",
            "primary_intraop_threads": 6,
            "interop_threads": 1,
            "cuda_admission": phase4a["cuda"],
        },
        "sizes": SIZES,
        "raw_records": records,
        "size_summaries": summaries,
        "n27_strong_replication": strong_replication,
        "transition_below_0_8": {
            "sampled_sizes": below_eight,
            "first_sampled_size": below_eight[0] if below_eight else None,
            "last_sampled_size": below_eight[-1] if below_eight else None,
        },
        "all_cpu_cells_eligible": all(record.get("eligible", False) for record in records),
        "cuda_replication": {
            "status": (
                "untested_unavailable"
                if phase4a["cuda"]["status"] == "untested_unavailable"
                else "requires_fresh_device_admission"
            ),
            "measurements_collected": False,
        },
        "performance_measurements_collected": True,
    }
    result_path = output / "replication.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = [f"{sha256(result_path)}  replication.json"]
    manifest.extend(
        f"{sha256(path)}  raw/{path.name}" for path in sorted(raw.glob("*.json"))
    )
    (output / "SHA256SUMS").write_text("\n".join(manifest) + "\n")
    print(f"wrote FD/FV Phase-4R replication to {result_path}")


if __name__ == "__main__":
    main()
