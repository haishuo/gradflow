#!/usr/bin/env python3
"""Orchestrate the admitted isolated CPU FD/FV Phase-4B campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("benchmark_worker.py")
COLD_WORKER = Path(__file__).with_name("cold_worker.py")
PROBLEM = Path(__file__).with_name("problem.py")
PROTOCOL = ROOT / "docs/FD_FV_PHASE_4_PROTOCOL.md"
PHASE4A_DIR = ROOT / "experiments/fd_fv_bakeoff/results/phase_4a_20260827"
PHASE4A_RECORD = PHASE4A_DIR / "qualification.json"
PHASE4A_VERIFY = Path(__file__).with_name("verify_phase_4a.py")
METHODS = ("fd", "fv")
SIZES = {
    1: (24, 36, 54, 81),
    2: (12, 18, 27, 40),
    3: (8, 12, 18, 27),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def run_worker(
    script: Path,
    arguments: tuple[str, ...],
    output: Path,
) -> tuple[dict[str, Any], float]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    with tempfile.TemporaryDirectory(prefix="gradflow-fd-fv-phase4-") as cache:
        environment["TORCHINDUCTOR_CACHE_DIR"] = cache
        started = time.perf_counter_ns()
        completed = subprocess.run(
            (sys.executable, str(script), *arguments, "--output", str(output)),
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
                    "error_type": "WorkerProcessFailure",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    result = json.loads(output.read_text())
    result["worker_returncode"] = completed.returncode
    result["worker_stdout"] = completed.stdout
    result["worker_stderr"] = completed.stderr
    return result, process_seconds


def classification(ratio: float) -> str:
    if ratio > 1.05:
        return "fd_faster"
    if ratio < 1.0 / 1.05:
        return "fv_faster"
    return "unresolved_within_5_percent"


def derive_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for dimension, sizes in SIZES.items():
        for cells in sizes:
            selected = {
                record["method"]: record
                for record in records
                if record.get("status") == "completed"
                and record["dimension"] == dimension
                and record["cells_per_axis"] == cells
            }
            if set(selected) != set(METHODS):
                continue
            fd = selected["fd"]
            fv = selected["fv"]
            modes = {}
            for mode in ("eager", "compiled"):
                solve_ratio = (
                    fv[mode]["complete_solve"]["median_seconds"]
                    / fd[mode]["complete_solve"]["median_seconds"]
                )
                step_ratio = (
                    fv[mode]["ssp_rk3_step"]["median_seconds"]
                    / fd[mode]["ssp_rk3_step"]["median_seconds"]
                )
                modes[mode] = {
                    "fv_over_fd_complete_solve_ratio": solve_ratio,
                    "complete_solve_classification": classification(solve_ratio),
                    "fv_over_fd_step_ratio": step_ratio,
                    "step_classification": classification(step_ratio),
                }
            best = {}
            for method, record in selected.items():
                candidates = {
                    mode: record[mode]["complete_solve"]["median_seconds"]
                    for mode in ("eager", "compiled")
                }
                selected_mode = min(candidates, key=candidates.get)
                best[method] = {
                    "mode": selected_mode,
                    "median_seconds": candidates[selected_mode],
                    "l2_error": record["accuracy"][f"{selected_mode}_l2_error"],
                    "peak_process_rss_bytes": record["memory"][
                        "peak_process_rss_bytes"
                    ],
                }
            ratio = best["fv"]["median_seconds"] / best["fd"]["median_seconds"]
            comparisons.append(
                {
                    "dimension": dimension,
                    "cells_per_axis": cells,
                    "logical_cells": cells**dimension,
                    "eligible": fd["eligible"] and fv["eligible"],
                    "matched_modes": modes,
                    "best_practical": {
                        "fd": best["fd"],
                        "fv": best["fv"],
                        "fv_over_fd_complete_solve_ratio": ratio,
                        "classification": classification(ratio),
                    },
                }
            )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")

    verification = subprocess.run(
        (sys.executable, str(PHASE4A_VERIFY)),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if verification.returncode:
        raise RuntimeError(
            "Phase 4A verification failed; refusing performance measurement"
        )
    phase4a = json.loads(PHASE4A_RECORD.read_text())
    if not phase4a["passed"] or phase4a["performance_measurements_collected"]:
        raise RuntimeError("Phase 4A did not provide a clean timing-free admission")

    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    if source_dirty:
        raise RuntimeError("refusing Phase 4B measurement from a dirty source tree")
    output.mkdir(parents=True)
    raw = output / "raw"
    raw.mkdir()
    records = []
    for dimension, sizes in SIZES.items():
        for cells in sizes:
            for method in METHODS:
                path = raw / f"cpu_{method}_{dimension}d_n{cells}.json"
                result, process_seconds = run_worker(
                    WORKER,
                    (
                        "--method",
                        method,
                        "--dimension",
                        str(dimension),
                        "--cells",
                        str(cells),
                    ),
                    path,
                )
                result["total_worker_process_seconds"] = process_seconds
                path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
                records.append(result)
                print(
                    f"warm {method} {dimension}D N={cells}: "
                    f"{result.get('status')}",
                    flush=True,
                )

    cold_records = []
    for dimension, sizes in SIZES.items():
        cells = sizes[-1]
        for method in METHODS:
            path = raw / f"cold_cpu_{method}_{dimension}d_n{cells}.json"
            result, process_seconds = run_worker(
                COLD_WORKER,
                (
                    "--method",
                    method,
                    "--dimension",
                    str(dimension),
                    "--cells",
                    str(cells),
                ),
                path,
            )
            result["process_launch_to_exit_seconds"] = process_seconds
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            cold_records.append(result)
            print(
                f"cold {method} {dimension}D N={cells}: "
                f"{result.get('status')}",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "phase": "fd_fv_phase_4b",
        "measurement_date": "2026-08-27",
        "protocol_commit": "6dbd4d1",
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "phase_4a": {
            "source_commit": phase4a["source_commit"],
            "record_sha256": sha256(PHASE4A_RECORD),
            "verification_stdout": verification.stdout.strip(),
            "passed": True,
            "performance_measurements_collected": False,
        },
        "source_hashes": {
            "docs/FD_FV_PHASE_4_PROTOCOL.md": sha256(PROTOCOL),
            "experiments/fd_fv_bakeoff/problem.py": sha256(PROBLEM),
            "experiments/fd_fv_bakeoff/benchmark_worker.py": sha256(WORKER),
            "experiments/fd_fv_bakeoff/cold_worker.py": sha256(COLD_WORKER),
            "experiments/fd_fv_bakeoff/benchmark_phase_4b.py": sha256(
                Path(__file__)
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "visible_logical_cpus": os.cpu_count(),
            "cpu_model": "AMD Ryzen 5 7600X 6-Core Processor",
            "cpu_intraop_threads": 6,
            "cpu_interop_threads": 1,
            "cuda": phase4a["cuda"],
            "mps": phase4a["mps"],
        },
        "matrix": {str(key): list(value) for key, value in SIZES.items()},
        "warm_records": records,
        "cold_records": cold_records,
        "comparisons": derive_comparisons(records),
        "prepared_aot": {"status": "not_implemented"},
        "cuda_measurements": {
            "status": (
                "not_collected_unavailable"
                if phase4a["cuda"]["status"] == "untested_unavailable"
                else "not_collected"
            )
        },
        "all_warm_cells_eligible": all(
            record.get("eligible", False) for record in records
        ),
        "all_cold_cells_eligible": all(
            record.get("eligible", False) for record in cold_records
        ),
        "performance_measurements_collected": True,
    }
    result_path = output / "benchmark.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = [f"{sha256(result_path)}  benchmark.json"]
    manifest.extend(
        f"{sha256(path)}  raw/{path.name}" for path in sorted(raw.glob("*.json"))
    )
    (output / "SHA256SUMS").write_text("\n".join(manifest) + "\n")
    print(f"wrote FD/FV Phase-4B benchmark to {result_path}")


if __name__ == "__main__":
    main()
