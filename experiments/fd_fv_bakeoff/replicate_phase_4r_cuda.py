#!/usr/bin/env python3
"""Run the frozen Phase-4R CUDA admission and resident replication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

import torch

from problem import METHODS, SIZES as ADMISSION_SIZES, projected_state, step_function


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("cuda_replication_worker.py")
PROBLEM = Path(__file__).with_name("problem.py")
PROTOCOL = ROOT / "docs/FD_FV_PHASE_4_REPLICATION_PROTOCOL.md"
CPU_RECORD = (
    ROOT / "experiments/fd_fv_bakeoff/results/phase_4r_20260827/replication.json"
)
VERIFY_CPU = Path(__file__).with_name("verify_phase_4r.py")
CUDA_SIZES = (18, 27, 40, 64)
REPLICATES = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()


def classification(ratio: float) -> str:
    if ratio > 1.05:
        return "fd_faster"
    if ratio < 1.0 / 1.05:
        return "fv_faster"
    return "unresolved_within_5_percent"


def device_record() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=False,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver or None,
        "device": torch.cuda.get_device_name(0),
        "device_uuid": str(getattr(properties, "uuid", "unknown")),
        "device_total_memory_bytes": properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "multiprocessor_count": properties.multi_processor_count,
    }


def fresh_admission() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "failed_unavailable", "available": False, "cases": []}
    cases = []
    for method in METHODS:
        for dimension, sizes in ADMISSION_SIZES.items():
            cells = sizes[-1]
            cpu_state = projected_state(method, dimension, cells)
            gpu_state = cpu_state.cuda()
            step, _ = step_function(method, dimension, cells)
            cpu_eager = step(cpu_state)
            gpu_eager = step(gpu_state)
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(step)(gpu_state)
            torch._dynamo.reset()
            gpu_compiled = torch.compile(step, fullgraph=True, dynamic=False)(gpu_state)
            eager_difference = float(torch.max(torch.abs(gpu_eager.cpu() - cpu_eager)))
            compiled_difference = float(torch.max(torch.abs(gpu_compiled - gpu_eager)))
            finite = bool(torch.isfinite(gpu_compiled).all())
            passed = (
                eager_difference <= 2.0e-11
                and compiled_difference <= 2.0e-11
                and explanation.graph_count == 1
                and explanation.graph_break_count == 0
                and finite
                and gpu_eager.device.type == "cuda"
                and gpu_compiled.device.type == "cuda"
            )
            cases.append(
                {
                    "method": method,
                    "dimension": dimension,
                    "cells_per_axis": cells,
                    "cpu_eager_gpu_eager_maximum_absolute_difference": eager_difference,
                    "compiled_eager_maximum_absolute_difference": compiled_difference,
                    "graph_count": explanation.graph_count,
                    "graph_break_count": explanation.graph_break_count,
                    "finite": finite,
                    "resident": gpu_eager.device.type == "cuda"
                    and gpu_compiled.device.type == "cuda",
                    "passed": passed,
                }
            )
    passed = all(case["passed"] for case in cases)
    return {
        "status": "passed" if passed else "failed",
        "available": True,
        "environment": device_record(),
        "cases": cases,
        "passed": passed,
        "performance_measurements_collected": False,
    }


def run_worker(method: str, cells: int, replicate: int, output: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    with tempfile.TemporaryDirectory(
        prefix=f"gradflow-phase4r-cuda-{method}-n{cells}-r{replicate}-"
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


def summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for cells in CUDA_SIZES:
        selected = {
            method: sorted(
                (
                    record
                    for record in records
                    if record["method"] == method and record["cells_per_axis"] == cells
                ),
                key=lambda record: record["replicate"],
            )
            for method in METHODS
        }
        modes = {}
        for mode in ("eager", "compiled"):
            worker_medians = {
                method: [
                    record["timing"][mode]["median_seconds"]
                    for record in selected[method]
                ]
                for method in METHODS
            }
            process_medians = {
                method: statistics.median(values)
                for method, values in worker_medians.items()
            }
            ratio = process_medians["fv"] / process_medians["fd"]
            modes[mode] = {
                "fd_worker_medians_seconds": worker_medians["fd"],
                "fv_worker_medians_seconds": worker_medians["fv"],
                "fd_process_median_seconds": process_medians["fd"],
                "fv_process_median_seconds": process_medians["fv"],
                "paired_fv_over_fd_ratios": [
                    fv / fd
                    for fd, fv in zip(worker_medians["fd"], worker_medians["fv"])
                ],
                "fv_over_fd_process_median_ratio": ratio,
                "classification": classification(ratio),
            }
        result.append(
            {
                "cells_per_axis": cells,
                "logical_cells": cells**3,
                "replicates": REPLICATES,
                "eligible": all(
                    record.get("eligible", False)
                    for rows in selected.values()
                    for record in rows
                ),
                "modes": modes,
            }
        )
    return result


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
        raise RuntimeError("refusing CUDA replication from a dirty source tree")
    verified = subprocess.run(
        (sys.executable, str(VERIFY_CPU)),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if verified.returncode != 0:
        raise RuntimeError("Phase-4R CPU record did not verify")
    admission = fresh_admission()
    output.mkdir(parents=True)
    admission_path = output / "admission.json"
    admission_path.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")
    if admission["status"] != "passed":
        raise RuntimeError("fresh CUDA admission failed; no timing collected")
    raw = output / "raw"
    raw.mkdir()
    records = []
    for cells in CUDA_SIZES:
        for replicate in range(REPLICATES):
            order = METHODS if replicate % 2 == 0 else tuple(reversed(METHODS))
            for method in order:
                path = raw / f"cuda_{method}_3d_n{cells}_r{replicate}.json"
                record = run_worker(method, cells, replicate, path)
                records.append(record)
                print(
                    f"{method} CUDA N={cells} replicate={replicate}: "
                    f"{record.get('status')}",
                    flush=True,
                )
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_phase_4r_cuda_supplement",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_commit": "037e980",
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "command": " ".join(sys.argv),
        "cpu_replication_sha256": sha256(CPU_RECORD),
        "cpu_verification": {
            "returncode": verified.returncode,
            "stdout": verified.stdout.strip(),
            "stderr": verified.stderr.strip(),
            "passed": verified.returncode == 0,
        },
        "source_hashes": {
            "docs/FD_FV_PHASE_4_REPLICATION_PROTOCOL.md": sha256(PROTOCOL),
            "experiments/fd_fv_bakeoff/problem.py": sha256(PROBLEM),
            "experiments/fd_fv_bakeoff/cuda_replication_worker.py": sha256(WORKER),
            "experiments/fd_fv_bakeoff/replicate_phase_4r_cuda.py": sha256(
                Path(__file__)
            ),
            "src/gradflow/fv_weno5.py": sha256(ROOT / "src/gradflow/fv_weno5.py"),
            "src/gradflow/weno_js.py": sha256(ROOT / "src/gradflow/weno_js.py"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cuda": admission["environment"],
        },
        "fresh_admission": admission,
        "sizes": CUDA_SIZES,
        "replicates": REPLICATES,
        "raw_records": records,
        "size_summaries": summaries(records),
        "all_cells_eligible": all(record.get("eligible", False) for record in records),
        "performance_measurements_collected": True,
        "timing_scope": (
            "device_resident_cuda_events_excludes_transfers_and_compilation"
        ),
    }
    result_path = output / "replication_cuda.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest = [
        f"{sha256(admission_path)}  admission.json",
        f"{sha256(result_path)}  replication_cuda.json",
    ]
    manifest.extend(
        f"{sha256(path)}  raw/{path.name}" for path in sorted(raw.glob("*.json"))
    )
    (output / "SHA256SUMS").write_text("\n".join(manifest) + "\n")
    print(f"wrote FD/FV Phase-4R CUDA supplement to {result_path}")


if __name__ == "__main__":
    main()
