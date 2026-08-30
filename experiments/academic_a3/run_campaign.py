#!/usr/bin/env python3
"""Run the frozen Academic A3 inverse and performance campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from problem import (  # noqa: E402
    TRUE_SPEED,
    autograd_inverse,
    evaluate,
    golden_section_search,
    make_problem,
)


FINITE_DIFFERENCE_STEPS = tuple(10.0**-power for power in range(1, 9))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derivative_gate() -> dict[str, Any]:
    problem = make_problem(128)
    speed = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    value = problem.objective(speed)
    derivative = torch.autograd.grad(value, speed)[0]
    autograd_value = float(derivative.detach())
    records = []
    for step in FINITE_DIFFERENCE_STEPS:
        plus = evaluate(problem.objective, 0.9 + step)
        minus = evaluate(problem.objective, 0.9 - step)
        centered = (plus - minus) / (2.0 * step)
        absolute = abs(centered - autograd_value)
        relative = absolute / max(abs(autograd_value), 1.0e-12)
        records.append(
            {
                "step": step,
                "objective_plus": plus,
                "objective_minus": minus,
                "centered_derivative": centered,
                "absolute_error": absolute,
                "relative_error": relative,
                "finite": all(
                    math.isfinite(item)
                    for item in (plus, minus, centered, absolute, relative)
                ),
            }
        )
    registered_window = [
        record for record in records if record["step"] in (1.0e-3, 1.0e-4, 1.0e-5)
    ]
    passed = all(record["finite"] for record in records) and any(
        record["relative_error"] <= 2.0e-6 or record["absolute_error"] <= 1.0e-10
        for record in registered_window
    )
    return {
        "n": 128,
        "order": 11,
        "evaluation_speed": 0.9,
        "objective": float(value.detach()),
        "autograd_derivative": autograd_value,
        "records": records,
        "registered_window_passed": passed,
    }


def inverse_gate() -> dict[str, Any]:
    problem = make_problem(128)
    initial_objective = evaluate(problem.objective, 0.8)
    autograd = autograd_inverse(problem.objective)
    golden = golden_section_search(problem.objective)
    speed_difference = abs(autograd["speed"] - golden["speed"])
    truth_error = abs(autograd["speed"] - TRUE_SPEED)
    reduction = autograd["objective"] / initial_objective
    scan = []
    for index in range(201):
        speed = 0.5 + index / 200.0
        scan.append({"speed": speed, "objective": evaluate(problem.objective, speed)})
    sampled_minimum = min(scan, key=lambda record: record["objective"])
    return {
        "n": 128,
        "order": 11,
        "initial_speed": 0.8,
        "initial_objective": initial_objective,
        "autograd": autograd,
        "golden_section": golden,
        "autograd_golden_speed_difference": speed_difference,
        "autograd_truth_error": truth_error,
        "terminal_over_initial_objective": reduction,
        "objective_scan": scan,
        "sampled_scan_minimum": sampled_minimum,
        "passed": (
            speed_difference <= 2.0e-6 and truth_error <= 2.0e-3 and reduction <= 1.0e-4
        ),
    }


def resolution_study() -> list[dict[str, Any]]:
    records = []
    for n in (64, 128, 256):
        problem = make_problem(n)
        result = golden_section_search(problem.objective)
        records.append(
            {
                "n": n,
                "steps": problem.steps,
                "dx": problem.dx,
                "dt": problem.dt,
                "speed": result["speed"],
                "truth_error": abs(result["speed"] - TRUE_SPEED),
                "objective": result["objective"],
                "objective_evaluations": result["objective_evaluations"],
                "final_interval": result["final_interval"],
            }
        )
    return records


def run_worker(device: str, raw_directory: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(HERE)))
    with tempfile.TemporaryDirectory(prefix=f"gradflow_a3_{device}_") as cache:
        environment["TORCHINDUCTOR_CACHE_DIR"] = cache
        completed = subprocess.run(
            [sys.executable, str(HERE / "benchmark_worker.py"), "--device", device],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    raw_directory.mkdir(parents=True, exist_ok=True)
    (raw_directory / f"benchmark_{device}.stdout").write_text(completed.stdout)
    (raw_directory / f"benchmark_{device}.stderr").write_text(completed.stderr)
    payload = None
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        pass
    return {
        "returncode": completed.returncode,
        "record": payload,
        "stdout_path": f"raw/benchmark_{device}.stdout",
        "stderr_path": f"raw/benchmark_{device}.stderr",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing existing output: {arguments.output}")
    evidence = arguments.output.parent
    document: dict[str, Any] = {
        "schema": "gradflow-academic-a3-campaign-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "protocol_commit": "faa83d7",
        "canonical_source_changed": False,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "derivative_gate": derivative_gate(),
    }
    evidence.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")
    document["inverse_gate"] = inverse_gate()
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")
    document["resolution_study"] = resolution_study()
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")
    raw = evidence / "raw"
    document["benchmarks"] = {"cpu": run_worker("cpu", raw)}
    if torch.cuda.is_available():
        document["benchmarks"]["cuda"] = run_worker("cuda", raw)
    else:
        document["benchmarks"]["cuda"] = {
            "returncode": None,
            "record": {
                "schema": "gradflow-academic-a3-benchmark-v1",
                "status": "unavailable",
                "device": "cuda",
            },
        }
    source_paths = (
        ROOT / "docs/ACADEMIC_A3_PROTOCOL.md",
        ROOT / "src/gradflow/weno_js.py",
        HERE / "problem.py",
        HERE / "benchmark_worker.py",
        Path(__file__).resolve(),
    )
    document["source_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in source_paths
    }
    document["complete"] = True
    document["completed_utc"] = datetime.now(timezone.utc).isoformat()
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
