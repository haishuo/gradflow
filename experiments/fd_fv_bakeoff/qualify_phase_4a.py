#!/usr/bin/env python3
"""Run the timing-free multidimensional FD/FV Phase-4A admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

from problem import (
    FINAL_TIME,
    METHODS,
    METHOD_IDS,
    SIZES,
    conservation,
    errors,
    projected_state,
    solve,
    step_function,
    timestep,
    velocities,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "experiments/fd_fv_bakeoff/results/phase_4a_20260827"
PROTOCOL = ROOT / "docs/FD_FV_PHASE_4_PROTOCOL.md"
PROTOCOL_COMMIT = "6dbd4d1"
PHASE3R_RECORD = (
    ROOT
    / "experiments/fd_fv_qualification/results/phase_3r_20260827/resolution.json"
)
PHASE3R_VERIFY = (
    ROOT / "experiments/fd_fv_qualification/verify_phase_3r.py"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def rates(values: list[float], sizes: tuple[int, ...]) -> list[float]:
    return [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            values, values[1:], sizes, sizes[1:]
        )
    ]


def phase3r_verification() -> dict[str, Any]:
    completed = subprocess.run(
        (sys.executable, str(PHASE3R_VERIFY)),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "record_sha256": sha256(PHASE3R_RECORD),
        "passed": completed.returncode == 0,
    }


def convergence() -> dict[str, Any]:
    records = {}
    for method in METHODS:
        dimensions = {}
        for dimension, sizes in SIZES.items():
            runs = []
            l1_values = []
            l2_values = []
            for cells in sizes:
                initial = projected_state(method, dimension, cells)
                expected = projected_state(
                    method, dimension, cells, time=FINAL_TIME
                )
                step, steps = step_function(method, dimension, cells)
                final = solve(initial, step, steps)
                l1, l2 = errors(final, expected)
                mass_change, mass_bound, mass_passed = conservation(
                    initial, final, dimension, cells
                )
                l1_values.append(l1)
                l2_values.append(l2)
                runs.append(
                    {
                        "cells_per_axis": cells,
                        "logical_cells": cells**dimension,
                        "steps": steps,
                        "dt": timestep(dimension, cells)[0],
                        "l1_error": l1,
                        "l2_error": l2,
                        "finite": math.isfinite(l1)
                        and math.isfinite(l2)
                        and bool(torch.isfinite(final).all()),
                        "mass_change": mass_change,
                        "mass_bound": mass_bound,
                        "conservation_passed": mass_passed,
                    }
                )
            l1_rates = rates(l1_values, sizes)
            l2_rates = rates(l2_values, sizes)
            monotone_l1 = all(
                fine < coarse for coarse, fine in zip(l1_values, l1_values[1:])
            )
            monotone_l2 = all(
                fine < coarse for coarse, fine in zip(l2_values, l2_values[1:])
            )
            passed = (
                monotone_l1
                and monotone_l2
                and max(l2_rates) >= 4.0
                and all(run["finite"] for run in runs)
                and all(run["conservation_passed"] for run in runs)
            )
            dimensions[str(dimension)] = {
                "sizes": sizes,
                "runs": runs,
                "l1_rates": l1_rates,
                "l2_rates": l2_rates,
                "monotone_l1": monotone_l1,
                "monotone_l2": monotone_l2,
                "passed": passed,
            }
        records[method] = {
            "formulation_id": METHOD_IDS[method],
            "dimensions": dimensions,
            "passed": all(item["passed"] for item in dimensions.values()),
        }
    return {
        "methods": records,
        "passed": all(item["passed"] for item in records.values()),
    }


def compiler_admission() -> dict[str, Any]:
    records = {}
    for method in METHODS:
        dimensions = {}
        for dimension, sizes in SIZES.items():
            cells = sizes[-1]
            state = projected_state(method, dimension, cells)
            step, _ = step_function(method, dimension, cells)
            eager = step(state)
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(step)(state)
            torch._dynamo.reset()
            compiled = torch.compile(step, fullgraph=True, dynamic=False)
            actual = compiled(state)
            difference = float(torch.max(torch.abs(actual - eager)))
            passed = (
                explanation.graph_count == 1
                and explanation.graph_break_count == 0
                and difference <= 2.0e-11
                and actual.shape == state.shape
                and actual.dtype == state.dtype
                and actual.device == state.device
            )
            dimensions[str(dimension)] = {
                "cells_per_axis": cells,
                "graph_count": explanation.graph_count,
                "graph_break_count": explanation.graph_break_count,
                "break_reasons": [
                    str(reason) for reason in explanation.break_reasons
                ],
                "maximum_absolute_difference": difference,
                "shape_preserved": actual.shape == state.shape,
                "dtype_preserved": actual.dtype == state.dtype,
                "device_preserved": actual.device == state.device,
                "passed": passed,
            }
        records[method] = {
            "dimensions": dimensions,
            "passed": all(item["passed"] for item in dimensions.values()),
        }
    return {
        "methods": records,
        "passed": all(item["passed"] for item in records.values()),
    }


def cuda_admission() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "untested_unavailable", "available": False}
    records = {}
    for method in METHODS:
        dimensions = {}
        for dimension, sizes in SIZES.items():
            cells = sizes[-1]
            cpu_state = projected_state(method, dimension, cells)
            gpu_state = cpu_state.cuda()
            step, _ = step_function(method, dimension, cells)
            cpu_expected = step(cpu_state)
            gpu_eager = step(gpu_state)
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(step)(gpu_state)
            torch._dynamo.reset()
            gpu_compiled = torch.compile(
                step, fullgraph=True, dynamic=False
            )(gpu_state)
            eager_difference = float(
                torch.max(torch.abs(gpu_eager.cpu() - cpu_expected))
            )
            compiled_difference = float(
                torch.max(torch.abs(gpu_compiled - gpu_eager))
            )
            passed = (
                eager_difference <= 2.0e-11
                and compiled_difference <= 2.0e-11
                and explanation.graph_count == 1
                and explanation.graph_break_count == 0
                and gpu_eager.device.type == "cuda"
                and gpu_compiled.device.type == "cuda"
            )
            dimensions[str(dimension)] = {
                "cells_per_axis": cells,
                "cpu_eager_maximum_absolute_difference": eager_difference,
                "compiled_eager_maximum_absolute_difference": compiled_difference,
                "graph_count": explanation.graph_count,
                "graph_break_count": explanation.graph_break_count,
                "resident": gpu_eager.device.type == "cuda"
                and gpu_compiled.device.type == "cuda",
                "passed": passed,
            }
        records[method] = {
            "dimensions": dimensions,
            "passed": all(item["passed"] for item in dimensions.values()),
        }
    passed = all(item["passed"] for item in records.values())
    return {
        "status": "passed" if passed else "failed",
        "available": True,
        "device": torch.cuda.get_device_name(),
        "methods": records,
    }


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        ),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def record() -> dict[str, Any]:
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    prior = phase3r_verification()
    convergence_result = convergence()
    compiler = compiler_admission()
    cuda = cuda_admission()
    gates = {
        "phase_3r_verified": prior["passed"],
        "multidimensional_convergence": convergence_result["passed"],
        "cpu_compiler": compiler["passed"],
        "cuda_if_available": cuda["status"] in {
            "passed",
            "untested_unavailable",
        },
    }
    return {
        "schema_version": 1,
        "phase": "fd_fv_phase_4a",
        "qualification_date": "2026-08-27",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "problem": {
            "equation": "constant-coefficient scalar linear advection",
            "dimensions": sorted(SIZES),
            "sizes": SIZES,
            "velocities": {
                str(dimension): velocities(dimension) for dimension in SIZES
            },
            "final_time": FINAL_TIME,
            "dtype": "float64",
            "boundary": "unique_periodic",
            "time_integrator": "SSP-RK3",
        },
        "source_hashes": {
            "docs/FD_FV_PHASE_4_PROTOCOL.md": sha256(PROTOCOL),
            "experiments/fd_fv_bakeoff/problem.py": sha256(
                Path(__file__).with_name("problem.py")
            ),
            "experiments/fd_fv_bakeoff/qualify_phase_4a.py": sha256(
                Path(__file__)
            ),
            "src/gradflow/fv_weno5.py": sha256(
                ROOT / "src/gradflow/fv_weno5.py"
            ),
            "src/gradflow/weno_js.py": sha256(ROOT / "src/gradflow/weno_js.py"),
        },
        "phase_3r": prior,
        "environment": environment(),
        "convergence": convergence_result,
        "compiler": compiler,
        "cuda": cuda,
        "mps": {
            "status": (
                "not_executed"
                if environment()["mps_available"]
                else "untested_unavailable"
            )
        },
        "gate_decisions": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "passed": all(gates.values()) and not source_dirty,
        "performance_measurements_collected": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    record_path = output / "qualification.json"
    sums_path = output / "SHA256SUMS"
    if record_path.exists() or sums_path.exists():
        raise FileExistsError(f"refusing to overwrite Phase-4A record in {output}")
    output.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record(), indent=2, sort_keys=True) + "\n")
    sums_path.write_text(f"{sha256(record_path)}  qualification.json\n")
    print(f"wrote FD/FV Phase-4A qualification to {record_path}")


if __name__ == "__main__":
    main()
