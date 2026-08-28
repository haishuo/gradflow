#!/usr/bin/env python3
"""Execute CUDA correctness gates deferred by earlier sandbox visibility."""

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
from typing import Any, Callable

import torch

from gradflow import (
    euler1d_cfl_timestep,
    euler1d_rhs,
    fv_weno5_face_states,
    fv_weno5_rhs,
    ssp_rk3_step,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/DEFERRED_CUDA_GATES_PROTOCOL.md"
FV_SOURCE = ROOT / "src/gradflow/fv_weno5.py"
WENO_SOURCE = ROOT / "src/gradflow/weno_js.py"
EULER_SOURCE = ROOT / "src/gradflow/euler1d.py"
PREDECESSORS = {
    "fd_fv_phase_3": (
        ROOT / "experiments/fd_fv_qualification/verify_phase_3.py",
        ROOT
        / "experiments/fd_fv_qualification/results/phase_3_20260827/qualification.json",
    ),
    "fd_fv_phase_3r": (
        ROOT / "experiments/fd_fv_qualification/verify_phase_3r.py",
        ROOT
        / "experiments/fd_fv_qualification/results/phase_3r_20260827/resolution.json",
    ),
    "euler_boundary_shock_phase_b": (
        ROOT / "experiments/euler_boundary_shock/verify_phase_b.py",
        ROOT
        / "experiments/euler_boundary_shock/results/phase_b_20260827/qualification.json",
    ),
}
ORDERS = (5, 7, 9, 11, 13, 15)
REPRESENTATIVE_ORDERS = (5, 11, 15)
BOUNDARIES = ("periodic", "transmissive")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()


def verify_predecessors() -> dict[str, Any]:
    records = {}
    for name, (script, record) in PREDECESSORS.items():
        completed = subprocess.run(
            (sys.executable, str(script)),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        records[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "record": str(record.relative_to(ROOT)),
            "record_sha256": sha256(record),
            "passed": completed.returncode == 0,
        }
    return {
        "records": records,
        "passed": all(record["passed"] for record in records.values()),
    }


def environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=False,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver or None,
        "device": torch.cuda.get_device_name(0),
        "device_uuid": str(getattr(properties, "uuid", "unknown")),
        "device_total_memory_bytes": properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "multiprocessor_count": properties.multi_processor_count,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def deterministic_fv_state(dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(-0.4, 0.7, 37, dtype=dtype)


def scalar_fv_agreement() -> dict[str, Any]:
    cases = {}
    for dtype, tolerance in ((torch.float32, 2.0e-4), (torch.float64, 2.0e-11)):
        cpu_state = deterministic_fv_state(dtype)
        gpu_state = cpu_state.cuda()
        cpu_left, cpu_right = fv_weno5_face_states(cpu_state)
        gpu_left, gpu_right = fv_weno5_face_states(gpu_state)
        cpu_rhs = fv_weno5_rhs(cpu_state, 1.0 / 37.0, lambda value: value, 1.0)
        gpu_rhs = fv_weno5_rhs(gpu_state, 1.0 / 37.0, lambda value: value, 1.0)
        differences = {
            "left_face_maximum_absolute_difference": float(
                torch.max(torch.abs(gpu_left.cpu() - cpu_left))
            ),
            "right_face_maximum_absolute_difference": float(
                torch.max(torch.abs(gpu_right.cpu() - cpu_right))
            ),
            "rhs_maximum_absolute_difference": float(
                torch.max(torch.abs(gpu_rhs.cpu() - cpu_rhs))
            ),
        }
        passed = (
            max(differences.values()) <= tolerance
            and bool(torch.isfinite(gpu_left).all())
            and bool(torch.isfinite(gpu_right).all())
            and bool(torch.isfinite(gpu_rhs).all())
            and gpu_left.device.type == "cuda"
            and gpu_right.device.type == "cuda"
            and gpu_rhs.device.type == "cuda"
        )
        cases[str(dtype).removeprefix("torch.")] = {
            **differences,
            "tolerance": tolerance,
            "resident": gpu_left.device.type
            == gpu_right.device.type
            == gpu_rhs.device.type
            == "cuda",
            "finite": bool(
                torch.isfinite(gpu_left).all()
                and torch.isfinite(gpu_right).all()
                and torch.isfinite(gpu_rhs).all()
            ),
            "passed": passed,
        }
    return {"cases": cases, "passed": all(case["passed"] for case in cases.values())}


def compile_case(
    name: str,
    call: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    tolerance: float,
) -> dict[str, Any]:
    eager = call(state)
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(call)(state)
    torch._dynamo.reset()
    compiled = torch.compile(call, fullgraph=True, dynamic=False)(state)
    difference = float(torch.max(torch.abs(compiled - eager)))
    finite = bool(torch.isfinite(compiled).all())
    resident = compiled.device == state.device
    passed = (
        explanation.graph_count == 1
        and explanation.graph_break_count == 0
        and difference <= tolerance
        and finite
        and resident
    )
    return {
        "name": name,
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "break_reasons": [str(reason) for reason in explanation.break_reasons],
        "compiled_eager_maximum_absolute_difference": difference,
        "tolerance": tolerance,
        "finite": finite,
        "resident": resident,
        "passed": passed,
    }


def scalar_fv_compilation() -> dict[str, Any]:
    cases = {}
    for dtype, tolerance in ((torch.float32, 5.0e-5), (torch.float64, 2.0e-11)):
        state = deterministic_fv_state(dtype).cuda()

        def rhs(values: torch.Tensor) -> torch.Tensor:
            return fv_weno5_rhs(values, 1.0 / 37.0, lambda value: value, 1.0)

        def step(values: torch.Tensor) -> torch.Tensor:
            return ssp_rk3_step(values, 0.01 / 37.0, rhs)

        dtype_name = str(dtype).removeprefix("torch.")
        for name, call in (("rhs", rhs), ("ssp_rk3_step", step)):
            key = f"{dtype_name}_{name}"
            cases[key] = compile_case(key, call, state, tolerance)
    return {"cases": cases, "passed": all(case["passed"] for case in cases.values())}


def is_movement_event(name: str) -> bool:
    lowered = name.lower()
    return name in {"aten::_to_copy", "aten::copy_"} or any(
        marker in lowered
        for marker in (
            "memcpy",
            "host to device",
            "device to host",
            "h2d",
            "d2h",
        )
    )


def scalar_fv_movement() -> dict[str, Any]:
    state = deterministic_fv_state(torch.float64).cuda()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        result = fv_weno5_rhs(state, 1.0 / 37.0, lambda value: value, 1.0)
        torch.cuda.synchronize()
    averaged = list(profiler.key_averages())
    movement = sorted(event.key for event in averaged if is_movement_event(event.key))
    to_events = []
    for event in averaged:
        if event.key == "aten::to":
            to_events.append(
                {
                    "key": event.key,
                    "count": event.count,
                    "cpu_memory_usage": getattr(event, "cpu_memory_usage", None),
                    "self_cpu_memory_usage": getattr(
                        event, "self_cpu_memory_usage", None
                    ),
                    "device_memory_usage": getattr(
                        event, "device_memory_usage", None
                    ),
                    "self_device_memory_usage": getattr(
                        event, "self_device_memory_usage", None
                    ),
                }
            )
    resident = result.device == state.device and result.dtype == state.dtype
    return {
        "aten_to_events": to_events,
        "movement_events": movement,
        "input_device": str(state.device),
        "output_device": str(result.device),
        "dtype": str(result.dtype).removeprefix("torch."),
        "resident": resident,
        "passed": not movement and resident,
    }


def euler_state(dtype: torch.dtype) -> torch.Tensor:
    x = (torch.arange(37, dtype=dtype) + 0.5) / 37.0
    density = 1.0 + 0.1 * torch.sin(2.0 * math.pi * x)
    velocity = 0.2 + 0.05 * torch.cos(2.0 * math.pi * x)
    pressure = 1.0 + 0.05 * torch.sin(4.0 * math.pi * x)
    energy = pressure / 0.4 + 0.5 * density * velocity.square()
    return torch.stack((density, density * velocity, energy))


def euler_agreement() -> dict[str, Any]:
    cases = {}
    for order in ORDERS:
        for dtype, tolerance in ((torch.float32, 3.0e-4), (torch.float64, 5.0e-11)):
            cpu_state = euler_state(dtype)
            gpu_state = cpu_state.cuda()
            for boundary in BOUNDARIES:
                cpu = euler1d_rhs(
                    cpu_state, 1.0 / 37.0, order=order, boundary=boundary
                )
                gpu = euler1d_rhs(
                    gpu_state, 1.0 / 37.0, order=order, boundary=boundary
                )
                difference = float(torch.max(torch.abs(gpu.cpu() - cpu)))
                key = (
                    f"order{order}_{str(dtype).removeprefix('torch.')}_{boundary}"
                )
                cases[key] = {
                    "order": order,
                    "dtype": str(dtype).removeprefix("torch."),
                    "boundary": boundary,
                    "maximum_absolute_difference": difference,
                    "tolerance": tolerance,
                    "finite": bool(torch.isfinite(gpu).all()),
                    "resident": gpu.device.type == "cuda",
                    "passed": difference <= tolerance
                    and bool(torch.isfinite(gpu).all())
                    and gpu.device.type == "cuda",
                }
    return {"cases": cases, "passed": all(case["passed"] for case in cases.values())}


def euler_compilation() -> dict[str, Any]:
    cases = {}
    state = euler_state(torch.float64).cuda()
    for order in REPRESENTATIVE_ORDERS:
        for boundary in BOUNDARIES:

            def rhs(
                values: torch.Tensor,
                order: int = order,
                boundary: str = boundary,
            ) -> torch.Tensor:
                return euler1d_rhs(
                    values, 1.0 / 37.0, order=order, boundary=boundary
                )

            key = f"order{order}_float64_{boundary}"
            cases[key] = compile_case(key, rhs, state, 5.0e-11)
    return {"cases": cases, "passed": all(case["passed"] for case in cases.values())}


def euler_cfl() -> dict[str, Any]:
    state = euler_state(torch.float64).cuda()
    timestep = euler1d_cfl_timestep(state, 1.0 / 37.0)
    finite = bool(torch.isfinite(timestep))
    positive = bool(timestep > 0.0)
    resident = timestep.device == state.device
    return {
        "shape": list(timestep.shape),
        "dtype": str(timestep.dtype).removeprefix("torch."),
        "device": str(timestep.device),
        "finite": finite,
        "positive": positive,
        "resident": resident,
        "passed": timestep.ndim == 0 and finite and positive and resident,
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
        raise RuntimeError("deferred CUDA gates require a clean source tree")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the qualification process")
    predecessors = verify_predecessors()
    if not predecessors["passed"]:
        raise RuntimeError("a predecessor record did not verify")
    fv_agreement = scalar_fv_agreement()
    fv_compiler = scalar_fv_compilation()
    fv_movement = scalar_fv_movement()
    euler_device = euler_agreement()
    euler_compiler = euler_compilation()
    cfl = euler_cfl()
    gates = {
        "predecessors": predecessors["passed"],
        "scalar_fv_cpu_cuda_agreement": fv_agreement["passed"],
        "scalar_fv_cuda_compilation": fv_compiler["passed"],
        "scalar_fv_cuda_movement": fv_movement["passed"],
        "euler_cpu_cuda_agreement": euler_device["passed"],
        "euler_cuda_compilation": euler_compiler["passed"],
        "euler_cuda_cfl": cfl["passed"],
    }
    payload = {
        "schema_version": 1,
        "phase": "deferred_cuda_correctness_gates",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "protocol": "docs/DEFERRED_CUDA_GATES_PROTOCOL.md",
        "source_hashes": {
            "docs/DEFERRED_CUDA_GATES_PROTOCOL.md": sha256(PROTOCOL),
            "experiments/deferred_cuda_gates/qualify.py": sha256(Path(__file__)),
            "src/gradflow/fv_weno5.py": sha256(FV_SOURCE),
            "src/gradflow/weno_js.py": sha256(WENO_SOURCE),
            "src/gradflow/euler1d.py": sha256(EULER_SOURCE),
        },
        "environment": environment(),
        "predecessors": predecessors,
        "scalar_fv": {
            "cpu_cuda_agreement": fv_agreement,
            "cuda_compilation": fv_compiler,
            "cuda_movement": fv_movement,
        },
        "euler1d": {
            "cpu_cuda_agreement": euler_device,
            "cuda_compilation": euler_compiler,
            "cuda_cfl": cfl,
        },
        "mps": {"status": "untested_unavailable", "available": False},
        "gate_decisions": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "passed": all(gates.values()),
        "performance_measurements_collected": False,
    }
    output.mkdir(parents=True)
    record_path = output / "qualification.json"
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "SHA256SUMS").write_text(
        f"{sha256(record_path)}  qualification.json\n"
    )
    print(f"wrote deferred CUDA qualification to {record_path}")


if __name__ == "__main__":
    main()
