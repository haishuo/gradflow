#!/usr/bin/env python3
"""Run the frozen FD/FV Phase-3R resolution study without timing."""

from __future__ import annotations

import argparse
import ast
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

from gradflow import fv_weno5_face_states, fv_weno5_rhs


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "experiments/fd_fv_qualification/results/phase_3r_20260827"
)
PROTOCOL_PATH = ROOT / "docs/FD_FV_PHASE_3_RESOLUTION_PROTOCOL.md"
FV_MODULE_PATH = ROOT / "src/gradflow/fv_weno5.py"
WENO_MODULE_PATH = ROOT / "src/gradflow/weno_js.py"
PHASE2_VERIFY = ROOT / "experiments/fd_fv_contract/verify_phase_2.py"
PHASE3_VERIFY = ROOT / "experiments/fd_fv_qualification/verify_phase_3.py"
PHASE3_RECORD = (
    ROOT
    / "experiments/fd_fv_qualification/results/phase_3_20260827/qualification.json"
)
PHASE3_SUMS = PHASE3_RECORD.with_name("SHA256SUMS")
PROTOCOL_COMMIT = "93ad80b"
CANONICAL_SOURCE_COMMIT = "1d920ea97ed7abec9e4e451b377343cf72316f4c"
CANONICAL_SOURCE_SHA256 = (
    "58b6c55b1fe1e84a5f0eaeb30f31acabf25d0cda713b02c9090085c04c3dbed0"
)
SIZES = (32, 48, 72, 108)
CRITICAL_SIZES = (32, 64, 128, 256)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def rates(errors: list[float], sizes: tuple[int, ...]) -> list[float]:
    return [
        math.log(coarse / fine) / math.log(fine_size / coarse_size)
        for coarse, fine, coarse_size, fine_size in zip(
            errors, errors[1:], sizes, sizes[1:]
        )
    ]


def error_record(errors: list[float], sizes: tuple[int, ...]) -> dict[str, Any]:
    observed = rates(errors, sizes)
    monotone = all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    return {
        "sizes": sizes,
        "l2_errors": errors,
        "rates": observed,
        "monotone": monotone,
        "last_two_rates": observed[-2:],
        "passed": monotone and all(rate >= 4.7 for rate in observed[-2:]),
    }


def exponential_cell_averages(cells: int) -> torch.Tensor:
    dx = 1.0 / cells
    left = torch.arange(cells, dtype=torch.float64) * dx
    return torch.exp(left) * math.expm1(dx) / dx


def noncritical_design_order() -> dict[str, Any]:
    errors: dict[str, list[float]] = {
        "left_face": [],
        "right_face": [],
        "positive_rhs": [],
        "negative_rhs": [],
    }
    evaluated_indices: dict[str, list[int]] = {}
    for cells in SIZES:
        dx = 1.0 / cells
        state = exponential_cell_averages(cells)
        face_indices = torch.arange(cells, dtype=torch.float64)
        exact_faces = torch.exp((face_indices + 1.0) * dx)
        left, right = fv_weno5_face_states(state)

        left_slice = slice(2, cells - 2)
        right_slice = slice(1, cells - 3)
        errors["left_face"].append(
            float(torch.sqrt(torch.mean((left[left_slice] - exact_faces[left_slice]) ** 2)))
        )
        errors["right_face"].append(
            float(
                torch.sqrt(
                    torch.mean((right[right_slice] - exact_faces[right_slice]) ** 2)
                )
            )
        )

        exact_rhs = -(
            torch.exp((face_indices + 1.0) * dx) - torch.exp(face_indices * dx)
        ) / dx
        positive = fv_weno5_rhs(state, dx, lambda value: value, 1.0)
        negative = fv_weno5_rhs(state, dx, lambda value: -value, 1.0)
        positive_slice = slice(3, cells - 2)
        negative_slice = slice(2, cells - 3)
        errors["positive_rhs"].append(
            float(
                torch.sqrt(
                    torch.mean((positive[positive_slice] - exact_rhs[positive_slice]) ** 2)
                )
            )
        )
        errors["negative_rhs"].append(
            float(
                torch.sqrt(
                    torch.mean((negative[negative_slice] + exact_rhs[negative_slice]) ** 2)
                )
            )
        )
        if not evaluated_indices:
            evaluated_indices = {
                "left_face": [2, cells - 3],
                "right_face": [1, cells - 4],
                "positive_rhs": [3, cells - 3],
                "negative_rhs": [2, cells - 4],
            }

    sequences = {
        name: error_record(values, SIZES) for name, values in errors.items()
    }
    return {
        "field": "exp(x)",
        "dtype": "float64",
        "evaluated_index_ranges_at_n32": evaluated_indices,
        "sequences": sequences,
        "passed": all(sequence["passed"] for sequence in sequences.values()),
    }


def fourier_cell_averages(cells: int) -> torch.Tensor:
    dx = 1.0 / cells
    left = torch.arange(cells, dtype=torch.float64) * dx
    right = left + dx
    return (
        (torch.cos(2.0 * math.pi * left) - torch.cos(2.0 * math.pi * right))
        / (2.0 * math.pi * dx)
        + 0.15
        * (torch.sin(6.0 * math.pi * right) - torch.sin(6.0 * math.pi * left))
        / (6.0 * math.pi * dx)
    )


def fourier_derivative(cells: int, speed: float) -> torch.Tensor:
    dx = 1.0 / cells
    faces = torch.arange(cells + 1, dtype=torch.float64) * dx
    values = torch.sin(2.0 * math.pi * faces) + 0.15 * torch.cos(
        6.0 * math.pi * faces
    )
    return -speed * (values[1:] - values[:-1]) / dx


def reproduce_mixed_fourier() -> dict[str, Any]:
    original = json.loads(PHASE3_RECORD.read_text())["smooth_spatial"]
    directions = {}
    maximum_difference = 0.0
    for speed in (1.0, -1.0):
        errors = []
        for cells in SIZES:
            state = fourier_cell_averages(cells)
            actual = fv_weno5_rhs(
                state,
                1.0 / cells,
                lambda value, speed=speed: speed * value,
                abs(speed),
            )
            exact = fourier_derivative(cells, speed)
            errors.append(float(torch.sqrt(torch.mean((actual - exact) ** 2))))
        key = str(int(speed))
        expected = original["directions"][key]["l2_errors"]
        difference = max(abs(actual - prior) for actual, prior in zip(errors, expected))
        maximum_difference = max(maximum_difference, difference)
        directions[key] = {
            "sizes": SIZES,
            "l2_errors": errors,
            "rates": rates(errors, SIZES),
            "original_l2_errors": expected,
            "maximum_absolute_reproduction_difference": difference,
        }
    return {
        "directions": directions,
        "maximum_absolute_reproduction_difference": maximum_difference,
        "passed": maximum_difference <= 1.0e-15,
    }


def aligned_critical_point() -> dict[str, Any]:
    left_errors = []
    right_errors = []
    indices = []
    for cells in CRITICAL_SIZES:
        dx = 1.0 / cells
        cell_left = torch.arange(cells, dtype=torch.float64) * dx
        cell_right = cell_left + dx
        averages = (
            torch.cos(2.0 * math.pi * cell_left)
            - torch.cos(2.0 * math.pi * cell_right)
        ) / (2.0 * math.pi * dx)
        left, right = fv_weno5_face_states(averages)
        face_index = cells // 4 - 1
        indices.append(face_index)
        left_errors.append(abs(float(left[face_index]) - 1.0))
        right_errors.append(abs(float(right[face_index]) - 1.0))

    def characterize(errors: list[float]) -> dict[str, Any]:
        finite = all(math.isfinite(error) for error in errors)
        monotone = all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
        return {
            "absolute_errors": errors,
            "rates": rates(errors, CRITICAL_SIZES),
            "finite": finite,
            "monotone": monotone,
            "passed": finite and monotone,
        }

    left_record = characterize(left_errors)
    right_record = characterize(right_errors)
    return {
        "field": "sin(2*pi*x)",
        "critical_face": 0.25,
        "sizes": CRITICAL_SIZES,
        "face_indices": indices,
        "left": left_record,
        "right": right_record,
        "passed": left_record["passed"] and right_record["passed"],
    }


def critical_point_characterization() -> dict[str, Any]:
    mixed = reproduce_mixed_fourier()
    aligned = aligned_critical_point()
    return {
        "mixed_fourier_reproduction": mixed,
        "aligned_simple_critical_point": aligned,
        "passed": mixed["passed"] and aligned["passed"],
    }


def static_source_audit() -> dict[str, Any]:
    forbidden_attributes = {"cpu", "cuda", "item", "numpy"}
    forbidden_calls = []
    dtype_only_to_calls = []
    forbidden_to_calls = []
    for path in (FV_MODULE_PATH, WENO_MODULE_PATH):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            ):
                continue
            relative = str(path.relative_to(ROOT))
            location = {"path": relative, "line": node.lineno}
            if node.func.attr in forbidden_attributes:
                forbidden_calls.append({**location, "attribute": node.func.attr})
            if node.func.attr != "to":
                continue
            keyword_names = {keyword.arg for keyword in node.keywords}
            if not node.args and keyword_names == {"dtype"}:
                dtype_only_to_calls.append(location)
            else:
                forbidden_to_calls.append(
                    {
                        **location,
                        "positional_arguments": len(node.args),
                        "keyword_names": sorted(
                            name if name is not None else "**kwargs"
                            for name in keyword_names
                        ),
                    }
                )
    return {
        "inspected_sources": [
            str(FV_MODULE_PATH.relative_to(ROOT)),
            str(WENO_MODULE_PATH.relative_to(ROOT)),
        ],
        "forbidden_calls": forbidden_calls,
        "forbidden_to_calls": forbidden_to_calls,
        "dtype_only_to_calls": dtype_only_to_calls,
        "passed": not forbidden_calls and not forbidden_to_calls,
    }


def is_movement_event(name: str) -> bool:
    lowered = name.lower()
    return name in {"aten::_to_copy", "aten::copy_"} or any(
        marker in lowered
        for marker in ("memcpy", "host to device", "device to host", "h2d", "d2h")
    )


def profiler_probe(device: str) -> dict[str, Any]:
    state = torch.linspace(-0.4, 0.7, 37, dtype=torch.float64, device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        result = fv_weno5_rhs(state, 1.0 / 37.0, lambda value: value, 1.0)
        if device == "cuda":
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
        "device": device,
        "input_device": str(state.device),
        "output_device": str(result.device),
        "input_dtype": str(state.dtype).removeprefix("torch."),
        "output_dtype": str(result.dtype).removeprefix("torch."),
        "aten_to_events": to_events,
        "movement_events": movement,
        "resident": resident,
        "passed": not movement and resident,
    }


def movement_evidence() -> dict[str, Any]:
    static = static_source_audit()
    cpu = profiler_probe("cpu")
    if torch.cuda.is_available():
        cuda = profiler_probe("cuda")
    else:
        cuda = {"status": "untested_unavailable", "available": False}
    return {
        "static": static,
        "cpu": cpu,
        "cuda": cuda,
        "passed": static["passed"]
        and cpu["passed"]
        and cuda.get("passed", cuda.get("status") == "untested_unavailable"),
    }


def immutable_record_verification() -> dict[str, Any]:
    checks = {}
    for name, script in (("phase_2", PHASE2_VERIFY), ("phase_3", PHASE3_VERIFY)):
        completed = subprocess.run(
            (sys.executable, str(script)),
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        checks[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "passed": completed.returncode == 0,
        }
    expected_hash, expected_name = PHASE3_SUMS.read_text().strip().split("  ", 1)
    manifest = {
        "expected_name": expected_name,
        "expected_sha256": expected_hash,
        "actual_sha256": sha256(PHASE3_RECORD),
        "passed": expected_name == PHASE3_RECORD.name
        and expected_hash == sha256(PHASE3_RECORD),
    }
    return {
        "verifiers": checks,
        "phase_3_manifest": manifest,
        "passed": all(check["passed"] for check in checks.values())
        and manifest["passed"],
    }


def environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name() if cuda_available else None,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def resolution_record() -> dict[str, Any]:
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    immutable = immutable_record_verification()
    noncritical = noncritical_design_order()
    critical = critical_point_characterization()
    movement = movement_evidence()
    source_identity = {
        "canonical_source_commit": CANONICAL_SOURCE_COMMIT,
        "expected_sha256": CANONICAL_SOURCE_SHA256,
        "actual_sha256": sha256(FV_MODULE_PATH),
        "passed": sha256(FV_MODULE_PATH) == CANONICAL_SOURCE_SHA256,
    }
    gates = {
        "immutable_records": immutable["passed"],
        "canonical_source_identity": source_identity["passed"],
        "noncritical_design_order": noncritical["passed"],
        "critical_point_characterization": critical["passed"],
        "movement_evidence": movement["passed"],
    }
    return {
        "schema_version": 1,
        "phase": "fd_fv_phase_3r",
        "resolution_date": "2026-08-27",
        "formulation_id": "fv_dimensional_js5_global_lf_periodic_v1",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "source_hashes": {
            "docs/FD_FV_PHASE_3_RESOLUTION_PROTOCOL.md": sha256(PROTOCOL_PATH),
            "experiments/fd_fv_qualification/resolve_phase_3.py": sha256(
                Path(__file__)
            ),
            "src/gradflow/fv_weno5.py": sha256(FV_MODULE_PATH),
            "src/gradflow/weno_js.py": sha256(WENO_MODULE_PATH),
        },
        "original_record_hashes": {
            str(PHASE3_RECORD.relative_to(ROOT)): sha256(PHASE3_RECORD),
            str(PHASE3_SUMS.relative_to(ROOT)): sha256(PHASE3_SUMS),
        },
        "environment": environment(),
        "immutable_record_verification": immutable,
        "canonical_source_identity": source_identity,
        "noncritical_design_order": noncritical,
        "critical_point_characterization": critical,
        "movement_evidence": movement,
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
    record_path = output / "resolution.json"
    sums_path = output / "SHA256SUMS"
    if record_path.exists() or sums_path.exists():
        raise FileExistsError(f"refusing to overwrite Phase-3R record in {output}")
    output.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(resolution_record(), indent=2, sort_keys=True) + "\n")
    sums_path.write_text(f"{sha256(record_path)}  resolution.json\n")
    print(f"wrote FD/FV Phase-3R resolution to {record_path}")


if __name__ == "__main__":
    main()
