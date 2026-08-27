#!/usr/bin/env python3
"""Run the frozen correctness-only FD/FV Phase-3 qualification."""

from __future__ import annotations

import argparse
import ast
from fractions import Fraction
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

import gradflow.fv_weno5 as implementation
from gradflow import (
    FV_WENO5_FORMULATION_ID,
    fv_global_lax_friedrichs_flux,
    fv_weno5_face_states,
    fv_weno5_rhs,
    ssp_rk3_step,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "experiments/fd_fv_qualification/results/phase_3_20260827"
)
PHASE2_DIR = ROOT / "experiments/fd_fv_contract/results/phase_2_20260827"
ORACLE_PATH = PHASE2_DIR / "oracle_cases.json"
CONTRACT_PATH = PHASE2_DIR / "contract.json"
PROTOCOL_PATH = ROOT / "docs/FD_FV_PHASE_3_PROTOCOL.md"
MODULE_PATH = ROOT / "src/gradflow/fv_weno5.py"
FORMULATION_ID = "fv_dimensional_js5_global_lf_periodic_v1"
SPATIAL_SIZES = (32, 48, 72, 108)
SMOOTH_SIZES = (24, 36, 54, 81)
DISCONTINUITY_SIZES = (64, 128, 256)


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


def parse_fractions(values: list[str], *, device: str = "cpu") -> torch.Tensor:
    return torch.tensor(
        [float(Fraction(value)) for value in values],
        dtype=torch.float64,
        device=device,
    )


def fourier_cell_averages(
    cells: int,
    *,
    shift: float = 0.0,
    device: str = "cpu",
) -> torch.Tensor:
    dx = 1.0 / cells
    left = torch.arange(cells, dtype=torch.float64, device=device) * dx - shift
    right = left + dx
    return (
        (torch.cos(2.0 * math.pi * left) - torch.cos(2.0 * math.pi * right))
        / (2.0 * math.pi * dx)
        + 0.15
        * (torch.sin(6.0 * math.pi * right) - torch.sin(6.0 * math.pi * left))
        / (6.0 * math.pi * dx)
    )


def fourier_cell_average_derivative(cells: int, speed: float) -> torch.Tensor:
    dx = 1.0 / cells
    faces = torch.arange(cells + 1, dtype=torch.float64) * dx
    values = torch.sin(2.0 * math.pi * faces) + 0.15 * torch.cos(
        6.0 * math.pi * faces
    )
    return -speed * (values[1:] - values[:-1]) / dx


def indicator_cell_averages(
    cells: int,
    *,
    shift: float,
    device: str = "cpu",
) -> torch.Tensor:
    start = (0.2 + shift) % 1.0
    end = start + 0.4
    segments = [(start, min(end, 1.0))]
    if end > 1.0:
        segments.append((0.0, end - 1.0))
    dx = 1.0 / cells
    left = torch.arange(cells, dtype=torch.float64, device=device) * dx
    right = left + dx
    result = torch.zeros(cells, dtype=torch.float64, device=device)
    for segment_left, segment_right in segments:
        overlap = torch.clamp(
            torch.minimum(right, torch.tensor(segment_right, device=device))
            - torch.maximum(left, torch.tensor(segment_left, device=device)),
            min=0.0,
        )
        result = result + overlap / dx
    return result


def evolve(
    state: torch.Tensor,
    *,
    dx: float,
    final_time: float,
    nominal_dt: float,
) -> tuple[torch.Tensor, int]:
    def rhs(values: torch.Tensor) -> torch.Tensor:
        return fv_weno5_rhs(values, dx, lambda value: value, 1.0)

    result = state
    current_time = 0.0
    steps = 0
    while current_time < final_time:
        timestep = min(nominal_dt, final_time - current_time)
        result = ssp_rk3_step(result, timestep, rhs)
        current_time += timestep
        steps += 1
        if steps > 100_000:
            raise RuntimeError("Phase-3 step guard exceeded")
    return result, steps


def oracle_parity() -> dict[str, Any]:
    record = json.loads(ORACLE_PATH.read_text())
    semidiscrete = record["semidiscrete"]
    state = parse_fractions(semidiscrete["deterministic_cell_averages"])
    spacing = float(Fraction(semidiscrete["spacing"]))
    directions = {}
    for direction in ("positive", "negative"):
        expected = semidiscrete["linear_advection"][direction]
        speed = float(Fraction(expected["speed"]))
        left, right = fv_weno5_face_states(state)
        fluxes = fv_global_lax_friedrichs_flux(
            left, right, lambda value, speed=speed: speed * value, abs(speed)
        )
        rhs = fv_weno5_rhs(
            state,
            spacing,
            lambda value, speed=speed: speed * value,
            abs(speed),
        )
        differences = {
            "left": float(
                torch.max(
                    torch.abs(left - parse_fractions(expected["left_face_states"]))
                )
            ),
            "right": float(
                torch.max(
                    torch.abs(right - parse_fractions(expected["right_face_states"]))
                )
            ),
            "flux": float(
                torch.max(torch.abs(fluxes - parse_fractions(expected["face_fluxes"])))
            ),
            "rhs": float(
                torch.max(torch.abs(rhs - parse_fractions(expected["rhs"])))
            ),
        }
        directions[direction] = {
            "maximum_absolute_differences": differences,
            "passed": max(differences.values()) <= 2.0e-13,
        }

    constants = {}
    for dtype, tolerance in ((torch.float32, 2.0e-6), (torch.float64, 5.0e-15)):
        state = torch.full((2, 37), 7.0 / 3.0, dtype=dtype)
        left, right = fv_weno5_face_states(state)
        rhs = fv_weno5_rhs(state, 1.0 / 37.0, lambda value: 2.0 * value, 2.0)
        error = max(
            float(torch.max(torch.abs(left - state))),
            float(torch.max(torch.abs(right - state))),
            float(torch.max(torch.abs(rhs))),
        )
        constants[str(dtype).removeprefix("torch.")] = {
            "maximum_absolute_difference": error,
            "tolerance": tolerance,
            "passed": error <= tolerance,
        }
    return {
        "directions": directions,
        "constants": constants,
        "passed": all(item["passed"] for item in directions.values())
        and all(item["passed"] for item in constants.values()),
    }


def refusal_contract() -> dict[str, Any]:
    state = torch.ones(8, dtype=torch.float64)
    checks: dict[str, bool] = {}

    def rejects(name: str, exception: type[Exception], call: Callable[[], Any]) -> None:
        try:
            call()
        except exception:
            checks[name] = True
        else:
            checks[name] = False

    rejects(
        "integer_state",
        TypeError,
        lambda: fv_weno5_face_states(torch.ones(8, dtype=torch.int64)),
    )
    rejects(
        "too_few_cells",
        ValueError,
        lambda: fv_weno5_face_states(torch.ones(4, dtype=torch.float64)),
    )
    rejects("invalid_axis", ValueError, lambda: fv_weno5_face_states(state, axis=3))
    rejects(
        "invalid_bias",
        ValueError,
        lambda: implementation._fv_weno5_reconstruct(
            state, bias="center", axis=-1
        ),
    )
    rejects(
        "nonpositive_python_dx",
        ValueError,
        lambda: fv_weno5_rhs(state, 0.0, lambda value: value, 1.0),
    )
    rejects(
        "nonpositive_python_alpha",
        ValueError,
        lambda: fv_weno5_rhs(state, 1.0, lambda value: value, -1.0),
    )
    rejects(
        "nonscalar_tensor_alpha",
        ValueError,
        lambda: fv_weno5_rhs(
            state, 1.0, lambda value: value, torch.ones(2, dtype=torch.float64)
        ),
    )
    rejects(
        "wrong_dtype_tensor_dx",
        TypeError,
        lambda: fv_weno5_rhs(
            state,
            torch.tensor(1.0, dtype=torch.float32),
            lambda value: value,
            1.0,
        ),
    )
    rejects(
        "flux_shape",
        ValueError,
        lambda: fv_weno5_rhs(state, 1.0, lambda value: value[:-1], 1.0),
    )
    rejects(
        "flux_dtype",
        TypeError,
        lambda: fv_weno5_rhs(state, 1.0, lambda value: value.float(), 1.0),
    )
    return {"checks": checks, "passed": all(checks.values())}


def smooth_spatial() -> dict[str, Any]:
    directions = {}
    for speed in (1.0, -1.0):
        errors = []
        for cells in SPATIAL_SIZES:
            state = fourier_cell_averages(cells)
            actual = fv_weno5_rhs(
                state,
                1.0 / cells,
                lambda value, speed=speed: speed * value,
                abs(speed),
            )
            exact = fourier_cell_average_derivative(cells, speed)
            errors.append(float(torch.sqrt(torch.mean((actual - exact).square()))))
        observed = rates(errors, SPATIAL_SIZES)
        directions[str(int(speed))] = {
            "sizes": SPATIAL_SIZES,
            "l2_errors": errors,
            "rates": observed,
            "monotone": all(fine < coarse for coarse, fine in zip(errors, errors[1:])),
            "maximum_rate": max(observed),
            "passed": all(
                fine < coarse for coarse, fine in zip(errors, errors[1:])
            )
            and max(observed) >= 4.7,
        }
    return {
        "directions": directions,
        "passed": all(item["passed"] for item in directions.values()),
    }


def smooth_complete_solve() -> dict[str, Any]:
    l1_errors = []
    l2_errors = []
    runs = []
    for cells in SMOOTH_SIZES:
        dx = 1.0 / cells
        initial = fourier_cell_averages(cells)
        final, steps = evolve(
            initial,
            dx=dx,
            final_time=0.01,
            nominal_dt=0.2 * dx ** (5.0 / 3.0),
        )
        exact = fourier_cell_averages(cells, shift=0.01)
        l1 = float(torch.mean(torch.abs(final - exact)))
        l2 = float(torch.sqrt(torch.mean((final - exact).square())))
        mass_change = float(torch.abs(dx * torch.sum(final - initial)))
        bound = float(
            32.0
            * torch.finfo(torch.float64).eps
            * dx
            * torch.sum(torch.abs(initial))
            + 1.0e-15
        )
        l1_errors.append(l1)
        l2_errors.append(l2)
        runs.append(
            {
                "cells": cells,
                "steps": steps,
                "l1_error": l1,
                "l2_error": l2,
                "mass_change": mass_change,
                "mass_bound": bound,
                "conservation_passed": mass_change <= bound,
            }
        )
    l2_rates = rates(l2_errors, SMOOTH_SIZES)
    passed = (
        all(fine < coarse for coarse, fine in zip(l1_errors, l1_errors[1:]))
        and all(fine < coarse for coarse, fine in zip(l2_errors, l2_errors[1:]))
        and max(l2_rates) >= 4.0
        and all(run["conservation_passed"] for run in runs)
    )
    return {
        "runs": runs,
        "l1_rates": rates(l1_errors, SMOOTH_SIZES),
        "l2_rates": l2_rates,
        "passed": passed,
    }


def discontinuity() -> dict[str, Any]:
    errors = []
    runs = []
    for cells in DISCONTINUITY_SIZES:
        dx = 1.0 / cells
        initial = indicator_cell_averages(cells, shift=0.0)
        final, steps = evolve(
            initial,
            dx=dx,
            final_time=0.2,
            nominal_dt=0.2 * dx,
        )
        exact = indicator_cell_averages(cells, shift=0.2)
        error = float(torch.mean(torch.abs(final - exact)))
        minimum = float(torch.min(final))
        maximum = float(torch.max(final))
        mass_change = float(torch.abs(dx * torch.sum(final - initial)))
        bound = float(
            32.0
            * torch.finfo(torch.float64).eps
            * dx
            * torch.sum(torch.abs(initial))
            + 1.0e-15
        )
        errors.append(error)
        runs.append(
            {
                "cells": cells,
                "steps": steps,
                "l1_error": error,
                "minimum": minimum,
                "maximum": maximum,
                "finite": bool(torch.isfinite(final).all()),
                "mass_change": mass_change,
                "mass_bound": bound,
                "passed": math.isfinite(error)
                and minimum >= -0.1
                and maximum <= 1.1
                and mass_change <= bound,
            }
        )
    return {
        "runs": runs,
        "monotone_l1": all(
            fine < coarse for coarse, fine in zip(errors, errors[1:])
        ),
        "passed": all(
            fine < coarse for coarse, fine in zip(errors, errors[1:])
        )
        and all(run["passed"] for run in runs),
    }


def differentiation() -> dict[str, Any]:
    cells = 19
    state = torch.linspace(
        -0.4, 0.7, cells, dtype=torch.float64, requires_grad=True
    )

    def rhs(values: torch.Tensor) -> torch.Tensor:
        return fv_weno5_rhs(
            values,
            1.0 / cells,
            lambda value: 0.5 * value.square(),
            1.0,
        )

    def rhs_objective(values: torch.Tensor) -> torch.Tensor:
        return rhs(values).square().mean()

    gradcheck = bool(
        torch.autograd.gradcheck(
            rhs_objective,
            (state,),
            eps=1.0e-6,
            atol=2.0e-5,
            rtol=2.0e-4,
        )
    )

    def step_objective(values: torch.Tensor) -> torch.Tensor:
        result = values
        for _ in range(3):
            result = ssp_rk3_step(result, 0.01 / cells, rhs)
        return result.square().mean() + 0.1 * result.sin().mean()

    objective = step_objective(state)
    gradient = torch.autograd.grad(objective, state)[0]
    direction = torch.cos(torch.arange(cells, dtype=torch.float64))
    direction = direction / torch.linalg.vector_norm(direction)
    automatic = torch.sum(gradient * direction)
    step = 1.0e-6
    center = state.detach()
    centered = (
        step_objective(center + step * direction)
        - step_objective(center - step * direction)
    ) / (2.0 * step)
    absolute = float(torch.abs(automatic - centered))
    relative = absolute / max(
        float(torch.abs(centered)), torch.finfo(torch.float64).tiny
    )
    directional_passed = absolute <= 3.0e-6 + 3.0e-5 * float(torch.abs(centered))
    return {
        "rhs_gradcheck": gradcheck,
        "fixed_three_step": {
            "automatic": float(automatic),
            "centered": float(centered),
            "absolute_error": absolute,
            "relative_error": relative,
            "gradient_finite": bool(torch.isfinite(gradient).all()),
            "passed": bool(torch.isfinite(gradient).all()) and directional_passed,
        },
        "passed": gradcheck
        and bool(torch.isfinite(gradient).all())
        and directional_passed,
    }


def eager_and_device() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cpu = {}
    for dtype in (torch.float32, torch.float64):
        state = torch.linspace(-0.4, 0.7, 37, dtype=dtype)
        left, right = fv_weno5_face_states(state)
        rhs = fv_weno5_rhs(state, 1.0 / 37.0, lambda value: value, 1.0)
        key = str(dtype).removeprefix("torch.")
        cpu[key] = {
            "finite": bool(
                torch.isfinite(left).all()
                and torch.isfinite(right).all()
                and torch.isfinite(rhs).all()
            ),
            "dtype_preserved": left.dtype == right.dtype == rhs.dtype == dtype,
            "device_preserved": (
                left.device == right.device == rhs.device == state.device
            ),
        }
        cpu[key]["passed"] = all(cpu[key].values())
    cpu_record = {
        "dtypes": cpu,
        "passed": all(item["passed"] for item in cpu.values()),
    }

    if torch.cuda.is_available():
        cuda_cases = {}
        for dtype, tolerance in ((torch.float32, 2.0e-4), (torch.float64, 2.0e-11)):
            cpu_state = torch.linspace(-0.4, 0.7, 37, dtype=dtype)
            gpu_state = cpu_state.cuda()
            cpu_rhs = fv_weno5_rhs(cpu_state, 1.0 / 37.0, lambda value: value, 1.0)
            gpu_rhs = fv_weno5_rhs(gpu_state, 1.0 / 37.0, lambda value: value, 1.0)
            difference = float(torch.max(torch.abs(gpu_rhs.cpu() - cpu_rhs)))
            key = str(dtype).removeprefix("torch.")
            cuda_cases[key] = {
                "maximum_absolute_difference": difference,
                "tolerance": tolerance,
                "resident": gpu_rhs.device.type == "cuda",
                "passed": difference <= tolerance and gpu_rhs.device.type == "cuda",
            }
        cuda = {
            "status": (
                "passed"
                if all(case["passed"] for case in cuda_cases.values())
                else "failed"
            ),
            "device": torch.cuda.get_device_name(),
            "cases": cuda_cases,
        }
    else:
        cuda = {"status": "untested_unavailable", "available": False}

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    mps = (
        {"status": "not_executed", "available": True}
        if mps_available
        else {"status": "untested_unavailable", "available": False}
    )
    return cpu_record, cuda, mps


def compile_graphs() -> dict[str, Any]:
    cells = 37
    state = torch.linspace(-0.4, 0.7, cells, dtype=torch.float64)

    def rhs(values: torch.Tensor) -> torch.Tensor:
        return fv_weno5_rhs(values, 1.0 / cells, lambda value: value, 1.0)

    def step(values: torch.Tensor) -> torch.Tensor:
        return ssp_rk3_step(values, 0.01 / cells, rhs)

    cpu = {}
    for name, function in (("rhs", rhs), ("ssp_rk3_step", step)):
        eager = function(state)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(function)(state)
        torch._dynamo.reset()
        compiled = torch.compile(function, fullgraph=True, dynamic=False)
        actual = compiled(state)
        difference = float(torch.max(torch.abs(actual - eager)))
        cpu[name] = {
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "break_reasons": [str(reason) for reason in explanation.break_reasons],
            "maximum_absolute_difference": difference,
            "passed": explanation.graph_count == 1
            and explanation.graph_break_count == 0
            and difference <= 2.0e-12,
        }

    if torch.cuda.is_available():
        cuda_state = state.float().cuda()

        def cuda_rhs(values: torch.Tensor) -> torch.Tensor:
            return fv_weno5_rhs(values, 1.0 / cells, lambda value: value, 1.0)

        eager = cuda_rhs(cuda_state)
        torch._dynamo.reset()
        explanation = torch._dynamo.explain(cuda_rhs)(cuda_state)
        torch._dynamo.reset()
        actual = torch.compile(cuda_rhs, fullgraph=True, dynamic=False)(cuda_state)
        difference = float(torch.max(torch.abs(actual - eager)))
        cuda: dict[str, Any] = {
            "status": "passed"
            if explanation.graph_count == 1
            and explanation.graph_break_count == 0
            and difference <= 5.0e-5
            else "failed",
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "maximum_absolute_difference": difference,
            "resident": actual.device.type == "cuda",
        }
    else:
        cuda = {"status": "untested_unavailable", "available": False}
    return {
        "cpu": cpu,
        "cuda": cuda,
        "compilation_latency_timed": False,
        "passed": all(case["passed"] for case in cpu.values())
        and cuda["status"] in {"passed", "untested_unavailable"},
    }


def transfer_evidence() -> dict[str, Any]:
    source = MODULE_PATH.read_text()
    tree = ast.parse(source, filename=str(MODULE_PATH))
    forbidden = {"cpu", "cuda", "to", "item", "numpy"}
    calls = sorted(
        {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in forbidden
        }
    )
    state = torch.linspace(-0.4, 0.7, 37, dtype=torch.float64)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        fv_weno5_rhs(state, 1.0 / 37.0, lambda value: value, 1.0)
    forbidden_events = sorted(
        event.key
        for event in profiler.key_averages()
        if event.key in {"aten::to", "aten::_to_copy", "aten::copy_"}
    )
    return {
        "static_forbidden_calls": calls,
        "profiler_forbidden_events": forbidden_events,
        "passed": not calls and not forbidden_events,
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


def qualification_record() -> dict[str, Any]:
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    oracle = oracle_parity()
    refusals = refusal_contract()
    spatial = smooth_spatial()
    complete = smooth_complete_solve()
    discontinuous = discontinuity()
    gradients = differentiation()
    cpu, cuda, mps = eager_and_device()
    compiler = compile_graphs()
    transfers = transfer_evidence()
    gates = {
        "oracle_parity": oracle["passed"],
        "refusal_contract": refusals["passed"],
        "smooth_spatial": spatial["passed"],
        "smooth_complete_solve": complete["passed"],
        "discontinuity": discontinuous["passed"],
        "differentiation": gradients["passed"],
        "eager_cpu": cpu["passed"],
        "compiler": compiler["passed"],
        "transfer_evidence": transfers["passed"],
        "cuda_if_available": cuda["status"] in {"passed", "untested_unavailable"},
        "mps_recorded": mps["status"] in {
            "passed",
            "not_executed",
            "untested_unavailable",
        },
    }
    return {
        "schema_version": 1,
        "phase": "fd_fv_phase_3",
        "qualification_date": "2026-08-27",
        "formulation_id": FV_WENO5_FORMULATION_ID,
        "protocol_commit": "3fe79a93e44ace4ec64b3b19368a5fb603fc7903",
        "protocol_amendment_commit": "d8be254",
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "source_hashes": {
            "src/gradflow/fv_weno5.py": sha256(MODULE_PATH),
            "experiments/fd_fv_qualification/qualify_phase_3.py": sha256(
                Path(__file__)
            ),
            "docs/FD_FV_PHASE_3_PROTOCOL.md": sha256(PROTOCOL_PATH),
        },
        "phase_2_hashes": {
            "contract.json": sha256(CONTRACT_PATH),
            "oracle_cases.json": sha256(ORACLE_PATH),
        },
        "environment": environment(),
        "oracle_parity": oracle,
        "refusal_contract": refusals,
        "smooth_spatial": spatial,
        "smooth_complete_solve": complete,
        "discontinuity": discontinuous,
        "differentiation": gradients,
        "eager_cpu": cpu,
        "cuda": cuda,
        "mps": mps,
        "compiler": compiler,
        "transfer_evidence": transfers,
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
        raise FileExistsError(f"refusing to overwrite Phase-3 record in {output}")
    output.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(qualification_record(), indent=2, sort_keys=True) + "\n"
    )
    sums_path.write_text(f"{sha256(record_path)}  qualification.json\n")
    print(f"wrote FD/FV Phase-3 qualification to {record_path}")


if __name__ == "__main__":
    main()
