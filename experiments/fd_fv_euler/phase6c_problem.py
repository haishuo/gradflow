"""Frozen Euler helpers shared by Phase-6C isolated timing workers."""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any, Callable

import numpy as np
import torch

from experiments.fd_fv_euler.phase6b_problem import (
    PROJECTIONS,
    conserved_to_primitive,
    method_cfl,
    method_rhs,
    primitive_to_conserved,
    smooth_state,
)
from gradflow import EULER1D_FV_FORMULATION_ID


METHOD_IDS = {
    "fd": "fd_classical_characteristic_js5_global_lf_euler1d_v1",
    "fv": EULER1D_FV_FORMULATION_ID,
}
FINAL_SMOOTH_TIME = 0.1


def smooth_initial(method: str, cells: int) -> torch.Tensor:
    return smooth_state(method, cells, 0.0)


def smooth_expected(method: str, cells: int) -> torch.Tensor:
    return smooth_state(method, cells, FINAL_SMOOTH_TIME)


def stage_function(
    method: str, cells: int, boundary: str
) -> Callable[
    [torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    dx = (10.0 if boundary == "transmissive_shu_osher" else 1.0) / cells
    numerical_boundary = (
        "transmissive" if boundary.startswith("transmissive") else boundary
    )

    def stages(
        state: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = state + dt * method_rhs(
            method, state, dx, numerical_boundary
        )
        second = 0.75 * state + 0.25 * (
            first + dt * method_rhs(method, first, dx, numerical_boundary)
        )
        third = (
            state
            + 2.0
            * (
                second
                + dt * method_rhs(method, second, dx, numerical_boundary)
            )
        ) / 3.0
        return first, second, third

    return stages


def fixed_step_function(
    method: str, cells: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    dx = 1.0 / cells
    dt = 0.05 * dx
    stages = stage_function(method, cells, "periodic")

    def step(state: torch.Tensor) -> torch.Tensor:
        device_dt = state.new_tensor(dt)
        return stages(state, device_dt)[-1]

    return step


def adaptive_solve(
    method: str,
    initial: torch.Tensor,
    final_time: float,
    boundary: str,
    stages: Callable[
        [torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    *,
    check_stages: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    state = initial
    cells = state.shape[-1]
    dx = (10.0 if boundary == "transmissive_shu_osher" else 1.0) / cells
    numerical_boundary = (
        "transmissive" if boundary.startswith("transmissive") else boundary
    )
    time_value = 0.0
    steps = 0
    minimum_density = math.inf
    minimum_pressure = math.inf
    failure_stage = None
    while time_value < final_time:
        cfl_dt = method_cfl(method, state, dx)
        remaining = final_time - time_value
        dt_value = min(float(cfl_dt), remaining)
        dt = torch.minimum(cfl_dt, state.new_tensor(remaining))
        stage_values = stages(state, dt)
        if check_stages:
            for index, stage in enumerate(stage_values, 1):
                primitive = conserved_to_primitive(stage)
                density = float(torch.min(primitive[0]))
                pressure = float(torch.min(primitive[2]))
                finite = bool(torch.isfinite(primitive).all())
                minimum_density = min(minimum_density, density)
                minimum_pressure = min(minimum_pressure, pressure)
                if not finite or density <= 0.0 or pressure <= 0.0:
                    failure_stage = f"ssp_rk3_stage_{index}"
                    state = stage
                    break
        if failure_stage is not None:
            break
        state = stage_values[-1]
        time_value += dt_value
        steps += 1
        if steps > 1_000_000:
            raise RuntimeError("Phase 6C step guard exceeded")
    if not check_stages:
        primitive = conserved_to_primitive(state)
        minimum_density = float(torch.min(primitive[0]))
        minimum_pressure = float(torch.min(primitive[2]))
    return state, {
        "completed": failure_stage is None and time_value >= final_time,
        "failure_stage": failure_stage,
        "steps": steps,
        "simulated_time": time_value,
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
        "boundary": numerical_boundary,
        "cfl_scalar_host_controlled": True,
    }


def error_norms(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual - expected
    return {
        "l1_error": float(torch.mean(torch.abs(difference))),
        "l2_error": float(torch.sqrt(torch.mean(difference.square()))),
        "linf_error": float(torch.max(torch.abs(difference))),
    }


def conservation(
    initial: torch.Tensor,
    final: torch.Tensor,
    dx: float,
    steps: int,
) -> dict[str, Any]:
    drift = torch.abs(dx * torch.sum(final - initial, dim=-1))
    single = (
        64.0
        * torch.finfo(initial.dtype).eps
        * dx
        * torch.sum(torch.abs(initial), dim=-1)
        + 2.0e-15
    )
    accumulated = steps * (single - 2.0e-15) + 2.0e-15
    return {
        "componentwise_drift": drift.tolist(),
        "single_step_roundoff_bound": single.tolist(),
        "accumulated_roundoff_bound": accumulated.tolist(),
        "passed": bool(torch.all(drift <= accumulated)),
    }


def tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def shock_initial(method: str, problem: str, cells: int) -> torch.Tensor:
    if problem == "sod":
        left = torch.tensor([1.0, 0.0, 2.5], dtype=torch.float64)[:, None]
        right = torch.tensor([0.125, 0.0, 0.25], dtype=torch.float64)[:, None]
        indices = torch.arange(cells)
        return torch.where((indices < cells // 2)[None, :], left, right)
    if problem == "shu_osher":
        with np.load(PROJECTIONS) as archive:
            return torch.from_numpy(
                archive[f"shu_n{cells}_{method}_initial"].copy()
            )
    raise ValueError(f"unknown shock problem: {problem}")


def shock_expected(
    method: str, problem: str, cells: int
) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(PROJECTIONS) as archive:
        if problem == "sod":
            conserved = archive[f"sod_n{cells}_{method}_conserved"].copy()
            primitive = archive[f"sod_n{cells}_{method}_primitive"].copy()
        elif problem == "shu_osher":
            conserved = archive[
                f"shu_n{cells}_{method}_reference_conserved"
            ].copy()
            primitive = archive[
                f"shu_n{cells}_{method}_reference_primitive"
            ].copy()
        else:
            raise ValueError(f"unknown shock problem: {problem}")
    return torch.from_numpy(conserved), torch.from_numpy(primitive)


def primitive_error_metrics(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, Any]:
    difference = torch.abs(actual - expected)
    names = ("density", "velocity", "pressure")
    return {
        "l1": {
            name: float(torch.mean(difference[index]))
            for index, name in enumerate(names)
        },
        "l2": {
            name: float(torch.sqrt(torch.mean(difference[index].square())))
            for index, name in enumerate(names)
        },
        "linf": {
            name: float(torch.max(difference[index]))
            for index, name in enumerate(names)
        },
    }


def quantile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def statistics_record(samples: list[float]) -> dict[str, Any]:
    if not samples or not all(
        math.isfinite(value) and value > 0.0 for value in samples
    ):
        raise ValueError("timing samples must be finite and positive")
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
        "q1_seconds": quantile(ordered, 0.25),
        "q3_seconds": quantile(ordered, 0.75),
    }
