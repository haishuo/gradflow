"""Untimed numerical helpers for the frozen Euler Phase-6B qualification."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from experiments.fd_fv_euler.phase6a_oracle import (
    GAMMA,
    entropy_average,
    entropy_point,
)
from experiments.euler_boundary_shock.sod_exact import sod_solution
from gradflow import (
    euler1d_cfl_timestep,
    euler1d_fv_cfl_timestep,
    euler1d_fv_rhs,
    euler1d_fv_rhs_with_boundary_fluxes,
    euler1d_rhs,
    euler1d_rhs_with_boundary_fluxes,
)


ROOT = Path(__file__).resolve().parents[2]
PROJECTIONS = (
    ROOT
    / "experiments/fd_fv_euler/results/phase_6a_20260828/projections.npz"
)
SIZES = (24, 36, 54, 81)
SHOCK_SIZES = (200, 400, 800)
METHODS = ("fd", "fv")
BOUNDARIES = ("periodic", "transmissive")
FINAL_SMOOTH_TIME = 0.1


Rhs = Callable[[torch.Tensor, float, str], torch.Tensor]


def primitive_to_conserved(primitive: torch.Tensor) -> torch.Tensor:
    density, velocity, pressure = primitive
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity.square()
    return torch.stack((density, density * velocity, energy))


def conserved_to_primitive(conserved: torch.Tensor) -> torch.Tensor:
    density = conserved[0]
    velocity = conserved[1] / density
    pressure = (GAMMA - 1.0) * (
        conserved[2] - 0.5 * conserved[1].square() / density
    )
    return torch.stack((density, velocity, pressure))


def method_rhs(
    method: str, state: torch.Tensor, dx: float, boundary: str
) -> torch.Tensor:
    if method == "fd":
        return euler1d_rhs(state, dx, order=5, boundary=boundary)
    if method == "fv":
        return euler1d_fv_rhs(state, dx, boundary=boundary)
    raise ValueError(f"unknown method: {method}")


def method_rhs_fluxes(
    method: str, state: torch.Tensor, dx: float, boundary: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if method == "fd":
        return euler1d_rhs_with_boundary_fluxes(
            state, dx, order=5, boundary=boundary
        )
    if method == "fv":
        return euler1d_fv_rhs_with_boundary_fluxes(
            state, dx, boundary=boundary
        )
    raise ValueError(f"unknown method: {method}")


def method_cfl(method: str, state: torch.Tensor, dx: float) -> torch.Tensor:
    if method == "fd":
        return euler1d_cfl_timestep(state, dx, 0.1)
    if method == "fv":
        return euler1d_fv_cfl_timestep(state, dx, 0.1)
    raise ValueError(f"unknown method: {method}")


def smooth_state(method: str, cells: int, time: float) -> torch.Tensor:
    state, _ = (
        entropy_point(cells, time)
        if method == "fd"
        else entropy_average(cells, time)
    )
    return torch.from_numpy(state)


def smooth_rhs(method: str, cells: int, time: float) -> torch.Tensor:
    _, rhs = (
        entropy_point(cells, time)
        if method == "fd"
        else entropy_average(cells, time)
    )
    return torch.from_numpy(rhs)


def rk_stages(
    method: str,
    state: torch.Tensor,
    dx: float,
    dt: float | torch.Tensor,
    boundary: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = state + dt * method_rhs(method, state, dx, boundary)
    second = 0.75 * state + 0.25 * (
        first + dt * method_rhs(method, first, dx, boundary)
    )
    third = (
        state
        + 2.0
        * (second + dt * method_rhs(method, second, dx, boundary))
    ) / 3.0
    return first, second, third


def state_minima(state: torch.Tensor) -> tuple[float, float, bool]:
    primitive = conserved_to_primitive(state)
    return (
        float(torch.min(primitive[0])),
        float(torch.min(primitive[2])),
        bool(torch.isfinite(primitive).all()),
    )


def evolve(
    method: str,
    initial: torch.Tensor,
    dx: float,
    final_time: float,
    boundary: str,
) -> tuple[torch.Tensor, dict]:
    state = initial
    time = 0.0
    steps = 0
    minimum_density = math.inf
    minimum_pressure = math.inf
    failure_stage = None
    while time < final_time:
        dt = min(float(method_cfl(method, state, dx)), final_time - time)
        stages = rk_stages(method, state, dx, dt, boundary)
        for stage_index, stage in enumerate(stages, 1):
            density, pressure, finite = state_minima(stage)
            minimum_density = min(minimum_density, density)
            minimum_pressure = min(minimum_pressure, pressure)
            if not finite or density <= 0.0 or pressure <= 0.0:
                failure_stage = f"ssp_rk3_stage_{stage_index}"
                state = stage
                break
        if failure_stage is not None:
            break
        state = stages[-1]
        time += dt
        steps += 1
        if steps > 1_000_000:
            raise RuntimeError("Phase 6B step guard exceeded")
    return state, {
        "completed": failure_stage is None and time >= final_time,
        "failure_stage": failure_stage,
        "steps": steps,
        "simulated_time": time,
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
    }


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
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


def rates(errors: list[float], sizes: tuple[int, ...]) -> list[float]:
    return [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors, errors[1:], sizes, sizes[1:]
        )
    ]


def shock_initial(problem: str, cells: int) -> torch.Tensor:
    if problem == "sod":
        left = torch.tensor([1.0, 0.0, 2.5], dtype=torch.float64)[:, None]
        right = torch.tensor([0.125, 0.0, 0.25], dtype=torch.float64)[:, None]
        indices = torch.arange(cells)
        return torch.where((indices < cells // 2)[None, :], left, right)
    if problem == "shu_osher":
        with np.load(PROJECTIONS) as archive:
            return torch.from_numpy(archive[f"shu_n{cells}_fv_initial"].copy())
    raise ValueError(f"unknown shock problem: {problem}")


def shock_expected(problem: str, cells: int) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(PROJECTIONS) as archive:
        if problem == "sod":
            conserved = torch.from_numpy(
                archive[f"sod_n{cells}_fv_conserved"].copy()
            )
            primitive = torch.from_numpy(
                archive[f"sod_n{cells}_fv_primitive"].copy()
            )
        elif problem == "shu_osher":
            conserved = torch.from_numpy(
                archive[f"shu_n{cells}_fv_reference_conserved"].copy()
            )
            primitive = torch.from_numpy(
                archive[f"shu_n{cells}_fv_reference_primitive"].copy()
            )
        else:
            raise ValueError(f"unknown shock problem: {problem}")
    return conserved, primitive


def sod_wave_metrics(primitive: torch.Tensor, cells: int) -> dict:
    solution = sod_solution()
    exact_locations = {
        "contact": 0.5 + 0.2 * solution.star_velocity,
        "shock": 0.5 + 0.2 * solution.right_head_speed,
    }
    density_jumps = torch.abs(torch.diff(primitive[0]))
    interfaces = torch.arange(1, cells, dtype=torch.float64) / cells
    result = {}
    for name, exact in exact_locations.items():
        mask = torch.abs(interfaces - exact) <= 0.05
        candidates = torch.nonzero(mask).flatten()
        local = candidates[torch.argmax(density_jumps[mask])]
        detected = float(interfaces[local])
        result[name] = {
            "exact": exact,
            "detected": detected,
            "error_cells": abs(detected - exact) * cells,
        }
    return result


def shu_structure(actual: torch.Tensor, expected: torch.Tensor, cells: int) -> dict:
    x = -5.0 + (torch.arange(cells, dtype=torch.float64) + 0.5) * (10.0 / cells)
    window = (x >= -3.0) & (x <= 3.0)
    actual_density = actual[0, window]
    expected_density = expected[0, window]
    actual_centered = actual_density - torch.mean(actual_density)
    expected_centered = expected_density - torch.mean(expected_density)
    correlation = torch.dot(actual_centered, expected_centered) / (
        torch.linalg.vector_norm(actual_centered)
        * torch.linalg.vector_norm(expected_centered)
    )
    actual_tv = torch.sum(torch.abs(torch.diff(actual_density)))
    expected_tv = torch.sum(torch.abs(torch.diff(expected_density)))
    return {
        "window": [-3.0, 3.0],
        "density_correlation": float(correlation),
        "density_total_variation_ratio": float(actual_tv / expected_tv),
    }
