"""Frozen matched scalar problem and execution surfaces for FD/FV Phase 4."""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch

from gradflow import WENOJS, fv_weno5_rhs, ssp_rk3_step


Method = Literal["fd", "fv"]
METHODS: tuple[Method, ...] = ("fd", "fv")
METHOD_IDS = {
    "fd": "fd_classical_js5_global_lf_periodic_v1",
    "fv": "fv_dimensional_js5_global_lf_periodic_v1",
}
SIZES = {
    1: (24, 36, 54, 81),
    2: (12, 18, 27, 40),
    3: (8, 12, 18, 27),
}
PHASES = (0.07, 0.19, 0.31)
FINAL_TIME = 0.01
CFL_FACTOR = 0.2
_FD_SCHEME = WENOJS(5)


def velocities(dimension: int) -> tuple[float, ...]:
    if dimension not in SIZES:
        raise ValueError("Phase 4 supports dimensions 1, 2, and 3")
    return tuple(1.0 / dimension for _ in range(dimension))


def timestep(dimension: int, cells: int) -> tuple[float, int]:
    dx = 1.0 / cells
    nominal = CFL_FACTOR * dx ** (5.0 / 3.0) / sum(velocities(dimension))
    steps = math.ceil(FINAL_TIME / nominal)
    return FINAL_TIME / steps, steps


def projected_state(
    method: Method,
    dimension: int,
    cells: int,
    *,
    time: float = 0.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Return the exact method-appropriate projection of the continuous field."""
    if method not in METHODS:
        raise ValueError("method must be 'fd' or 'fv'")
    speeds = velocities(dimension)
    shape = (cells,) * dimension
    result = torch.full(shape, 0.5, dtype=torch.float64, device=device)
    dx = 1.0 / cells
    for axis, (phase, speed) in enumerate(zip(PHASES, speeds)):
        coordinate_shape = [1] * dimension
        coordinate_shape[axis] = cells
        left = (
            torch.arange(cells, dtype=torch.float64, device=device).reshape(
                coordinate_shape
            )
            * dx
        )
        shifted_phase = phase + speed * time
        if method == "fd":
            mode = torch.sin(2.0 * math.pi * (left - shifted_phase))
        else:
            right = left + dx
            mode = (
                torch.cos(2.0 * math.pi * (left - shifted_phase))
                - torch.cos(2.0 * math.pi * (right - shifted_phase))
            ) / (2.0 * math.pi * dx)
        result = result + (0.2 / dimension) * mode
    return result


def rhs_function(
    method: Method,
    dimension: int,
    cells: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    dx = 1.0 / cells
    speeds = velocities(dimension)

    def rhs(values: torch.Tensor) -> torch.Tensor:
        result = None
        for axis, speed in enumerate(speeds):
            if method == "fd":
                directional = _FD_SCHEME.rhs(
                    values,
                    dx,
                    lambda state, speed=speed: speed * state,
                    alpha=abs(speed),
                    axis=axis,
                )
            else:
                directional = fv_weno5_rhs(
                    values,
                    dx,
                    lambda state, speed=speed: speed * state,
                    abs(speed),
                    axis=axis,
                )
            result = directional if result is None else result + directional
        assert result is not None
        return result

    return rhs


def step_function(
    method: Method,
    dimension: int,
    cells: int,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    rhs = rhs_function(method, dimension, cells)
    dt, steps = timestep(dimension, cells)

    def step(values: torch.Tensor) -> torch.Tensor:
        return ssp_rk3_step(values, dt, rhs)

    return step, steps


def solve(
    initial: torch.Tensor,
    step: Callable[[torch.Tensor], torch.Tensor],
    steps: int,
) -> torch.Tensor:
    result = initial
    for _ in range(steps):
        result = step(result)
    return result


def errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = actual - expected
    return (
        float(torch.mean(torch.abs(difference))),
        float(torch.sqrt(torch.mean(difference.square()))),
    )


def conservation(
    initial: torch.Tensor,
    final: torch.Tensor,
    dimension: int,
    cells: int,
) -> tuple[float, float, bool]:
    volume = (1.0 / cells) ** dimension
    change = float(torch.abs(volume * torch.sum(final - initial)))
    bound = float(
        64.0
        * torch.finfo(torch.float64).eps
        * volume
        * torch.sum(torch.abs(initial))
        + 2.0e-15
    )
    return change, bound, change <= bound
