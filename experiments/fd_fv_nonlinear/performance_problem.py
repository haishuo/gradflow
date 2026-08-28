"""Frozen Burgers problem helpers shared by Phase-5C timing workers."""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable

import torch

from gradflow import (
    BURGERS_FD_WENO5_FORMULATION_ID,
    BURGERS_FV_WENO5_FORMULATION_ID,
    burgers_fd_weno5_rhs,
    burgers_fv_weno5_rhs,
    ssp_rk3_step,
)
from experiments.fd_fv_nonlinear.burgers_oracle import (
    FINAL_TIME,
    LF_ALPHA,
    projected_state,
)


METHOD_IDS = {
    "fd": BURGERS_FD_WENO5_FORMULATION_ID,
    "fv": BURGERS_FV_WENO5_FORMULATION_ID,
}
METHOD_RHS = {
    "fd": burgers_fd_weno5_rhs,
    "fv": burgers_fv_weno5_rhs,
}


def state(method: str, cells: int, time: float = 0.0) -> torch.Tensor:
    return torch.tensor(
        projected_state(method, cells, time), dtype=torch.float64
    )


def timestep(cells: int) -> tuple[float, int]:
    dx = 1.0 / cells
    nominal = 0.2 * dx ** (5.0 / 3.0) / LF_ALPHA
    steps = math.ceil(FINAL_TIME / nominal)
    return FINAL_TIME / steps, steps


def step_function(
    method: str, cells: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    rhs = METHOD_RHS[method]
    dx = 1.0 / cells
    dt, _ = timestep(cells)

    def step(values: torch.Tensor) -> torch.Tensor:
        return ssp_rk3_step(
            values,
            dt,
            lambda stage: rhs(stage, dx, LF_ALPHA),
        )

    return step


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
    initial: torch.Tensor, final: torch.Tensor, cells: int
) -> tuple[float, float, bool]:
    dx = 1.0 / cells
    change = float(torch.abs(dx * torch.sum(final - initial)))
    bound = float(
        64.0
        * torch.finfo(torch.float64).eps
        * dx
        * torch.sum(torch.abs(initial))
        + 2.0e-15
    )
    return change, bound, change <= bound


def quantile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def statistics_record(samples: list[float]) -> dict[str, Any]:
    valid = all(math.isfinite(value) and value > 0.0 for value in samples)
    if not samples or not valid:
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
