"""Frozen smooth inverse-advection problem for Academic A3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from gradflow import WENOJS


ORDER = 11
TRUE_SPEED = 1.1
FINAL_TIME = 0.125
LOWER_SPEED = 0.5
UPPER_SPEED = 1.5
SENSOR_COUNT = 16


def initial_condition(x: torch.Tensor) -> torch.Tensor:
    return (
        0.7
        + 0.2 * torch.sin(x)
        + 0.1 * torch.cos(2.0 * x + 0.3)
        + 0.05 * torch.sin(3.0 * x - 0.2)
    )


@dataclass(frozen=True)
class InverseProblem:
    n: int
    steps: int
    dx: float
    dt: float
    initial_state: torch.Tensor
    sensor_indices: torch.Tensor
    target: torch.Tensor
    objective: Callable[[torch.Tensor], torch.Tensor]


def make_problem(
    n: int,
    *,
    device: torch.device | str = "cpu",
) -> InverseProblem:
    if n not in (64, 128, 256):
        raise ValueError("A3 registers only N=64, 128, or 256")
    if n % SENSOR_COUNT:
        raise ValueError("N must be divisible by the sensor count")
    selected_device = torch.device(device)
    dtype = torch.float64
    x = 2.0 * math.pi * torch.arange(n, dtype=dtype, device=selected_device) / n
    initial = initial_condition(x)
    sensor_indices = torch.arange(
        0, n, n // SENSOR_COUNT, dtype=torch.int64, device=selected_device
    )
    sensor_x = x[sensor_indices]
    target = initial_condition(sensor_x - TRUE_SPEED * FINAL_TIME)
    steps = n // 8
    dx = 2.0 * math.pi / n
    dt = FINAL_TIME / steps
    scheme = WENOJS(ORDER)

    def right_hand_side(state: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        return scheme.rhs(
            state,
            dx,
            lambda values: speed * values,
            alpha=speed,
        )

    def solve(speed: torch.Tensor) -> torch.Tensor:
        state = initial
        for _ in range(steps):
            first = state + dt * right_hand_side(state, speed)
            second = 0.75 * state + 0.25 * (first + dt * right_hand_side(first, speed))
            state = (1.0 / 3.0) * state + (2.0 / 3.0) * (
                second + dt * right_hand_side(second, speed)
            )
        return state

    def objective(speed: torch.Tensor) -> torch.Tensor:
        residual = solve(speed)[sensor_indices] - target
        return 0.5 * torch.mean(residual.square())

    return InverseProblem(
        n=n,
        steps=steps,
        dx=dx,
        dt=dt,
        initial_state=initial,
        sensor_indices=sensor_indices,
        target=target,
        objective=objective,
    )


def evaluate(objective: Callable[[torch.Tensor], torch.Tensor], speed: float) -> float:
    with torch.inference_mode():
        parameter = torch.tensor(speed, dtype=torch.float64)
        return float(objective(parameter))


def golden_section_search(
    objective: Callable[[torch.Tensor], torch.Tensor],
    *,
    iterations: int = 64,
) -> dict[str, object]:
    left = LOWER_SPEED
    right = UPPER_SPEED
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    first = right - inverse_phi * (right - left)
    second = left + inverse_phi * (right - left)
    first_value = evaluate(objective, first)
    second_value = evaluate(objective, second)
    evaluations = 2
    history = []
    for iteration in range(iterations):
        if first_value <= second_value:
            right = second
            second = first
            second_value = first_value
            first = right - inverse_phi * (right - left)
            first_value = evaluate(objective, first)
        else:
            left = first
            first = second
            first_value = second_value
            second = left + inverse_phi * (right - left)
            second_value = evaluate(objective, second)
        evaluations += 1
        history.append(
            {
                "iteration": iteration,
                "left": left,
                "right": right,
                "first": first,
                "second": second,
                "first_objective": first_value,
                "second_objective": second_value,
            }
        )
    speed = 0.5 * (left + right)
    value = evaluate(objective, speed)
    evaluations += 1
    return {
        "speed": speed,
        "objective": value,
        "iterations": iterations,
        "objective_evaluations": evaluations,
        "final_interval": [left, right],
        "history": history,
    }


def autograd_inverse(
    objective: Callable[[torch.Tensor], torch.Tensor],
    *,
    initial_speed: float = 0.8,
) -> dict[str, object]:
    fraction = (initial_speed - LOWER_SPEED) / (UPPER_SPEED - LOWER_SPEED)
    initial_theta = math.log(fraction / (1.0 - fraction))
    theta = torch.tensor(initial_theta, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        (theta,),
        lr=1.0,
        max_iter=40,
        max_eval=60,
        tolerance_grad=1.0e-13,
        tolerance_change=1.0e-15,
        line_search_fn="strong_wolfe",
    )
    history: list[dict[str, float | int]] = []

    def bounded_speed() -> torch.Tensor:
        return LOWER_SPEED + (UPPER_SPEED - LOWER_SPEED) * torch.sigmoid(theta)

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        speed = bounded_speed()
        loss = objective(speed)
        loss.backward()
        assert theta.grad is not None
        history.append(
            {
                "evaluation": len(history),
                "speed": float(speed.detach()),
                "objective": float(loss.detach()),
                "theta_gradient": float(theta.grad.detach()),
            }
        )
        return loss

    optimizer.step(closure)
    final_speed_tensor = bounded_speed()
    final_value = objective(final_speed_tensor)
    final_gradient = torch.autograd.grad(final_value, final_speed_tensor)[0]
    return {
        "speed": float(final_speed_tensor.detach()),
        "objective": float(final_value.detach()),
        "speed_gradient": float(final_gradient.detach()),
        "closure_evaluations": len(history),
        "history": history,
    }
