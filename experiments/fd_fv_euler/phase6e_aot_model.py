"""Ordinary-PyTorch AOT candidates for the frozen Euler Phase 6E study."""

from __future__ import annotations

import torch

from experiments.fd_fv_euler.phase6b_problem import (
    conserved_to_primitive,
    method_cfl,
)
from experiments.fd_fv_euler.phase6c_problem import stage_function


def boundary_and_time(problem: str) -> tuple[str, float, float]:
    if problem == "sod":
        return "transmissive", 1.0, 0.2
    if problem == "shu_osher":
        return "transmissive_shu_osher", 10.0, 1.8
    raise ValueError(f"unknown problem: {problem}")


def stage_diagnostics(
    stages: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    primitive = tuple(conserved_to_primitive(stage) for stage in stages)
    density_minima = torch.stack(tuple(torch.min(value[0]) for value in primitive))
    pressure_minima = torch.stack(tuple(torch.min(value[2]) for value in primitive))
    finite = torch.stack(tuple(torch.isfinite(value).all() for value in primitive))
    return density_minima, pressure_minima, finite


class HostControlledAdvance(torch.nn.Module):
    """One faithful adaptive CFL-plus-SSP-RK3 advance for AOT packaging."""

    def __init__(self, method: str, problem: str, cells: int = 800) -> None:
        super().__init__()
        boundary, length, _ = boundary_and_time(problem)
        self.method = method
        self.dx = length / cells
        self.stages = stage_function(method, cells, boundary)

    def forward(
        self, state: torch.Tensor, remaining: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        dt = torch.minimum(method_cfl(self.method, state, self.dx), remaining)
        stages = self.stages(state, dt)
        density, pressure, finite = stage_diagnostics(stages)
        return stages[-1], dt, density, pressure, finite


class DeviceLoopSolve(torch.nn.Module):
    """Full adaptive solve expressed with structured tensor control flow."""

    def __init__(self, method: str, problem: str, cells: int = 800) -> None:
        super().__init__()
        boundary, length, final_time = boundary_and_time(problem)
        self.method = method
        self.dx = length / cells
        self.final_time = final_time
        self.stages = stage_function(method, cells, boundary)

    def forward(
        self, initial: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        zero = initial.new_zeros(())
        final_time = initial.new_full((), self.final_time)
        step_zero = torch.zeros((), dtype=torch.int64, device=initial.device)
        minimum_density_initial = initial.new_full((), float("inf"))
        minimum_pressure_initial = initial.new_full((), float("inf"))
        failed_zero = torch.zeros((), dtype=torch.bool, device=initial.device)

        def condition(
            state: torch.Tensor,
            time_value: torch.Tensor,
            steps: torch.Tensor,
            minimum_density: torch.Tensor,
            minimum_pressure: torch.Tensor,
            failed: torch.Tensor,
        ) -> torch.Tensor:
            del state, minimum_density, minimum_pressure
            return (time_value < final_time) & (~failed) & (steps < 1_000_000)

        def body(
            state: torch.Tensor,
            time_value: torch.Tensor,
            steps: torch.Tensor,
            minimum_density: torch.Tensor,
            minimum_pressure: torch.Tensor,
            failed: torch.Tensor,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            del failed
            remaining = final_time - time_value
            dt = torch.minimum(method_cfl(self.method, state, self.dx), remaining)
            stages = self.stages(state, dt)
            density, pressure, finite = stage_diagnostics(stages)
            stage_valid = finite & (density > 0.0) & (pressure > 0.0)
            first_failure = ~stage_valid[0]
            second_failure = stage_valid[0] & (~stage_valid[1])
            any_failure = ~torch.all(stage_valid)
            failure_state = torch.where(
                first_failure,
                stages[0],
                torch.where(second_failure, stages[1], stages[2]),
            )
            next_state = torch.where(any_failure, failure_state, stages[2])
            next_time = torch.where(any_failure, time_value, time_value + dt)
            next_steps = torch.where(any_failure, steps, steps + 1)
            return (
                next_state,
                next_time,
                next_steps,
                torch.minimum(minimum_density, torch.min(density)),
                torch.minimum(minimum_pressure, torch.min(pressure)),
                any_failure,
            )

        return torch.while_loop(
            condition,
            body,
            (
                initial,
                zero,
                step_zero,
                minimum_density_initial,
                minimum_pressure_initial,
                failed_zero,
            ),
        )
