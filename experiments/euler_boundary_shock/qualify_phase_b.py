#!/usr/bin/env python3
"""Record the frozen Phase-B one-dimensional Euler qualification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import gradflow.euler1d as implementation  # noqa: E402
import gradflow.euler3d as shared_implementation  # noqa: E402
from gradflow import (  # noqa: E402
    QUALIFIED_EULER_WENO_ORDERS,
    euler1d_cfl_timestep,
    euler1d_rhs,
    euler1d_rhs_with_boundary_fluxes,
    generate_weno_js_coefficients,
)
from experiments.euler_boundary_shock.sod_exact import (  # noqa: E402
    sample_solution,
    sod_solution,
)
from experiments.euler_boundary_shock.verify_phase_a import (  # noqa: E402
    DEFAULT_RECORD as PHASE_A_RECORD,
    verify as verify_phase_a,
)


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "results" / "phase_b_20260827"
)
REPRESENTATIVE_ORDERS = (5, 11, 15)
PILOT_ORDERS = (7, 9, 13)
REFINEMENT_POINTS = (200, 400, 800)
SMOOTH_POINTS = (24, 36, 54, 81)
ROUND_OFF_FLOOR = 1.0e-11
GAMMA = 1.4
CFL = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def primitive_to_conserved(primitive: torch.Tensor) -> torch.Tensor:
    density, velocity, pressure = primitive
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity.square()
    return torch.stack((density, density * velocity, energy))


def conserved_to_primitive(conserved: torch.Tensor) -> torch.Tensor:
    density = conserved[0]
    velocity = conserved[1] / density
    pressure = (GAMMA - 1.0) * (
        conserved[2] - 0.5 * density * velocity.square()
    )
    return torch.stack((density, velocity, pressure))


def entropy_wave(points: int, *, dtype: torch.dtype = torch.float64) -> tuple[
    torch.Tensor, torch.Tensor
]:
    x = (torch.arange(points, dtype=dtype) + 0.5) / points
    density = 1.0 + 0.1 * torch.sin(2.0 * math.pi * x)
    derivative = 0.2 * math.pi * torch.cos(2.0 * math.pi * x)
    velocity = 0.7
    pressure = torch.ones_like(x)
    state = primitive_to_conserved(torch.stack((density, velocity + 0.0 * x, pressure)))
    density_rhs = -velocity * derivative
    exact = torch.stack(
        (
            density_rhs,
            velocity * density_rhs,
            0.5 * velocity**2 * density_rhs,
        )
    )
    return state, exact


def exact_payload_hash() -> str:
    payload = []
    for order in QUALIFIED_EULER_WENO_ORDERS:
        coefficients = generate_weno_js_coefficients(order)
        payload.append(
            {
                "order": order,
                "candidate_offsets": coefficients.candidate_offsets,
                "candidate_coefficients": [
                    [str(value) for value in candidate]
                    for candidate in coefficients.candidate_coefficients
                ],
                "optimal_weights": [
                    str(value) for value in coefficients.optimal_weights
                ],
                "smoothness_matrices": [
                    [[str(value) for value in row] for row in matrix]
                    for matrix in coefficients.smoothness_matrices
                ],
            }
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def local_gates() -> dict[str, Any]:
    uniform: dict[str, Any] = {}
    overlap: dict[str, Any] = {}
    convergence: dict[str, Any] = {}
    conservation: dict[str, Any] = {}
    for order in QUALIFIED_EULER_WENO_ORDERS:
        uniform[str(order)] = {}
        conservation[str(order)] = {}
        for dtype in (torch.float32, torch.float64):
            density = torch.full((max(order, 19),), 1.2, dtype=dtype)
            primitive = torch.stack(
                (
                    density,
                    torch.full_like(density, 0.3),
                    torch.full_like(density, 0.9),
                )
            )
            state = primitive_to_conserved(primitive)
            for boundary in ("periodic", "transmissive"):
                error = float(
                    torch.max(
                        torch.abs(
                            euler1d_rhs(
                                state,
                                0.05,
                                order=order,
                                boundary=boundary,
                            )
                        )
                    )
                )
                uniform[str(order)][f"{str(dtype)}_{boundary}"] = error

        points = max(order + 4, 21)
        state, _ = entropy_wave(points)
        duplicated = torch.cat((state, state[:, :1]), dim=-1)
        scheme = shared_implementation._EULER_WENO_SCHEMES[order]
        expected = shared_implementation._generated_line_rhs(
            duplicated, float(points), scheme
        )[:, :-1]
        actual = euler1d_rhs(
            state, 1.0 / points, order=order, boundary="periodic"
        )
        overlap[str(order)] = float(torch.max(torch.abs(actual - expected)))

        errors = []
        for points in SMOOTH_POINTS:
            state, exact = entropy_wave(points)
            actual = euler1d_rhs(
                state, 1.0 / points, order=order, boundary="periodic"
            )
            errors.append(float(torch.sqrt(torch.mean((actual - exact).square()))))
        rates = [
            math.log(coarse / fine) / math.log(fine_n / coarse_n)
            for coarse, fine, coarse_n, fine_n in zip(
                errors, errors[1:], SMOOTH_POINTS, SMOOTH_POINTS[1:]
            )
        ]
        observable = [
            rate
            for rate, coarse, fine in zip(rates, errors, errors[1:])
            if coarse > ROUND_OFF_FLOOR and fine > ROUND_OFF_FLOOR
        ]
        convergence[str(order)] = {
            "points": SMOOTH_POINTS,
            "l2_errors": errors,
            "rates": rates,
            "observable_rates": observable,
            "floor_limited": not observable,
        }

        points = 43
        x = (torch.arange(points, dtype=torch.float64) + 0.5) / points
        primitive = torch.stack(
            (
                1.1 + 0.07 * torch.sin(2.0 * math.pi * x),
                0.25 + 0.03 * torch.cos(2.0 * math.pi * x),
                0.9 + 0.04 * torch.sin(4.0 * math.pi * x),
            )
        )
        state = primitive_to_conserved(primitive)
        for boundary in ("periodic", "transmissive"):
            dx = 1.0 / points
            rhs, fluxes = euler1d_rhs_with_boundary_fluxes(
                state, dx, order=order, boundary=boundary
            )
            residual = torch.abs(
                dx * torch.sum(rhs, dim=-1) + fluxes[:, 1] - fluxes[:, 0]
            )
            scale = torch.finfo(state.dtype).eps * torch.clamp_min(
                dx * torch.sum(torch.abs(rhs), dim=-1)
                + torch.abs(fluxes[:, 0])
                + torch.abs(fluxes[:, 1]),
                1.0,
            )
            conservation[str(order)][boundary] = float(torch.max(residual / scale))

    return {
        "uniform": uniform,
        "periodic_overlap": overlap,
        "smooth_convergence": convergence,
        "conservation": conservation,
    }


def autograd_gates() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for order in REPRESENTATIVE_ORDERS:
        points = 19
        x = (torch.arange(points, dtype=torch.float64) + 0.5) / points
        base = primitive_to_conserved(
            torch.stack(
                (
                    1.1 + 0.05 * torch.sin(1.3 * math.pi * x),
                    0.2 + 0.02 * x,
                    1.0 + 0.03 * torch.cos(0.7 * math.pi * x),
                )
            )
        )
        direction = torch.sin(
            torch.arange(base.numel(), dtype=base.dtype).reshape_as(base) + 0.3
        )
        direction /= torch.linalg.vector_norm(direction)
        weights = torch.linspace(0.3, 1.7, points, dtype=base.dtype)
        component_weights = torch.tensor([1.0, -0.2, 0.1])[:, None]

        def objective(values: torch.Tensor) -> torch.Tensor:
            dt = 2.0e-4
            rhs0 = euler1d_rhs(
                values, 1.0 / points, order=order, boundary="transmissive"
            )
            first = values + dt * rhs0
            rhs1 = euler1d_rhs(
                first, 1.0 / points, order=order, boundary="transmissive"
            )
            second = 0.75 * values + 0.25 * (first + dt * rhs1)
            rhs2 = euler1d_rhs(
                second, 1.0 / points, order=order, boundary="transmissive"
            )
            advanced = (values + 2.0 * (second + dt * rhs2)) / 3.0
            return torch.sum(advanced * component_weights * weights)

        state = base.detach().requires_grad_(True)
        gradient = torch.autograd.grad(objective(state), state)[0]
        actual = torch.sum(gradient * direction)
        epsilon = 1.0e-6
        expected = (
            objective(base + epsilon * direction)
            - objective(base - epsilon * direction)
        ) / (2.0 * epsilon)
        absolute = float(torch.abs(actual - expected))
        relative = absolute / max(float(torch.abs(expected)), 1.0e-30)
        results[str(order)] = {
            "autograd_directional": float(actual),
            "centered_finite_difference": float(expected),
            "absolute_error": absolute,
            "relative_error": relative,
            "finite": bool(torch.isfinite(gradient).all()),
        }
    return results


def compiler_gates() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for order in REPRESENTATIVE_ORDERS:
        results[str(order)] = {}
        for boundary in ("periodic", "transmissive"):
            state, _ = entropy_wave(max(order + 4, 21))

            def rhs(values: torch.Tensor) -> torch.Tensor:
                return euler1d_rhs(
                    values,
                    1.0 / values.shape[-1],
                    order=order,
                    boundary=boundary,
                )

            eager = rhs(state)
            compiled = torch.compile(rhs, fullgraph=True)(state)
            explanation = torch._dynamo.explain(rhs)(state)
            results[str(order)][boundary] = {
                "maximum_absolute_difference": float(
                    torch.max(torch.abs(eager - compiled))
                ),
                "graph_count": explanation.graph_count,
                "graph_break_count": explanation.graph_break_count,
            }
    return results


def device_gates() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA is unavailable"}
    results: dict[str, Any] = {"available": True, "cases": {}}
    for order in QUALIFIED_EULER_WENO_ORDERS:
        for dtype in (torch.float32, torch.float64):
            state, _ = entropy_wave(37, dtype=dtype)
            for boundary in ("periodic", "transmissive"):
                cpu = euler1d_rhs(
                    state, 1.0 / 37.0, order=order, boundary=boundary
                )
                cuda = euler1d_rhs(
                    state.cuda(), 1.0 / 37.0, order=order, boundary=boundary
                ).cpu()
                key = f"order{order}_{str(dtype)}_{boundary}"
                results["cases"][key] = float(torch.max(torch.abs(cpu - cuda)))
    return results


def static_gate() -> dict[str, Any]:
    source = inspect.getsource(implementation)
    shared_source = inspect.getsource(shared_implementation._generated_bounded_line_rhs)
    forbidden = (
        ".cpu(",
        ".cuda(",
        ".to(",
        ".item(",
        ".numpy(",
        "torch.library",
        "triton",
    )
    found = [token for token in forbidden if token in source or token in shared_source]
    return {"passed": not found, "forbidden_tokens_found": found}


def initial_state(problem: str, points: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    if problem == "sod":
        left, right, interface = 0.0, 1.0, 0.5
        x = left + (torch.arange(points, dtype=torch.float64) + 0.5) / points
        left_state = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)[:, None]
        right_state = torch.tensor([0.125, 0.0, 0.1], dtype=torch.float64)[:, None]
        primitive = torch.where(x[None, :] < interface, left_state, right_state)
    elif problem == "shu_osher":
        left, right, interface = -5.0, 5.0, -4.0
        dx = (right - left) / points
        x = left + (torch.arange(points, dtype=torch.float64) + 0.5) * dx
        left_state = torch.tensor(
            [3.857143, 2.629369, 10.33333], dtype=torch.float64
        )[:, None]
        right_state = torch.stack(
            (1.0 + 0.2 * torch.sin(5.0 * x), torch.zeros_like(x), torch.ones_like(x))
        )
        primitive = torch.where(x[None, :] < interface, left_state, right_state)
    else:
        raise ValueError(f"unknown problem {problem}")
    return x, primitive_to_conserved(primitive), (right - left) / points


def physical_minima(state: torch.Tensor) -> tuple[float, float, bool]:
    primitive = conserved_to_primitive(state)
    finite = bool(torch.isfinite(primitive).all())
    return float(torch.min(primitive[0])), float(torch.min(primitive[2])), finite


def run_shock(problem: str, order: int, points: int) -> dict[str, Any]:
    final_time = 0.2 if problem == "sod" else 1.8
    x, state, dx = initial_state(problem, points)
    time = 0.0
    steps = 0
    minimum_density = math.inf
    minimum_pressure = math.inf
    failure_stage: str | None = None
    with torch.no_grad():
        while time < final_time:
            dt = min(float(euler1d_cfl_timestep(state, dx, CFL)), final_time - time)
            rhs0 = euler1d_rhs(
                state, dx, order=order, boundary="transmissive"
            )
            first = state + dt * rhs0
            first_density, first_pressure, first_finite = physical_minima(first)
            minimum_density = min(minimum_density, first_density)
            minimum_pressure = min(minimum_pressure, first_pressure)
            if not first_finite or first_density <= 0.0 or first_pressure <= 0.0:
                failure_stage = "ssp_rk3_stage_1"
                state = first
                break

            rhs1 = euler1d_rhs(
                first, dx, order=order, boundary="transmissive"
            )
            second = 0.75 * state + 0.25 * (first + dt * rhs1)
            second_density, second_pressure, second_finite = physical_minima(second)
            minimum_density = min(minimum_density, second_density)
            minimum_pressure = min(minimum_pressure, second_pressure)
            if not second_finite or second_density <= 0.0 or second_pressure <= 0.0:
                failure_stage = "ssp_rk3_stage_2"
                state = second
                break

            rhs2 = euler1d_rhs(
                second, dx, order=order, boundary="transmissive"
            )
            advanced = (state + 2.0 * (second + dt * rhs2)) / 3.0
            final_density, final_pressure, final_finite = physical_minima(advanced)
            minimum_density = min(minimum_density, final_density)
            minimum_pressure = min(minimum_pressure, final_pressure)
            state = advanced
            if not final_finite or final_density <= 0.0 or final_pressure <= 0.0:
                failure_stage = "ssp_rk3_stage_3"
                break
            time += dt
            steps += 1
            if steps > 1_000_000:
                raise RuntimeError("shock qualification step guard exceeded")
    primitive = conserved_to_primitive(state)
    return {
        "problem": problem,
        "order": order,
        "points": points,
        "x": x.numpy(),
        "primitive": primitive.numpy(),
        "conserved": state.numpy(),
        "completed": failure_stage is None and time >= final_time,
        "failure_stage": failure_stage,
        "steps": steps,
        "simulated_time": time,
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
    }


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual - expected)
    names = ("density", "velocity", "pressure")
    return {
        "l1": {name: float(np.mean(difference[index])) for index, name in enumerate(names)},
        "linf": {name: float(np.max(difference[index])) for index, name in enumerate(names)},
    }


def sod_metrics(run: dict[str, Any]) -> dict[str, Any]:
    exact = sample_solution(
        sod_solution(), run["x"], time=0.2, interface=0.5
    )
    metrics = error_metrics(run["primitive"], exact)
    exact_energy = exact[2] / (GAMMA - 1.0) + 0.5 * exact[0] * exact[1] ** 2
    metrics["energy_l1"] = float(np.mean(np.abs(run["conserved"][2] - exact_energy)))
    solution = sod_solution()
    locations = {
        "contact": 0.5 + solution.star_velocity * 0.2,
        "shock": 0.5 + solution.right_head_speed * 0.2,
    }
    midpoints = 0.5 * (run["x"][:-1] + run["x"][1:])
    jumps = np.abs(np.diff(run["primitive"][0]))
    detected: dict[str, Any] = {}
    dx = 1.0 / run["points"]
    for name, exact_location in locations.items():
        mask = np.abs(midpoints - exact_location) <= 0.05
        local_indices = np.flatnonzero(mask)
        index = local_indices[int(np.argmax(jumps[mask]))]
        location = float(midpoints[index])
        detected[name] = {
            "exact": exact_location,
            "detected": location,
            "error_cells": abs(location - exact_location) / dx,
        }
    metrics["wave_locations"] = detected
    return metrics


def shu_osher_metrics(
    run: dict[str, Any], reference_x: np.ndarray, reference: np.ndarray
) -> dict[str, Any]:
    expected = np.stack(
        [np.interp(run["x"], reference_x, reference[index]) for index in range(3)]
    )
    metrics = error_metrics(run["primitive"], expected)
    window = (run["x"] >= -3.0) & (run["x"] <= 3.0)
    actual_density = run["primitive"][0, window]
    expected_density = expected[0, window]
    actual_centered = actual_density - np.mean(actual_density)
    expected_centered = expected_density - np.mean(expected_density)
    correlation = np.dot(actual_centered, expected_centered) / math.sqrt(
        float(np.dot(actual_centered, actual_centered))
        * float(np.dot(expected_centered, expected_centered))
    )
    actual_tv = np.sum(np.abs(np.diff(actual_density)))
    expected_tv = np.sum(np.abs(np.diff(expected_density)))
    metrics["structure"] = {
        "window": [-3.0, 3.0],
        "density_correlation": float(correlation),
        "density_total_variation_ratio": float(actual_tv / expected_tv),
    }
    return metrics


def shock_study(output_directory: Path) -> tuple[dict[str, Any], bool]:
    thresholds = json.loads((PHASE_A_RECORD / "thresholds.json").read_text())
    with np.load(
        PHASE_A_RECORD / "shu_osher_fv_wenoz_hllc_t1p8_n12800.npz"
    ) as archive:
        reference_x = archive["x"].copy()
        reference = archive["primitive"].copy()

    cases: dict[str, Any] = {}
    raw_runs: dict[tuple[str, int, int], dict[str, Any]] = {}
    for order in REPRESENTATIVE_ORDERS:
        for problem in ("sod", "shu_osher"):
            for points in REFINEMENT_POINTS:
                print(f"phase_b {problem} order={order} points={points}", flush=True)
                run = run_shock(problem, order, points)
                raw_runs[(problem, order, points)] = run
                metrics = (
                    sod_metrics(run)
                    if problem == "sod" and run["completed"]
                    else shu_osher_metrics(run, reference_x, reference)
                    if run["completed"]
                    else None
                )
                key = f"{problem}_order{order}_n{points}"
                cases[key] = {
                    name: value
                    for name, value in run.items()
                    if name not in {"x", "primitive", "conserved"}
                }
                cases[key]["metrics"] = metrics
                if points == 800:
                    np.savez_compressed(
                        output_directory / f"{key}.npz",
                        x=run["x"],
                        primitive=run["primitive"],
                        conserved=run["conserved"],
                    )

    for order in PILOT_ORDERS:
        for problem in ("sod", "shu_osher"):
            print(f"phase_b {problem} order={order} points=200 pilot", flush=True)
            run = run_shock(problem, order, 200)
            raw_runs[(problem, order, 200)] = run
            metrics = (
                sod_metrics(run)
                if problem == "sod" and run["completed"]
                else shu_osher_metrics(run, reference_x, reference)
                if run["completed"]
                else None
            )
            key = f"{problem}_order{order}_n200_pilot"
            cases[key] = {
                name: value
                for name, value in run.items()
                if name not in {"x", "primitive", "conserved"}
            }
            cases[key]["metrics"] = metrics

    decisions: dict[str, Any] = {}
    passed = True
    names = ("density", "velocity", "pressure")
    for order in REPRESENTATIVE_ORDERS:
        sod_runs = [raw_runs[("sod", order, points)] for points in REFINEMENT_POINTS]
        shu_runs = [
            raw_runs[("shu_osher", order, points)] for points in REFINEMENT_POINTS
        ]
        completed = all(run["completed"] for run in sod_runs + shu_runs)
        decision: dict[str, Any] = {"completed": completed}
        if completed:
            sod_values = [sod_metrics(run) for run in sod_runs]
            shu_values = [shu_osher_metrics(run, reference_x, reference) for run in shu_runs]
            sod_decreases = {
                name: all(
                    fine["l1"][name] < coarse["l1"][name]
                    for coarse, fine in zip(sod_values, sod_values[1:])
                )
                for name in names
            }
            energy_decreases = all(
                fine["energy_l1"] < coarse["energy_l1"]
                for coarse, fine in zip(sod_values, sod_values[1:])
            )
            sod_threshold = all(
                sod_values[-1]["l1"][name] <= thresholds["sod"]["l1_max"][name]
                for name in names
            )
            sod_ratio = all(
                sod_values[-1]["l1"][name] / sod_values[0]["l1"][name]
                <= thresholds["sod"]["finest_to_coarsest_error_ratio_max"]
                for name in names
            )
            wave_locations = all(
                value["error_cells"]
                <= thresholds["sod"]["wave_location_error_cells_max"]
                for value in sod_values[-1]["wave_locations"].values()
            )
            shu_threshold = all(
                shu_values[-1]["l1"][name]
                <= thresholds["shu_osher"]["l1_max_to_n12800"][name]
                for name in names
            )
            shu_ratio = (
                shu_values[-1]["l1"]["density"]
                / shu_values[0]["l1"]["density"]
                <= thresholds["shu_osher"][
                    "finest_to_coarsest_density_error_ratio_max"
                ]
            )
            structure = shu_values[-1]["structure"]
            shu_structure = (
                structure["density_correlation"]
                >= thresholds["shu_osher"]["density_correlation_min"]
                and structure["density_total_variation_ratio"]
                >= thresholds["shu_osher"]["density_total_variation_ratio_min"]
                and structure["density_total_variation_ratio"]
                <= thresholds["shu_osher"]["density_total_variation_ratio_max"]
            )
            decision.update(
                {
                    "sod_each_variable_decreases": sod_decreases,
                    "sod_energy_decreases": energy_decreases,
                    "sod_finest_thresholds": sod_threshold,
                    "sod_refinement_ratio": sod_ratio,
                    "sod_wave_locations": wave_locations,
                    "shu_osher_finest_thresholds": shu_threshold,
                    "shu_osher_density_refinement_ratio": shu_ratio,
                    "shu_osher_structure": shu_structure,
                }
            )
            order_passed = (
                all(sod_decreases.values())
                and energy_decreases
                and sod_threshold
                and sod_ratio
                and wave_locations
                and shu_threshold
                and shu_ratio
                and shu_structure
            )
        else:
            order_passed = False
        decision["passed"] = order_passed
        decisions[str(order)] = decision
        passed = passed and order_passed

    pilot_decisions = {
        str(order): {
            problem: raw_runs[(problem, order, 200)]["completed"]
            for problem in ("sod", "shu_osher")
        }
        for order in PILOT_ORDERS
    }
    return {
        "cases": cases,
        "representative_order_decisions": decisions,
        "pilot_order_admissibility": pilot_decisions,
        "passed": passed,
    }, passed


def evaluate_local_decision(
    local: dict[str, Any],
    autograd: dict[str, Any],
    compiler: dict[str, Any],
    device: dict[str, Any],
    static: dict[str, Any],
) -> bool:
    for order in QUALIFIED_EULER_WENO_ORDERS:
        values = local["uniform"][str(order)]
        if any(
            error > (2.0e-5 if "float32" in key else 2.0e-12)
            for key, error in values.items()
        ):
            return False
        if local["periodic_overlap"][str(order)] > 2.0e-12:
            return False
        if any(value > 64.0 for value in local["conservation"][str(order)].values()):
            return False
        convergence = local["smooth_convergence"][str(order)]
        errors = convergence["l2_errors"]
        if any(
            fine >= coarse
            for coarse, fine in zip(errors, errors[1:])
            if coarse > ROUND_OFF_FLOOR and fine > ROUND_OFF_FLOOR
        ):
            return False
        observable = convergence["observable_rates"]
        if observable:
            if max(observable) < order - 2:
                return False
        elif convergence["l2_errors"][0] > ROUND_OFF_FLOOR:
            return False
    for result in autograd.values():
        if (
            not result["finite"]
            or result["relative_error"] > 2.0e-5
            and result["absolute_error"] > 2.0e-7
        ):
            return False
    for order_results in compiler.values():
        for result in order_results.values():
            if (
                result["graph_count"] != 1
                or result["graph_break_count"] != 0
                or result["maximum_absolute_difference"] > 2.0e-12
            ):
                return False
    if device["available"]:
        for key, value in device["cases"].items():
            tolerance = 3.0e-4 if "float32" in key else 5.0e-11
            if value > tolerance:
                return False
    return bool(static["passed"])


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def qualify(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase-B qualification requires a clean source worktree")
    phase_a = verify_phase_a(PHASE_A_RECORD)
    thresholds = json.loads((PHASE_A_RECORD / "thresholds.json").read_text())
    source_commit = git("rev-parse", "HEAD")
    torch.set_num_threads(1)

    local = local_gates()
    autograd = autograd_gates()
    compiler = compiler_gates()
    device = device_gates()
    static = static_gate()
    local_passed = evaluate_local_decision(local, autograd, compiler, device, static)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase_b_", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        shocks, shock_passed = shock_study(temporary_path)
        decision = "PASS" if local_passed and shock_passed else "FAIL"
        manifest = {
            "schema": "gradflow.euler_boundary_shock.phase_b.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": source_commit,
            "source_worktree_clean": True,
            "source_hashes": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    ROOT / "src/gradflow/euler1d.py",
                    ROOT / "src/gradflow/euler3d.py",
                    ROOT / "src/gradflow/weno_js.py",
                    ROOT / "experiments/euler_boundary_shock/qualify_phase_b.py",
                    ROOT / "docs/EULER_BOUNDARY_SHOCK_PHASE_B_PROTOCOL.md",
                )
            },
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
                "platform": platform.platform(),
                "torch_threads": torch.get_num_threads(),
                "cuda_available": torch.cuda.is_available(),
            },
            "phase_a": {
                "source_commit": phase_a["source_commit"],
                "manifest_sha256": sha256(PHASE_A_RECORD / "manifest.json"),
                "thresholds_sha256": sha256(PHASE_A_RECORD / "thresholds.json"),
                "shu_osher_reference_sha256": sha256(
                    PHASE_A_RECORD
                    / "shu_osher_fv_wenoz_hllc_t1p8_n12800.npz"
                ),
                "thresholds": thresholds,
            },
            "coefficient_payload_sha256": exact_payload_hash(),
            "local_gates": local,
            "autograd": autograd,
            "compiler": compiler,
            "device": device,
            "static_inspection": static,
            "shock_study": shocks,
            "decision": decision,
            "claim_boundary": {
                "performance_measured": False,
                "dveb_modified": False,
                "stabilization_added": False,
                "navier_stokes": False,
                "publication_claim": False,
            },
        }
        write_json(temporary_path / "qualification.json", manifest)
        artifacts = sorted(temporary_path.iterdir())
        lines = [f"{sha256(path)}  {path.name}" for path in artifacts]
        (temporary_path / "SHA256SUMS").write_text("\n".join(lines) + "\n")
        temporary_path.rename(output)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pilot-problem", choices=("sod", "shu_osher"))
    parser.add_argument("--pilot-order", type=int, default=5)
    parser.add_argument("--pilot-points", type=int, default=200)
    arguments = parser.parse_args()
    if arguments.pilot_problem is not None:
        torch.set_num_threads(1)
        run = run_shock(
            arguments.pilot_problem, arguments.pilot_order, arguments.pilot_points
        )
        metrics = (
            sod_metrics(run)
            if arguments.pilot_problem == "sod" and run["completed"]
            else None
        )
        print(
            json.dumps(
                {
                    name: value
                    for name, value in run.items()
                    if name not in {"x", "primitive", "conserved"}
                }
                | {"metrics": metrics},
                indent=2,
                sort_keys=True,
            )
        )
        return
    qualify(arguments.output.resolve())


if __name__ == "__main__":
    main()
