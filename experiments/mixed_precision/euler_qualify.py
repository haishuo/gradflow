#!/usr/bin/env python3
"""Execute the frozen Phase-D Tier-2 characteristic-Euler qualification."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import gradflow.euler1d as euler1d_source  # noqa: E402
import gradflow.euler3d as euler3d_source  # noqa: E402
from gradflow import (  # noqa: E402
    EULER_GAMMA,
    QUALIFIED_EULER_WENO_ORDERS,
    WENOJSPrecisionPolicy,
    euler1d_cfl_timestep,
    euler1d_rhs,
    euler1d_rhs_with_boundary_fluxes,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    euler_weno_rhs,
    periodic_vortex,
)
from experiments.euler_boundary_shock import qualify_phase_b as phase_b  # noqa: E402
from experiments.mixed_precision.benchmark_worker import (  # noqa: E402
    POLICY_MASKS,
    policy_for_name,
)

ORDERS = tuple(QUALIFIED_EULER_WENO_ORDERS)
REPRESENTATIVE_ORDERS = (5, 11, 15)
POLICIES = (
    "all_f64",
    "indicators_f32",
    "weight_formation_f32",
    "indicators_and_weight_formation_f32",
)
PHASE_B_RECORD = (
    ROOT / "experiments/euler_boundary_shock/results/phase_b_20260827"
)
PHASE_A_RECORD = (
    ROOT / "experiments/euler_boundary_shock/results/phase_a_20260827"
)
LOCAL_THRESHOLDS = {
    "tight": {"linf": 1.0e-5, "rms": 1.0e-6},
    "engineering": {"linf": 5.0e-4, "rms": 1.0e-4},
}
TERMINAL_THRESHOLDS = {
    "tight": {"l1": 1.0e-4, "linf": 2.0e-3},
    "engineering": {"l1": 5.0e-4, "linf": 1.0e-2},
}
GAMMA = EULER_GAMMA
CFL = 0.1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def inherited_class(policy: str, order: int) -> str:
    if policy in {
        "indicators_f32",
        "indicators_and_weight_formation_f32",
    } and order == 5:
        return "engineering"
    return "tight"


def component_metrics(
    actual: torch.Tensor,
    oracle: torch.Tensor,
    scales: torch.Tensor,
) -> dict[str, Any]:
    difference = (actual - oracle).reshape(actual.shape[0], -1)
    normalized_scales = torch.clamp_min(
        scales.to(dtype=torch.float64, device=difference.device),
        torch.finfo(torch.float64).tiny,
    )
    linf_absolute = torch.amax(torch.abs(difference), dim=1)
    rms_absolute = torch.sqrt(torch.mean(difference.square(), dim=1))
    linf = linf_absolute / normalized_scales
    rms = rms_absolute / normalized_scales
    return {
        "finite": bool(torch.all(torch.isfinite(actual))),
        "scales": [float(value) for value in normalized_scales],
        "linf_absolute": [float(value) for value in linf_absolute],
        "rms_absolute": [float(value) for value in rms_absolute],
        "linf_normalized": [float(value) for value in linf],
        "rms_normalized": [float(value) for value in rms],
        "maximum_linf_normalized": float(torch.max(linf)),
        "maximum_rms_normalized": float(torch.max(rms)),
    }


def local_passed(record: dict[str, Any]) -> bool:
    threshold = LOCAL_THRESHOLDS[record["inherited_class"]]
    return (
        all(
            case["finite"]
            and case["maximum_linf_normalized"] <= threshold["linf"]
            and case["maximum_rms_normalized"] <= threshold["rms"]
            for case in record["parity_cases"].values()
        )
        and record["uniform_rhs_linf"] <= 2.0e-12
        and all(value <= 64.0 for value in record["conservation_ratios"].values())
    )


def primitive_state(
    points: int, amplitude: float
) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.arange(points, dtype=torch.float64) + 0.5) / points
    density = 1.0 + amplitude * torch.sin(2.0 * math.pi * x)
    velocity = torch.full_like(x, 0.7)
    pressure = torch.ones_like(x)
    state = phase_b.primitive_to_conserved(
        torch.stack((density, velocity, pressure))
    )
    density_scale = 0.7 * 2.0 * math.pi * amplitude
    scales = torch.tensor(
        [density_scale, 0.7 * density_scale, 0.5 * 0.7**2 * density_scale],
        dtype=torch.float64,
    )
    return state, scales


def three_dimensional_entropy(
    intervals: int,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    x = torch.arange(intervals + 1, dtype=torch.float64) / intervals
    density_line = 1.0 + 0.1 * torch.sin(2.0 * math.pi * x)
    shape = (intervals + 1,) * 3
    density = density_line.reshape(1, 1, -1).expand(shape)
    velocity = (0.7, 0.2, -0.1)
    pressure = 1.0
    speed_squared = sum(component**2 for component in velocity)
    state = torch.stack(
        (
            density,
            density * velocity[0],
            density * velocity[1],
            density * velocity[2],
            pressure / (GAMMA - 1.0) + 0.5 * density * speed_squared,
        )
    )
    return state, (1.0 / intervals,) * 3


def local_gates() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for order in ORDERS:
        oracles: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        state, scales = primitive_state(37, 0.1)
        for boundary in ("periodic", "transmissive"):
            oracle = euler1d_rhs(
                state, 1.0 / 37, order=order, boundary=boundary
            )
            oracles[f"entropy_{boundary}"] = (oracle, scales)
        for amplitude in (1.0e-4, 1.0e-6, 1.0e-7):
            state, scales = primitive_state(257, amplitude)
            for boundary in ("periodic", "transmissive"):
                oracle = euler1d_rhs(
                    state, 1.0 / 257, order=order, boundary=boundary
                )
                oracles[f"near_a{amplitude:.0e}_{boundary}"] = (oracle, scales)

        if order in REPRESENTATIVE_ORDERS:
            intervals = max(17, order)
            state3d, spacing3d = three_dimensional_entropy(intervals)
            oracle3d = euler_weno_rhs(state3d, spacing3d, order=order)
            scales3d = torch.amax(
                torch.abs(oracle3d.reshape(oracle3d.shape[0], -1)), dim=1
            )
            vortex, vortex_spacing = periodic_vortex(
                (intervals,) * 3, dtype=torch.float64
            )
            dt = euler_cfl_timestep(vortex, vortex_spacing, 0.1)
            vortex_oracle = euler_ssp_rk3_step(
                vortex, vortex_spacing, dt, order=order
            )
            vortex_scales = torch.clamp_min(
                torch.amax(
                    torch.abs(vortex_oracle.reshape(vortex_oracle.shape[0], -1)),
                    dim=1,
                ),
                1.0,
            )

        for policy_name in POLICIES:
            precision = policy_for_name(policy_name)
            parity_cases: dict[str, Any] = {}
            state, _ = primitive_state(37, 0.1)
            for boundary in ("periodic", "transmissive"):
                actual = euler1d_rhs(
                    state,
                    1.0 / 37,
                    order=order,
                    boundary=boundary,
                    precision=precision,
                )
                oracle, case_scales = oracles[f"entropy_{boundary}"]
                parity_cases[f"entropy_{boundary}"] = component_metrics(
                    actual, oracle, case_scales
                )
            for amplitude in (1.0e-4, 1.0e-6, 1.0e-7):
                state, _ = primitive_state(257, amplitude)
                for boundary in ("periodic", "transmissive"):
                    actual = euler1d_rhs(
                        state,
                        1.0 / 257,
                        order=order,
                        boundary=boundary,
                        precision=precision,
                    )
                    oracle, case_scales = oracles[
                        f"near_a{amplitude:.0e}_{boundary}"
                    ]
                    parity_cases[f"near_a{amplitude:.0e}_{boundary}"] = (
                        component_metrics(actual, oracle, case_scales)
                    )

            uniform_density = torch.full((max(19, order),), 1.2, dtype=torch.float64)
            uniform = phase_b.primitive_to_conserved(
                torch.stack(
                    (
                        uniform_density,
                        torch.full_like(uniform_density, 0.3),
                        torch.full_like(uniform_density, 0.9),
                    )
                )
            )
            uniform_error = max(
                float(
                    torch.max(
                        torch.abs(
                            euler1d_rhs(
                                uniform,
                                0.05,
                                order=order,
                                boundary=boundary,
                                precision=precision,
                            )
                        )
                    )
                )
                for boundary in ("periodic", "transmissive")
            )

            points = 43
            x = (torch.arange(points, dtype=torch.float64) + 0.5) / points
            physical = torch.stack(
                (
                    1.1 + 0.07 * torch.sin(2.0 * math.pi * x),
                    0.25 + 0.03 * torch.cos(2.0 * math.pi * x),
                    0.9 + 0.04 * torch.sin(4.0 * math.pi * x),
                )
            )
            conserved = phase_b.primitive_to_conserved(physical)
            conservation: dict[str, float] = {}
            for boundary in ("periodic", "transmissive"):
                rhs, fluxes = euler1d_rhs_with_boundary_fluxes(
                    conserved,
                    1.0 / points,
                    order=order,
                    boundary=boundary,
                    precision=precision,
                )
                residual = torch.abs(
                    torch.sum(rhs, dim=-1) / points + fluxes[:, 1] - fluxes[:, 0]
                )
                scale = torch.finfo(torch.float64).eps * torch.clamp_min(
                    torch.sum(torch.abs(rhs), dim=-1) / points
                    + torch.abs(fluxes[:, 0])
                    + torch.abs(fluxes[:, 1]),
                    1.0,
                )
                conservation[boundary] = float(torch.max(residual / scale))

            if order in REPRESENTATIVE_ORDERS:
                actual3d = euler_weno_rhs(
                    state3d,
                    spacing3d,
                    order=order,
                    precision=precision,
                )
                parity_cases["euler3d_entropy"] = component_metrics(
                    actual3d, oracle3d, scales3d
                )
                vortex_actual = euler_ssp_rk3_step(
                    vortex,
                    vortex_spacing,
                    dt,
                    order=order,
                    precision=precision,
                )
                parity_cases["euler3d_vortex_step"] = component_metrics(
                    vortex_actual, vortex_oracle, vortex_scales
                )

            record = {
                "order": order,
                "policy": policy_name,
                "mask": POLICY_MASKS[policy_name],
                "assignment": precision.as_names(),
                "inherited_class": inherited_class(policy_name, order),
                "parity_cases": parity_cases,
                "uniform_rhs_linf": uniform_error,
                "conservation_ratios": conservation,
            }
            record["passed"] = local_passed(record)
            records[f"order{order}_{policy_name}"] = record
    return records


def state_minima(state: torch.Tensor) -> tuple[float, float, bool]:
    primitive = phase_b.conserved_to_primitive(state).detach()
    return (
        float(torch.min(primitive[0])),
        float(torch.min(primitive[2])),
        bool(torch.all(torch.isfinite(primitive))),
    )


def advance(
    state: torch.Tensor,
    dx: float,
    dt: float,
    *,
    order: int,
    boundary: str,
    precision: WENOJSPrecisionPolicy,
) -> tuple[torch.Tensor, list[tuple[float, float, bool]]]:
    rhs0 = euler1d_rhs(
        state, dx, order=order, boundary=boundary, precision=precision
    )
    first = state + dt * rhs0
    rhs1 = euler1d_rhs(
        first, dx, order=order, boundary=boundary, precision=precision
    )
    second = 0.75 * state + 0.25 * (first + dt * rhs1)
    rhs2 = euler1d_rhs(
        second, dx, order=order, boundary=boundary, precision=precision
    )
    advanced = (state + 2.0 * (second + dt * rhs2)) / 3.0
    return advanced, [state_minima(value) for value in (first, second, advanced)]


def run_to_time(
    state: torch.Tensor,
    dx: float,
    final_time: float,
    *,
    order: int,
    boundary: str,
    precision: WENOJSPrecisionPolicy,
) -> dict[str, Any]:
    time_value = 0.0
    steps = 0
    minimum_density = math.inf
    minimum_pressure = math.inf
    failure = None
    with torch.no_grad():
        while time_value < final_time:
            dt = min(
                float(euler1d_cfl_timestep(state, dx, CFL)),
                final_time - time_value,
            )
            state, minima = advance(
                state,
                dx,
                dt,
                order=order,
                boundary=boundary,
                precision=precision,
            )
            for density, pressure, finite in minima:
                minimum_density = min(minimum_density, density)
                minimum_pressure = min(minimum_pressure, pressure)
                if not finite or density <= 0.0 or pressure <= 0.0:
                    failure = "nonphysical_ssp_rk3_stage"
                    break
            if failure is not None:
                break
            time_value += dt
            steps += 1
            if steps > 1_000_000:
                raise RuntimeError("Tier-2 integration step guard exceeded")
    return {
        "state": state,
        "completed": failure is None and time_value >= final_time,
        "failure": failure,
        "steps": steps,
        "simulated_time": time_value,
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
    }


def terminal_metrics(
    actual: torch.Tensor,
    oracle: torch.Tensor,
    scales: torch.Tensor,
) -> dict[str, Any]:
    difference = (actual - oracle).reshape(actual.shape[0], -1)
    normalized_scales = torch.clamp_min(scales, torch.finfo(torch.float64).tiny)
    l1 = torch.mean(torch.abs(difference), dim=1) / normalized_scales
    linf = torch.amax(torch.abs(difference), dim=1) / normalized_scales
    return {
        "l1_normalized": [float(value) for value in l1],
        "linf_normalized": [float(value) for value in linf],
        "maximum_l1_normalized": float(torch.max(l1)),
        "maximum_linf_normalized": float(torch.max(linf)),
    }


def repeated_step_gates() -> dict[str, Any]:
    records: dict[str, Any] = {}
    points = 64
    initial, _ = primitive_state(points, 0.1)
    scales = torch.amax(initial, dim=1) - torch.amin(initial, dim=1)
    final_time = 1.0 / 0.7
    for order in REPRESENTATIVE_ORDERS:
        oracle_run = run_to_time(
            initial.clone(),
            1.0 / points,
            final_time,
            order=order,
            boundary="periodic",
            precision=policy_for_name("all_f64"),
        )
        oracle = oracle_run["state"]
        oracle_error = torch.mean(torch.abs(oracle - initial), dim=1)
        for policy_name in POLICIES:
            run = (
                oracle_run
                if policy_name == "all_f64"
                else run_to_time(
                    initial.clone(),
                    1.0 / points,
                    final_time,
                    order=order,
                    boundary="periodic",
                    precision=policy_for_name(policy_name),
                )
            )
            metrics = terminal_metrics(run["state"], oracle, scales)
            analytic_error = torch.mean(torch.abs(run["state"] - initial), dim=1)
            analytic_bound = 1.05 * oracle_error + 64.0 * torch.finfo(
                torch.float32
            ).eps * scales
            class_name = inherited_class(policy_name, order)
            threshold = TERMINAL_THRESHOLDS[class_name]
            passed = (
                run["completed"]
                and run["minimum_density"] > 0.0
                and run["minimum_pressure"] > 0.0
                and metrics["maximum_l1_normalized"] <= threshold["l1"]
                and metrics["maximum_linf_normalized"] <= threshold["linf"]
                and bool(torch.all(analytic_error <= analytic_bound))
            )
            records[f"order{order}_{policy_name}"] = {
                **{name: value for name, value in run.items() if name != "state"},
                "order": order,
                "policy": policy_name,
                "inherited_class": class_name,
                "terminal_parity": metrics,
                "analytic_l1": [float(value) for value in analytic_error],
                "oracle_analytic_l1": [float(value) for value in oracle_error],
                "analytic_bound": [float(value) for value in analytic_bound],
                "passed": passed,
            }
    return records


def shock_gates(output: Path) -> dict[str, Any]:
    thresholds = json.loads((PHASE_A_RECORD / "thresholds.json").read_text())
    with np.load(
        PHASE_A_RECORD / "shu_osher_fv_wenoz_hllc_t1p8_n12800.npz"
    ) as archive:
        reference_x = archive["x"].copy()
        reference = archive["primitive"].copy()
    records: dict[str, Any] = {}
    for order in REPRESENTATIVE_ORDERS:
        for problem in ("sod", "shu_osher"):
            baseline_path = PHASE_B_RECORD / f"{problem}_order{order}_n800.npz"
            with np.load(baseline_path) as archive:
                baseline_primitive = archive["primitive"].copy()
            scales = np.maximum(
                np.ptp(baseline_primitive, axis=1), np.ones(3, dtype=np.float64)
            )
            for policy_name in POLICIES:
                key = f"{problem}_order{order}_{policy_name}"
                if policy_name == "all_f64":
                    records[key] = {
                        "problem": problem,
                        "order": order,
                        "policy": policy_name,
                        "inherited_class": inherited_class(policy_name, order),
                        "control_artifact": str(baseline_path.relative_to(ROOT)),
                        "control_sha256": sha256(baseline_path),
                        "completed": True,
                        "minimum_density": None,
                        "minimum_pressure": None,
                        "terminal_parity": {
                            "maximum_l1_normalized": 0.0,
                            "maximum_linf_normalized": 0.0,
                        },
                        "independent_passed": True,
                        "passed": True,
                    }
                    continue
                print(f"tier2 {problem} order={order} {policy_name}", flush=True)
                x, initial, dx = phase_b.initial_state(problem, 800)
                run = run_to_time(
                    initial,
                    dx,
                    0.2 if problem == "sod" else 1.8,
                    order=order,
                    boundary="transmissive",
                    precision=policy_for_name(policy_name),
                )
                primitive = phase_b.conserved_to_primitive(run["state"])
                difference = np.abs(primitive.numpy() - baseline_primitive)
                terminal = {
                    "l1_normalized": list(np.mean(difference, axis=1) / scales),
                    "linf_normalized": list(np.max(difference, axis=1) / scales),
                    "maximum_l1_normalized": float(
                        np.max(np.mean(difference, axis=1) / scales)
                    ),
                    "maximum_linf_normalized": float(
                        np.max(np.max(difference, axis=1) / scales)
                    ),
                }
                run_for_metrics = {
                    "x": x.numpy(),
                    "primitive": primitive.numpy(),
                    "conserved": run["state"].numpy(),
                    "points": 800,
                }
                if problem == "sod":
                    independent = phase_b.sod_metrics(run_for_metrics)
                    independent_passed = (
                        all(
                            independent["l1"][name]
                            <= thresholds["sod"]["l1_max"][name]
                            for name in ("density", "velocity", "pressure")
                        )
                        and all(
                            value["error_cells"]
                            <= thresholds["sod"]["wave_location_error_cells_max"]
                            for value in independent["wave_locations"].values()
                        )
                    )
                else:
                    independent = phase_b.shu_osher_metrics(
                        run_for_metrics, reference_x, reference
                    )
                    structure = independent["structure"]
                    independent_passed = (
                        all(
                            independent["l1"][name]
                            <= thresholds["shu_osher"]["l1_max_to_n12800"][name]
                            for name in ("density", "velocity", "pressure")
                        )
                        and structure["density_correlation"]
                        >= thresholds["shu_osher"]["density_correlation_min"]
                        and thresholds["shu_osher"][
                            "density_total_variation_ratio_min"
                        ]
                        <= structure["density_total_variation_ratio"]
                        <= thresholds["shu_osher"][
                            "density_total_variation_ratio_max"
                        ]
                    )
                class_name = inherited_class(policy_name, order)
                threshold = TERMINAL_THRESHOLDS[class_name]
                passed = (
                    run["completed"]
                    and run["minimum_density"] > 0.0
                    and run["minimum_pressure"] > 0.0
                    and terminal["maximum_l1_normalized"] <= threshold["l1"]
                    and terminal["maximum_linf_normalized"] <= threshold["linf"]
                    and independent_passed
                )
                np.savez_compressed(
                    output / f"{key}.npz",
                    x=x.numpy(),
                    primitive=primitive.numpy(),
                    conserved=run["state"].numpy(),
                )
                records[key] = {
                    **{name: value for name, value in run.items() if name != "state"},
                    "problem": problem,
                    "order": order,
                    "policy": policy_name,
                    "inherited_class": class_name,
                    "terminal_parity": terminal,
                    "independent_metrics": independent,
                    "independent_passed": independent_passed,
                    "passed": passed,
                }
    return records


def gradient_gates() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for order in REPRESENTATIVE_ORDERS:
        points = 19
        x = (torch.arange(points, dtype=torch.float64) + 0.5) / points
        base = phase_b.primitive_to_conserved(
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

        def objective(values: torch.Tensor, precision: WENOJSPrecisionPolicy):
            advanced, _ = advance(
                values,
                1.0 / points,
                2.0e-4,
                order=order,
                boundary="transmissive",
                precision=precision,
            )
            return torch.sum(advanced * component_weights * weights)

        reference_state = base.detach().requires_grad_(True)
        reference_objective = objective(
            reference_state, policy_for_name("all_f64")
        )
        reference_gradient = torch.autograd.grad(
            reference_objective, reference_state
        )[0]
        gradient_scale = float(torch.linalg.vector_norm(reference_gradient))
        for policy_name in POLICIES:
            precision = policy_for_name(policy_name)
            state = base.detach().requires_grad_(True)
            value = objective(state, precision)
            gradient = torch.autograd.grad(value, state)[0]
            directional = torch.sum(gradient * direction)
            epsilon = 1.0e-6
            finite_difference = (
                objective(base + epsilon * direction, precision)
                - objective(base - epsilon * direction, precision)
            ) / (2.0 * epsilon)
            absolute = float(torch.abs(directional - finite_difference))
            relative = absolute / max(float(torch.abs(finite_difference)), 1.0e-30)
            difference = gradient - reference_gradient
            l2 = float(torch.linalg.vector_norm(difference)) / max(
                gradient_scale, 1.0e-30
            )
            linf = float(torch.max(torch.abs(difference))) / max(
                float(torch.max(torch.abs(reference_gradient))), 1.0e-30
            )
            passed = (
                bool(torch.all(torch.isfinite(gradient)))
                and bool(torch.count_nonzero(gradient))
                and (relative <= 2.0e-5 or absolute <= 2.0e-7)
                and l2 <= 5.0e-4
                and linf <= 2.0e-3
            )
            records[f"order{order}_{policy_name}"] = {
                "order": order,
                "policy": policy_name,
                "directional_derivative": float(directional),
                "centered_finite_difference": float(finite_difference),
                "absolute_error": absolute,
                "relative_error": relative,
                "gradient_l2_normalized": l2,
                "gradient_linf_normalized": linf,
                "finite": bool(torch.all(torch.isfinite(gradient))),
                "nonzero": bool(torch.count_nonzero(gradient)),
                "passed": passed,
            }
    return records


def compiler_and_device_gates() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for order in REPRESENTATIVE_ORDERS:
        state, _ = primitive_state(max(21, order + 4), 0.1)
        for policy_name in POLICIES:
            precision = policy_for_name(policy_name)
            cpu = euler1d_rhs(
                state,
                1.0 / state.shape[-1],
                order=order,
                boundary="transmissive",
                precision=precision,
            )
            scales = torch.amax(torch.abs(cpu), dim=1)
            device_metric = None
            if torch.cuda.is_available():
                cuda = euler1d_rhs(
                    state.cuda(),
                    1.0 / state.shape[-1],
                    order=order,
                    boundary="transmissive",
                    precision=precision,
                ).cpu()
                device_metric = component_metrics(cuda, cpu, scales)
            compiled_records: dict[str, Any] = {}
            for device in ("cpu", "cuda"):
                if device == "cuda" and not torch.cuda.is_available():
                    compiled_records[device] = {
                        "available": False,
                        "reason": "CUDA unavailable",
                    }
                    continue
                values = state if device == "cpu" else state.cuda()

                def call(input_state: torch.Tensor) -> torch.Tensor:
                    return euler1d_rhs(
                        input_state,
                        1.0 / input_state.shape[-1],
                        order=order,
                        boundary="transmissive",
                        precision=precision,
                    )

                eager = call(values)
                explanation = torch._dynamo.explain(call)(values)
                compiled = torch.compile(call, fullgraph=True, dynamic=False)
                actual = compiled(values)
                parity = component_metrics(
                    actual,
                    eager,
                    torch.amax(
                        torch.abs(eager.reshape(eager.shape[0], -1)), dim=1
                    ),
                )
                compiled_records[device] = {
                    "available": True,
                    "graph_count": explanation.graph_count,
                    "graph_break_count": explanation.graph_break_count,
                    "parity": parity,
                    "passed": (
                        explanation.graph_count == 1
                        and explanation.graph_break_count == 0
                        and parity["maximum_linf_normalized"] <= 5.0e-5
                        and parity["maximum_rms_normalized"] <= 1.0e-5
                    ),
                }
            device_passed = (
                device_metric is None
                or device_metric["maximum_linf_normalized"] <= 5.0e-4
            )
            records[f"order{order}_{policy_name}"] = {
                "order": order,
                "policy": policy_name,
                "cpu_cuda": device_metric,
                "cpu_cuda_passed": device_passed,
                "compiled": compiled_records,
                "passed": device_passed
                and all(
                    result.get("passed", False)
                    for result in compiled_records.values()
                    if result["available"]
                ),
            }
    return records


def static_gate() -> dict[str, Any]:
    sources = inspect.getsource(euler1d_source) + inspect.getsource(euler3d_source)
    forbidden = (
        ".cpu(",
        ".cuda(",
        ".to(",
        ".item(",
        ".numpy(",
        "torch.library",
        "triton",
    )
    found = [token for token in forbidden if token in sources]
    return {"passed": not found, "forbidden_tokens_found": found}


def qualify(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if git_text("status", "--porcelain"):
        raise RuntimeError("Tier-2 qualification requires a clean source worktree")
    source_commit = git_text("rev-parse", "HEAD")
    torch.set_num_threads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="phase_d_tier2_", dir=output.parent
    ) as temporary:
        temporary_path = Path(temporary)
        print("tier2 local", flush=True)
        local = local_gates()
        print("tier2 repeated", flush=True)
        repeated = repeated_step_gates()
        print("tier2 gradients", flush=True)
        gradients = gradient_gates()
        print("tier2 compiler_device", flush=True)
        compiler_device = compiler_and_device_gates()
        print("tier2 shocks", flush=True)
        shocks = shock_gates(temporary_path)
        static = static_gate()
        passed = (
            all(record["passed"] for record in local.values())
            and all(record["passed"] for record in repeated.values())
            and all(record["passed"] for record in gradients.values())
            and all(record["passed"] for record in compiler_device.values())
            and all(record["passed"] for record in shocks.values())
            and static["passed"]
        )
        payload = {
            "schema_version": 1,
            "phase": "D-tier-2-characteristic-euler-mixed-precision",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": "docs/MIXED_PRECISION_PHASE_D_TIER2_PROTOCOL.md",
            "source_commit": source_commit,
            "source_worktree_clean": True,
            "orders": ORDERS,
            "representative_orders": REPRESENTATIVE_ORDERS,
            "policies": POLICIES,
            "local_thresholds": LOCAL_THRESHOLDS,
            "terminal_thresholds": TERMINAL_THRESHOLDS,
            "phase_a_thresholds_sha256": sha256(
                PHASE_A_RECORD / "thresholds.json"
            ),
            "phase_b_qualification_sha256": sha256(
                PHASE_B_RECORD / "qualification.json"
            ),
            "source_hashes": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    ROOT / "src/gradflow/weno_js.py",
                    ROOT / "src/gradflow/euler1d.py",
                    ROOT / "src/gradflow/euler3d.py",
                    Path(__file__),
                    ROOT / "docs/MIXED_PRECISION_PHASE_D_TIER2_PROTOCOL.md",
                )
            },
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_runtime": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "platform": platform.platform(),
                "torch_threads": torch.get_num_threads(),
            },
            "local": local,
            "repeated_step": repeated,
            "gradients": gradients,
            "compiler_device": compiler_device,
            "shocks": shocks,
            "static": static,
            "decision": "PASS" if passed else "FAIL",
            "claim_boundary": {
                "performance_measured": False,
                "production_default_selected": False,
                "euler_specific_precision_searched": False,
                "dveb_modified": False,
                "publication_claim": False,
            },
        }
        result_path = temporary_path / "qualification.json"
        result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        files = sorted(temporary_path.iterdir())
        (temporary_path / "SHA256SUMS").write_text(
            "\n".join(f"{sha256(path)}  {path.name}" for path in files) + "\n"
        )
        temporary_path.rename(output)
    print(f"{payload['decision']} {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.output.resolve())


if __name__ == "__main__":
    main()
