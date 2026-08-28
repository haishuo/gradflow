#!/usr/bin/env python3
"""Execute the frozen correctness-only FD/FV Euler Phase-6B gate."""

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
from typing import Any, Callable

import numpy as np
import torch

from experiments.fd_fv_euler.phase6a_oracle import (
    build_projections,
)
from experiments.fd_fv_euler.phase6b_problem import (
    BOUNDARIES,
    FINAL_SMOOTH_TIME,
    METHODS,
    PROJECTIONS,
    SHOCK_SIZES,
    SIZES,
    conserved_to_primitive,
    error_metrics,
    evolve,
    method_rhs,
    method_rhs_fluxes,
    primitive_to_conserved,
    rates,
    rk_stages,
    shock_expected,
    shock_initial,
    shu_structure,
    smooth_rhs,
    smooth_state,
    sod_wave_metrics,
)
from gradflow import euler1d_fv_rhs, euler1d_rhs
from gradflow.euler3d import _flux_and_roe_faces
from gradflow.weno_js import WENOJS


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_COMMIT = "6662943"
PROTOCOL = ROOT / "docs/FD_FV_PHASE_6B_PROTOCOL.md"
PHASE6A_RESULTS = (
    ROOT / "experiments/fd_fv_euler/results/phase_6a_20260828"
)
PHASE6A_CONTRACT = PHASE6A_RESULTS / "contract.json"
PHASE6A_VERIFY = ROOT / "experiments/fd_fv_euler/verify_phase6a.py"
PHASE_A_THRESHOLDS = (
    ROOT
    / "experiments/euler_boundary_shock/results/phase_a_20260827/thresholds.json"
)
FD_RECORD = (
    ROOT
    / "experiments/euler_boundary_shock/results/phase_b_20260827/qualification.json"
)
FD_VERIFY = ROOT / "experiments/euler_boundary_shock/verify_phase_b.py"
CUDA_RECORD = (
    ROOT
    / "experiments/deferred_cuda_gates/results/qualification_20260828/qualification.json"
)
CUDA_VERIFY = ROOT / "experiments/deferred_cuda_gates/verify.py"
SOURCES = (
    PROTOCOL,
    ROOT / "src/gradflow/euler1d.py",
    ROOT / "src/gradflow/euler1d_fv.py",
    ROOT / "src/gradflow/euler3d.py",
    ROOT / "src/gradflow/weno_js.py",
    ROOT / "experiments/fd_fv_euler/phase6a_oracle.py",
    ROOT / "experiments/fd_fv_euler/phase6b_problem.py",
    Path(__file__).resolve(),
)
SMOOTH_FLOOR = 1.0e-11


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def verify_predecessor(name: str, script: Path, record: Path) -> dict[str, Any]:
    completed = subprocess.run(
        (sys.executable, str(script)),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "name": name,
        "script": str(script.relative_to(ROOT)),
        "record": str(record.relative_to(ROOT)),
        "record_sha256": sha256(record),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0,
    }


def projection_identity() -> dict[str, Any]:
    generated, _ = build_projections()
    cases: dict[str, Any] = {}
    with np.load(PROJECTIONS) as frozen:
        same_keys = set(generated) == set(frozen.files)
        for key in sorted(set(generated) | set(frozen.files)):
            equal = (
                key in generated
                and key in frozen.files
                and np.array_equal(generated[key], frozen[key])
            )
            cases[key] = {"array_equal": equal, "passed": equal}
    return {
        "same_keys": same_keys,
        "cases": cases,
        "passed": same_keys and all(case["passed"] for case in cases.values()),
    }


def error_norms(difference: torch.Tensor) -> dict[str, float]:
    absolute = torch.abs(difference)
    return {
        "l1": float(torch.mean(absolute)),
        "l2": float(torch.sqrt(torch.mean(difference.square()))),
        "linf": float(torch.max(absolute)),
    }


def observable_rates(errors: list[float]) -> list[float]:
    return [
        rate
        for rate, coarse, fine in zip(rates(errors, SIZES), errors, errors[1:])
        if coarse > SMOOTH_FLOOR and fine > SMOOTH_FLOOR
    ]


def uniform_states() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cases: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    primitive = torch.tensor(
        [[1.2], [0.31], [0.93]], dtype=torch.float64
    ).expand(3, 19)
    state = primitive_to_conserved(primitive)
    for method in METHODS:
        for boundary in BOUNDARIES:
            rhs = method_rhs(method, state, 1.0 / 19.0, boundary)
            maximum = float(torch.max(torch.abs(rhs)))
            key = f"{method}_{boundary}"
            arrays[f"uniform_{key}_rhs"] = rhs.numpy()
            cases[key] = {
                "maximum_absolute_rhs": maximum,
                "tolerance": 2.0e-12,
                "passed": maximum <= 2.0e-12,
            }
    return (
        {"cases": cases, "passed": all(x["passed"] for x in cases.values())},
        arrays,
    )


def smooth_spatial_convergence() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    methods: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for method in METHODS:
        records = []
        errors = {name: [] for name in ("l1", "l2", "linf")}
        for cells in SIZES:
            state = smooth_state(method, cells, 0.0)
            expected = smooth_rhs(method, cells, 0.0)
            actual = method_rhs(method, state, 1.0 / cells, "periodic")
            norms = error_norms(actual - expected)
            arrays[f"spatial_{method}_n{cells}_actual"] = actual.numpy()
            arrays[f"spatial_{method}_n{cells}_expected"] = expected.numpy()
            for name, value in norms.items():
                errors[name].append(value)
            records.append({"cells": cells, **norms})
        method_rates = {
            name: rates(values, SIZES) for name, values in errors.items()
        }
        observable = observable_rates(errors["l2"])
        decreasing = all(
            fine < coarse
            for values in errors.values()
            for coarse, fine in zip(values, values[1:])
        )
        passed = decreasing and bool(observable) and max(observable) >= 4.0
        methods[method] = {
            "records": records,
            "rates": method_rates,
            "observable_l2_rates": observable,
            "decreasing": decreasing,
            "passed": passed,
        }
    return (
        {"methods": methods, "passed": all(x["passed"] for x in methods.values())},
        arrays,
    )


def conservation_bound(initial: torch.Tensor, dx: float) -> torch.Tensor:
    return (
        64.0
        * torch.finfo(initial.dtype).eps
        * dx
        * torch.sum(torch.abs(initial), dim=-1)
        + 2.0e-15
    )


def smooth_complete_solve_convergence() -> tuple[
    dict[str, Any], dict[str, np.ndarray]
]:
    methods: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for method in METHODS:
            records = []
            errors = {name: [] for name in ("l1", "l2", "linf")}
            for cells in SIZES:
                dx = 1.0 / cells
                initial = smooth_state(method, cells, 0.0)
                actual, statistics = evolve(
                    method, initial, dx, FINAL_SMOOTH_TIME, "periodic"
                )
                expected = smooth_state(method, cells, FINAL_SMOOTH_TIME)
                arrays[f"solve_{method}_n{cells}_initial"] = initial.numpy()
                arrays[f"solve_{method}_n{cells}_actual"] = actual.numpy()
                arrays[f"solve_{method}_n{cells}_expected"] = expected.numpy()
                norms = error_norms(actual - expected)
                for name, value in norms.items():
                    errors[name].append(value)
                drift = torch.abs(dx * torch.sum(actual - initial, dim=-1))
                single = conservation_bound(initial, dx)
                accumulated = statistics["steps"] * (single - 2.0e-15) + 2.0e-15
                conservation_passed = bool(torch.all(drift <= accumulated))
                records.append(
                    {
                        "cells": cells,
                        **norms,
                        **statistics,
                        "conservation_drift": drift.tolist(),
                        "single_step_roundoff_bound": single.tolist(),
                        "accumulated_roundoff_bound": accumulated.tolist(),
                        "conservation_passed": conservation_passed,
                    }
                )
            method_rates = {
                name: rates(values, SIZES) for name, values in errors.items()
            }
            observable = observable_rates(errors["l2"])
            decreasing = all(
                fine < coarse
                for name in ("l1", "l2")
                for coarse, fine in zip(errors[name], errors[name][1:])
            )
            passed = (
                decreasing
                and bool(observable)
                and max(observable) >= 2.5
                and all(record["completed"] for record in records)
                and all(record["conservation_passed"] for record in records)
            )
            methods[method] = {
                "records": records,
                "rates": method_rates,
                "observable_l2_rates": observable,
                "decreasing_l1_l2": decreasing,
                "passed": passed,
            }
    return (
        {"methods": methods, "passed": all(x["passed"] for x in methods.values())},
        arrays,
    )


def conservation() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cells = 43
    dx = 1.0 / cells
    x = (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    primitive = torch.stack(
        (
            1.1 + 0.07 * torch.sin(2.0 * math.pi * x),
            0.25 + 0.03 * torch.cos(2.0 * math.pi * x),
            0.9 + 0.04 * torch.sin(4.0 * math.pi * x),
        )
    )
    state = primitive_to_conserved(primitive)
    cases: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for method in METHODS:
        for boundary in BOUNDARIES:
            rhs, fluxes = method_rhs_fluxes(method, state, dx, boundary)
            residual = torch.abs(
                dx * torch.sum(rhs, dim=-1) + fluxes[:, 1] - fluxes[:, 0]
            )
            scale = torch.finfo(state.dtype).eps * torch.clamp_min(
                dx * torch.sum(torch.abs(rhs), dim=-1)
                + torch.abs(fluxes[:, 0])
                + torch.abs(fluxes[:, 1]),
                1.0,
            )
            ratio = residual / scale
            maximum = float(torch.max(ratio))
            key = f"{method}_{boundary}"
            arrays[f"conservation_{key}_rhs"] = rhs.numpy()
            arrays[f"conservation_{key}_fluxes"] = fluxes.numpy()
            cases[key] = {
                "residual": residual.tolist(),
                "roundoff_scale": scale.tolist(),
                "roundoff_scaled_ratio": ratio.tolist(),
                "maximum_ratio": maximum,
                "maximum_allowed": 64.0,
                "passed": maximum <= 64.0,
            }
    arrays["conservation_state"] = state.numpy()
    return (
        {"cases": cases, "passed": all(x["passed"] for x in cases.values())},
        arrays,
    )


def differentiation() -> dict[str, Any]:
    cells = 19
    dx = 1.0 / cells
    weights = torch.linspace(0.3, 1.7, cells, dtype=torch.float64)
    components = torch.tensor([1.0, -0.2, 0.1], dtype=torch.float64)[:, None]
    cases: dict[str, Any] = {}
    for method in METHODS:
        base = smooth_state(method, cells, 0.03)
        direction = torch.sin(
            torch.arange(base.numel(), dtype=base.dtype).reshape_as(base) + 0.3
        )
        direction /= torch.linalg.vector_norm(direction)
        for boundary in BOUNDARIES:
            def objective(values: torch.Tensor) -> torch.Tensor:
                advanced = rk_stages(method, values, dx, 2.0e-4, boundary)[-1]
                return torch.sum(advanced * components * weights)

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
            finite = bool(torch.isfinite(gradient).all())
            key = f"{method}_{boundary}"
            cases[key] = {
                "autograd_directional": float(actual),
                "centered_finite_difference": float(expected),
                "absolute_error": absolute,
                "relative_error": relative,
                "finite": finite,
                "passed": finite and (relative <= 2.0e-5 or absolute <= 2.0e-7),
            }
    return {"cases": cases, "passed": all(x["passed"] for x in cases.values())}


def deterministic_state(device: str) -> torch.Tensor:
    return smooth_state("fd", 37, 0.07).to(device=device)


def compile_case(
    name: str,
    call: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    eager = call(state)
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(call)(state)
    torch._dynamo.reset()
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    actual = compiled(state)
    if state.device.type == "cuda":
        torch.cuda.synchronize(state.device)
    difference = float(torch.max(torch.abs(actual - eager)))
    finite = bool(torch.isfinite(actual).all())
    resident = (
        actual.device == state.device
        and actual.dtype == state.dtype
        and actual.shape == state.shape
    )
    passed = (
        explanation.graph_count == 1
        and explanation.graph_break_count == 0
        and difference <= 5.0e-11
        and finite
        and resident
    )
    return (
        {
            "name": name,
            "device": str(state.device),
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "break_reasons": [str(reason) for reason in explanation.break_reasons],
            "compiled_eager_maximum_absolute_difference": difference,
            "tolerance": 5.0e-11,
            "finite": finite,
            "resident": resident,
            "passed": passed,
        },
        eager,
    )


def compiler_and_device() -> dict[str, Any]:
    cpu_cases: dict[str, Any] = {}
    cuda_cases: dict[str, Any] = {}
    agreements: dict[str, Any] = {}
    cpu_state = deterministic_state("cpu")
    for method in METHODS:
        for boundary in BOUNDARIES:
            key = f"{method}_{boundary}"

            def call(values: torch.Tensor) -> torch.Tensor:
                return method_rhs(method, values, 1.0 / 37.0, boundary)

            cpu_cases[key], cpu_eager = compile_case(key, call, cpu_state)
            cuda_state = cpu_state.cuda()
            cuda_cases[key], cuda_eager = compile_case(key, call, cuda_state)
            difference = float(torch.max(torch.abs(cuda_eager.cpu() - cpu_eager)))
            agreements[key] = {
                "maximum_absolute_difference": difference,
                "tolerance": 5.0e-11,
                "finite": bool(torch.isfinite(cuda_eager).all()),
                "resident": cuda_eager.device == cuda_state.device,
                "passed": difference <= 5.0e-11
                and bool(torch.isfinite(cuda_eager).all())
                and cuda_eager.device == cuda_state.device,
            }
    passed = all(
        item["passed"]
        for group in (cpu_cases, cuda_cases, agreements)
        for item in group.values()
    )
    return {
        "cpu_cases": cpu_cases,
        "cuda_cases": cuda_cases,
        "cpu_cuda_agreement": agreements,
        "passed": passed,
    }


def is_movement_event(name: str) -> bool:
    lowered = name.lower()
    return name in {"aten::_to_copy", "aten::copy_"} or any(
        marker in lowered
        for marker in ("memcpy", "host to device", "device to host", "h2d", "d2h")
    )


def profile_movement(method: str, device: str) -> dict[str, Any]:
    state = deterministic_state(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=activities, profile_memory=True, record_shapes=True
    ) as profiler:
        result = method_rhs(method, state, 1.0 / 37.0, "periodic")
        if device == "cuda":
            torch.cuda.synchronize()
    movement = sorted(
        event.key
        for event in profiler.key_averages()
        if is_movement_event(event.key)
    )
    resident = result.device == state.device and result.dtype == state.dtype
    return {
        "method": method,
        "device": device,
        "movement_events": movement,
        "resident": resident,
        "passed": not movement and resident,
    }


def transfer_evidence() -> dict[str, Any]:
    inspected = {
        "src/gradflow/euler1d_fv.py": (
            ROOT / "src/gradflow/euler1d_fv.py"
        ).read_text(),
        "gradflow.euler3d._flux_and_roe_faces": inspect.getsource(
            _flux_and_roe_faces
        ),
        "gradflow.weno_js.WENOJS.reconstruct_stencils": inspect.getsource(
            WENOJS.reconstruct_stencils
        ),
    }
    forbidden = (
        ".cpu(",
        ".cuda(",
        ".item(",
        ".numpy(",
        "triton",
        "torch.library",
        "cpp_extension",
    )
    hits = {
        name: [token for token in forbidden if token in source.lower()]
        for name, source in inspected.items()
    }
    profiles = {
        f"{device}_{method}": profile_movement(method, device)
        for device in ("cpu", "cuda")
        for method in METHODS
    }
    passed = (
        not any(hits.values())
        and all(profile["passed"] for profile in profiles.values())
    )
    return {"static_forbidden_hits": hits, "profiles": profiles, "passed": passed}


def component_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    difference = torch.abs(actual - expected)
    names = ("density", "momentum", "energy")
    return {
        "l1": {name: float(torch.mean(difference[i])) for i, name in enumerate(names)},
        "l2": {
            name: float(torch.sqrt(torch.mean(difference[i].square())))
            for i, name in enumerate(names)
        },
        "linf": {name: float(torch.max(difference[i])) for i, name in enumerate(names)},
    }


def shock_study(thresholds: dict[str, Any]) -> tuple[
    dict[str, Any], dict[str, np.ndarray]
]:
    problems: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for problem in ("sod", "shu_osher"):
            records = []
            for cells in SHOCK_SIZES:
                dx = (1.0 if problem == "sod" else 10.0) / cells
                final_time = 0.2 if problem == "sod" else 1.8
                initial = shock_initial(problem, cells)
                actual, statistics = evolve(
                    "fv", initial, dx, final_time, "transmissive"
                )
                record: dict[str, Any] = {"cells": cells, **statistics}
                if statistics["completed"]:
                    expected_conserved, expected_primitive = shock_expected(
                        problem, cells
                    )
                    actual_primitive = conserved_to_primitive(actual)
                    record["primitive_errors"] = error_metrics(
                        actual_primitive, expected_primitive
                    )
                    record["conserved_errors"] = component_error(
                        actual, expected_conserved
                    )
                    if problem == "sod":
                        record["wave_locations"] = sod_wave_metrics(
                            actual_primitive, cells
                        )
                    else:
                        record["structure"] = shu_structure(
                            actual_primitive, expected_primitive, cells
                        )
                    arrays[f"shock_{problem}_n{cells}_conserved"] = actual.numpy()
                    arrays[f"shock_{problem}_n{cells}_primitive"] = (
                        actual_primitive.numpy()
                    )
                records.append(record)
            limits = thresholds[problem]
            completed = all(record["completed"] for record in records)
            positive = all(
                record["minimum_density"] > 0.0
                and record["minimum_pressure"] > 0.0
                for record in records
            )
            if problem == "sod" and completed:
                error_sequences = {
                    name: [record["primitive_errors"]["l1"][name] for record in records]
                    for name in ("density", "velocity", "pressure")
                }
                decreasing = all(
                    fine < coarse
                    for values in error_sequences.values()
                    for coarse, fine in zip(values, values[1:])
                )
                ratios = {
                    name: values[-1] / values[0]
                    for name, values in error_sequences.items()
                }
                finest = records[-1]
                threshold_pass = all(
                    error_sequences[name][-1] <= limits["l1_max"][name]
                    for name in error_sequences
                )
                ratio_pass = all(
                    value <= limits["finest_to_coarsest_error_ratio_max"]
                    for value in ratios.values()
                )
                location_pass = all(
                    item["error_cells"] <= limits["wave_location_error_cells_max"]
                    for item in finest["wave_locations"].values()
                )
                gates = {
                    "completed": completed,
                    "positive_stages": positive,
                    "monotonic_refinement": decreasing,
                    "finest_l1_thresholds": threshold_pass,
                    "finest_to_coarsest_ratio": ratio_pass,
                    "wave_locations": location_pass,
                }
                diagnostics = {"error_sequences": error_sequences, "ratios": ratios}
            elif problem == "shu_osher" and completed:
                density_errors = [
                    record["primitive_errors"]["l1"]["density"]
                    for record in records
                ]
                finest = records[-1]
                finest_errors = finest["primitive_errors"]["l1"]
                density_ratio = density_errors[-1] / density_errors[0]
                gates = {
                    "completed": completed,
                    "positive_stages": positive,
                    "finest_l1_thresholds": all(
                        finest_errors[name] <= limits["l1_max_to_n12800"][name]
                        for name in ("density", "velocity", "pressure")
                    ),
                    "finest_to_coarsest_density_ratio": density_ratio
                    <= limits["finest_to_coarsest_density_error_ratio_max"],
                    "density_correlation": finest["structure"]["density_correlation"]
                    >= limits["density_correlation_min"],
                    "density_total_variation_ratio": limits[
                        "density_total_variation_ratio_min"
                    ]
                    <= finest["structure"]["density_total_variation_ratio"]
                    <= limits["density_total_variation_ratio_max"],
                }
                diagnostics = {
                    "density_errors": density_errors,
                    "density_finest_to_coarsest_ratio": density_ratio,
                }
            else:
                gates = {"completed": completed, "positive_stages": positive}
                diagnostics = {}
            problems[problem] = {
                "records": records,
                "diagnostics": diagnostics,
                "gate_decisions": gates,
                "passed": all(gates.values()),
            }
    return (
        {
            "problems": problems,
            "passed": all(problem["passed"] for problem in problems.values()),
        },
        arrays,
    )


def environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    query = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=driver_version,uuid",
            "--format=csv,noheader",
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_process_visible": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device": properties.name,
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "cuda_total_memory_bytes": properties.total_memory,
        "cuda_multiprocessor_count": properties.multi_processor_count,
        "cuda_driver_uuid_query": query.stdout.strip(),
        "cuda_driver_query_returncode": query.returncode,
        "mps_tested": False,
    }


def qualify(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6B requires a clean committed source tree")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 6B requires freshly admitted Forge CUDA")
    torch.set_num_threads(1)
    predecessors = {
        "phase_6a": verify_predecessor(
            "phase_6a", PHASE6A_VERIFY, PHASE6A_CONTRACT
        ),
        "fd_phase_b": verify_predecessor("fd_phase_b", FD_VERIFY, FD_RECORD),
        "deferred_cuda": verify_predecessor(
            "deferred_cuda", CUDA_VERIFY, CUDA_RECORD
        ),
    }
    projections = projection_identity()
    uniform, uniform_arrays = uniform_states()
    spatial, spatial_arrays = smooth_spatial_convergence()
    solves, solve_arrays = smooth_complete_solve_convergence()
    conservation_result, conservation_arrays = conservation()
    gradients = differentiation()
    compiler = compiler_and_device()
    transfers = transfer_evidence()
    thresholds = json.loads(PHASE_A_THRESHOLDS.read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="phase6b_", dir=output.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        shocks, shock_arrays = shock_study(thresholds)
        np.savez_compressed(
            temporary / "raw_arrays.npz",
            **uniform_arrays,
            **spatial_arrays,
            **solve_arrays,
            **conservation_arrays,
            **shock_arrays,
        )
        gates = {
            "predecessors": all(x["passed"] for x in predecessors.values()),
            "projection_identity": projections["passed"],
            "uniform_states": uniform["passed"],
            "smooth_spatial_convergence": spatial["passed"],
            "smooth_complete_solve_convergence": solves["passed"],
            "conservation": conservation_result["passed"],
            "differentiation": gradients["passed"],
            "compiler_and_device": compiler["passed"],
            "no_hidden_transfer": transfers["passed"],
            "fv_shocks": shocks["passed"],
            "inherited_fd_shock_decision": json.loads(FD_RECORD.read_text())[
                "decision"
            ]
            == "PASS",
        }
        payload = {
            "schema_version": 1,
            "phase": "fd_fv_euler_phase_6b",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": git("rev-parse", "HEAD"),
            "source_dirty": False,
            "protocol_commit": PROTOCOL_COMMIT,
            "source_hashes": {
                str(path.relative_to(ROOT)): sha256(path) for path in SOURCES
            },
            "authority_hashes": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in (
                    PHASE6A_CONTRACT,
                    PROJECTIONS,
                    PHASE_A_THRESHOLDS,
                    FD_RECORD,
                    CUDA_RECORD,
                )
            },
            "environment": environment(),
            "predecessors": predecessors,
            "projection_identity": projections,
            "uniform_states": uniform,
            "smooth_spatial_convergence": spatial,
            "smooth_complete_solve_convergence": solves,
            "conservation": conservation_result,
            "differentiation": gradients,
            "compiler_and_device": compiler,
            "transfer_evidence": transfers,
            "shock_thresholds": thresholds,
            "shock_study": shocks,
            "gate_decisions": gates,
            "failed_gates": sorted(name for name, passed in gates.items() if not passed),
            "passed": all(gates.values()),
            "performance_measurements_collected": False,
            "phase_6c_begun": False,
            "dveb_modified": False,
            "publication_claim": False,
        }
        write_json(temporary / "qualification.json", payload)
        artifacts = sorted(temporary.iterdir())
        (temporary / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts)
        )
        temporary.rename(output)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    qualify(arguments.output.resolve())


if __name__ == "__main__":
    main()
