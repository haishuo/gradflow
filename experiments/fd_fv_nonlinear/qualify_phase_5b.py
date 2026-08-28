#!/usr/bin/env python3
"""Execute the frozen correctness-only nonlinear FD/FV Phase-5B gate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from gradflow import (
    BURGERS_FD_WENO5_FORMULATION_ID,
    BURGERS_FV_WENO5_FORMULATION_ID,
    burgers_fd_weno5_rhs,
    burgers_flux,
    burgers_fv_weno5_rhs,
    ssp_rk3_step,
)
from experiments.fd_fv_nonlinear.burgers_oracle import (
    BASE,
    FINAL_TIME,
    LF_ALPHA,
    exact_point,
    exact_spatial_derivative,
    initial_derivative,
    projected_state,
)
from experiments.infrastructure.device_admission import classify_device_admission


PROTOCOL_COMMIT = "0d7b427"
PROTOCOL = ROOT / "docs/FD_FV_PHASE_5B_PROTOCOL.md"
PHASE5A_PROTOCOL = ROOT / "docs/FD_FV_PHASE_5A_PROTOCOL.md"
PHASE5A_ORACLE = ROOT / "experiments/fd_fv_nonlinear/burgers_oracle.py"
PHASE5A_RESULTS = (
    ROOT / "experiments/fd_fv_nonlinear/results/phase_5a_20260828"
)
PHASE5A_CASES = PHASE5A_RESULTS / "oracle_cases.json"
PHASE5A_CONTRACT = PHASE5A_RESULTS / "contract.json"
BURGERS_SOURCE = ROOT / "src/gradflow/burgers.py"
WENO_SOURCE = ROOT / "src/gradflow/weno_js.py"
FV_SOURCE = ROOT / "src/gradflow/fv_weno5.py"
INFRASTRUCTURE = ROOT / "docs/EXECUTION_INFRASTRUCTURE_ADMISSION.md"
QUALIFICATION_SIZES = (24, 36, 54, 81)
FORMULATIONS = {
    "fd": {
        "id": BURGERS_FD_WENO5_FORMULATION_ID,
        "rhs": burgers_fd_weno5_rhs,
        "projection": "fd",
    },
    "fv": {
        "id": BURGERS_FV_WENO5_FORMULATION_ID,
        "rhs": burgers_fv_weno5_rhs,
        "projection": "fv",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def rates(errors: list[float]) -> list[float]:
    return [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors,
            errors[1:],
            QUALIFICATION_SIZES,
            QUALIFICATION_SIZES[1:],
        )
    ]


def verify_phase_5a() -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "experiments/fd_fv_nonlinear/verify_phase_5a.py"),
    ]
    result = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "contract_sha256": sha256(PHASE5A_CONTRACT),
        "oracle_cases_sha256": sha256(PHASE5A_CASES),
        "manifest_sha256": sha256(PHASE5A_RESULTS / "SHA256SUMS"),
        "passed": result.returncode == 0,
    }


def projection_parity() -> dict[str, Any]:
    frozen = json.loads(PHASE5A_CASES.read_text())
    cases: dict[str, Any] = {}
    for cells_text, time_cases in frozen["projections"].items():
        cells = int(cells_text)
        for time_name, case in time_cases.items():
            time = float.fromhex(case["time_hex"])
            for method, field in (
                ("fd", "fd_point_values_hex"),
                ("fv", "fv_cell_averages_hex"),
            ):
                actual = [value.hex() for value in projected_state(method, cells, time)]
                expected = case[field]
                key = f"{method}_n{cells}_{time_name}"
                cases[key] = {
                    "cells": cells,
                    "time_hex": case["time_hex"],
                    "projection": method,
                    "hex_values_equal": actual == expected,
                    "passed": actual == expected,
                }

    tree = ast.parse(PHASE5A_ORACLE.read_text())
    forbidden = {"torch", "numpy", "gradflow"}
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    independence = imported_roots.isdisjoint(forbidden)
    return {
        "cases": cases,
        "oracle_forbidden_imports": sorted(imported_roots & forbidden),
        "oracle_independent": independence,
        "passed": independence and all(case["passed"] for case in cases.values()),
    }


def exact_rhs(method: str, cells: int, time: float = 0.0) -> torch.Tensor:
    dx = 1.0 / cells
    if method == "fd":
        return torch.tensor(
            [
                -exact_point(index * dx, time)
                * exact_spatial_derivative(index * dx, time)
                for index in range(cells)
            ],
            dtype=torch.float64,
        )
    faces = torch.tensor(
        [exact_point(index * dx, time) for index in range(cells + 1)],
        dtype=torch.float64,
    )
    return -(burgers_flux(faces[1:]) - burgers_flux(faces[:-1])) / dx


def conservation_bound(initial: torch.Tensor, cells: int) -> float:
    dx = 1.0 / cells
    return float(
        64.0
        * torch.finfo(torch.float64).eps
        * dx
        * torch.sum(torch.abs(initial))
        + 2.0e-15
    )


def constants_and_conservation() -> dict[str, Any]:
    cases: dict[str, Any] = {}
    cells = 37
    dx = 1.0 / cells
    constant = torch.full((cells,), 0.4, dtype=torch.float64)
    coordinates = torch.arange(cells, dtype=torch.float64) * dx
    nonconstant = 0.5 + 0.1 * torch.sin(2.0 * math.pi * coordinates) + 0.03 * torch.cos(
        6.0 * math.pi * coordinates
    )
    for method, configuration in FORMULATIONS.items():
        rhs = configuration["rhs"]
        constant_rhs = rhs(constant, dx, LF_ALPHA)
        step = ssp_rk3_step(
            constant,
            1.0e-3,
            lambda values, rhs=rhs: rhs(values, dx, LF_ALPHA),
        )
        random_rhs = rhs(nonconstant, dx, LF_ALPHA)
        telescoping = float(torch.abs(dx * torch.sum(random_rhs)))
        bound = conservation_bound(nonconstant, cells)
        maximum_rhs = float(torch.max(torch.abs(constant_rhs)))
        maximum_step = float(torch.max(torch.abs(step - constant)))
        cases[method] = {
            "constant_rhs_maximum_absolute": maximum_rhs,
            "constant_step_maximum_absolute": maximum_step,
            "constant_tolerance": 5.0e-13,
            "nonconstant_rhs_mass_residual": telescoping,
            "conservation_bound": bound,
            "passed": maximum_rhs <= 5.0e-13
            and maximum_step <= 5.0e-13
            and telescoping <= bound,
        }
    return {"cases": cases, "passed": all(case["passed"] for case in cases.values())}


def periodic_distance(left: float, right: float) -> float:
    difference = abs(left - right) % 1.0
    return min(difference, 1.0 - difference)


def spatial_convergence() -> dict[str, Any]:
    methods: dict[str, Any] = {}
    critical_points = ((0.07 + 0.25) % 1.0, (0.07 + 0.75) % 1.0)
    for method, configuration in FORMULATIONS.items():
        rhs = configuration["rhs"]
        projection = configuration["projection"]
        records = []
        l1_errors: list[float] = []
        l2_errors: list[float] = []
        noncritical_errors: list[float] = []
        for cells in QUALIFICATION_SIZES:
            dx = 1.0 / cells
            state = torch.tensor(
                projected_state(projection, cells, 0.0), dtype=torch.float64
            )
            difference = rhs(state, dx, LF_ALPHA) - exact_rhs(method, cells)
            coordinate_offset = 0.0 if method == "fd" else 0.5
            coordinates = [
                (index + coordinate_offset) * dx for index in range(cells)
            ]
            mask = torch.tensor(
                [
                    min(periodic_distance(x, point) for point in critical_points)
                    >= 0.1
                    for x in coordinates
                ],
                dtype=torch.bool,
            )
            l1 = float(torch.mean(torch.abs(difference)))
            l2 = float(torch.sqrt(torch.mean(difference.square())))
            noncritical = float(torch.mean(torch.abs(difference[mask])))
            l1_errors.append(l1)
            l2_errors.append(l2)
            noncritical_errors.append(noncritical)
            records.append(
                {
                    "cells": cells,
                    "l1_error": l1,
                    "l2_error": l2,
                    "noncritical_l1_error": noncritical,
                    "noncritical_count": int(torch.sum(mask)),
                    "finite": bool(torch.isfinite(difference).all()),
                }
            )
        l1_rates = rates(l1_errors)
        l2_rates = rates(l2_errors)
        noncritical_rates = rates(noncritical_errors)
        decreasing = all(
            fine < coarse
            for errors in (l1_errors, l2_errors, noncritical_errors)
            for coarse, fine in zip(errors, errors[1:])
        )
        passed = (
            decreasing
            and all(record["finite"] for record in records)
            and l1_rates[-1] >= 3.0
            and noncritical_rates[-1] >= 4.3
        )
        methods[method] = {
            "records": records,
            "l1_rates": l1_rates,
            "l2_rates": l2_rates,
            "noncritical_l1_rates": noncritical_rates,
            "final_l1_rate_minimum": 3.0,
            "final_noncritical_l1_rate_minimum": 4.3,
            "decreasing": decreasing,
            "passed": passed,
        }
    return {
        "critical_points": list(critical_points),
        "noncritical_exclusion_radius": 0.1,
        "methods": methods,
        "passed": all(method["passed"] for method in methods.values()),
    }


def solve_convergence() -> dict[str, Any]:
    methods: dict[str, Any] = {}
    with torch.no_grad():
        for method, configuration in FORMULATIONS.items():
            rhs = configuration["rhs"]
            projection = configuration["projection"]
            records = []
            l1_errors: list[float] = []
            l2_errors: list[float] = []
            for cells in QUALIFICATION_SIZES:
                dx = 1.0 / cells
                initial = torch.tensor(
                    projected_state(projection, cells, 0.0), dtype=torch.float64
                )
                result = initial
                nominal_dt = 0.2 * dx ** (5.0 / 3.0) / LF_ALPHA
                steps = math.ceil(FINAL_TIME / nominal_dt)
                dt = FINAL_TIME / steps
                for _ in range(steps):
                    result = ssp_rk3_step(
                        result,
                        dt,
                        lambda values, rhs=rhs: rhs(values, dx, LF_ALPHA),
                    )
                expected = torch.tensor(
                    projected_state(projection, cells, FINAL_TIME),
                    dtype=torch.float64,
                )
                difference = result - expected
                l1 = float(torch.mean(torch.abs(difference)))
                l2 = float(torch.sqrt(torch.mean(difference.square())))
                mass_change = float(torch.abs(dx * torch.sum(result - initial)))
                bound = conservation_bound(initial, cells)
                l1_errors.append(l1)
                l2_errors.append(l2)
                records.append(
                    {
                        "cells": cells,
                        "steps": steps,
                        "dt_hex": dt.hex(),
                        "l1_error": l1,
                        "l2_error": l2,
                        "mass_change": mass_change,
                        "conservation_bound": bound,
                        "conservation_passed": mass_change <= bound,
                        "finite": bool(torch.isfinite(result).all()),
                    }
                )
            l1_rates = rates(l1_errors)
            l2_rates = rates(l2_errors)
            decreasing = all(
                fine < coarse
                for errors in (l1_errors, l2_errors)
                for coarse, fine in zip(errors, errors[1:])
            )
            passed = (
                decreasing
                and l1_rates[-1] >= 3.0
                and l1_errors[-1] <= 2.0e-5
                and l2_errors[-1] <= 2.0e-5
                and all(record["finite"] for record in records)
                and all(record["conservation_passed"] for record in records)
            )
            methods[method] = {
                "records": records,
                "l1_rates": l1_rates,
                "l2_rates": l2_rates,
                "final_l1_rate_minimum": 3.0,
                "largest_error_maximum": 2.0e-5,
                "decreasing": decreasing,
                "passed": passed,
            }
    return {"methods": methods, "passed": all(m["passed"] for m in methods.values())}


def differentiation() -> dict[str, Any]:
    cells = 19
    dx = 1.0 / cells
    coordinates = torch.arange(cells, dtype=torch.float64) * dx
    state = 0.5 + 0.1 * torch.sin(2.0 * math.pi * coordinates) + 0.03 * torch.cos(
        4.0 * math.pi * coordinates
    )
    direction = 0.2 * torch.sin(6.0 * math.pi * coordinates) + 0.1 * torch.cos(
        2.0 * math.pi * coordinates
    )
    epsilon = 1.0e-6
    cases: dict[str, Any] = {}
    for method, configuration in FORMULATIONS.items():
        rhs = configuration["rhs"]

        def three_steps(values: torch.Tensor) -> torch.Tensor:
            result = values
            for _ in range(3):
                result = ssp_rk3_step(
                    result,
                    1.0e-3,
                    lambda stage: rhs(stage, dx, LF_ALPHA),
                )
            return result

        _, jvp = torch.autograd.functional.jvp(
            three_steps, (state,), (direction,), create_graph=False
        )
        centered = (
            three_steps(state + epsilon * direction)
            - three_steps(state - epsilon * direction)
        ) / (2.0 * epsilon)
        difference = jvp - centered
        maximum = float(torch.max(torch.abs(difference)))
        relative_l2 = float(
            torch.linalg.vector_norm(difference)
            / torch.clamp_min(
                torch.linalg.vector_norm(centered),
                torch.finfo(torch.float64).tiny,
            )
        )
        finite = bool(torch.isfinite(jvp).all() and torch.isfinite(centered).all())
        cases[method] = {
            "maximum_absolute_difference": maximum,
            "maximum_absolute_tolerance": 3.0e-6,
            "relative_l2_difference": relative_l2,
            "relative_l2_tolerance": 3.0e-5,
            "finite": finite,
            "passed": finite and maximum <= 3.0e-6 and relative_l2 <= 3.0e-5,
        }
    return {"cases": cases, "passed": all(case["passed"] for case in cases.values())}


def deterministic_state(device: torch.device | str) -> torch.Tensor:
    cells = 37
    coordinates = torch.arange(cells, dtype=torch.float64, device=device) / cells
    return 0.5 + 0.1 * torch.sin(2.0 * math.pi * coordinates) + 0.03 * torch.cos(
        6.0 * math.pi * coordinates
    )


def callable_pair(
    rhs: Callable[[torch.Tensor, float, float], torch.Tensor],
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    def rhs_call(values: torch.Tensor) -> torch.Tensor:
        return rhs(values, 1.0 / 37.0, LF_ALPHA)

    def step_call(values: torch.Tensor) -> torch.Tensor:
        return ssp_rk3_step(values, 1.0e-3, rhs_call)

    return rhs_call, step_call


def compile_case(
    name: str,
    call: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
) -> dict[str, Any]:
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
    resident = actual.device == state.device and actual.dtype == state.dtype
    passed = (
        explanation.graph_count == 1
        and explanation.graph_break_count == 0
        and difference <= 2.0e-11
        and finite
        and resident
    )
    return {
        "name": name,
        "device": str(state.device),
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "break_reasons": [str(reason) for reason in explanation.break_reasons],
        "compiled_eager_maximum_absolute_difference": difference,
        "tolerance": 2.0e-11,
        "finite": finite,
        "resident": resident,
        "compilation_executed": True,
        "compilation_duration_measured": False,
        "passed": passed,
    }


def compiler_and_device() -> dict[str, Any]:
    cpu_cases: dict[str, Any] = {}
    cuda_cases: dict[str, Any] = {}
    agreement: dict[str, Any] = {}
    cpu_state = deterministic_state("cpu")
    cuda_visible = torch.cuda.is_available()
    for method, configuration in FORMULATIONS.items():
        calls = callable_pair(configuration["rhs"])
        for call_name, call in zip(("rhs", "step"), calls):
            key = f"{method}_{call_name}"
            cpu_cases[key] = compile_case(key, call, cpu_state)
            if cuda_visible:
                cuda_state = cpu_state.cuda()
                cpu_eager = call(cpu_state)
                cuda_eager = call(cuda_state)
                torch.cuda.synchronize()
                difference = float(
                    torch.max(torch.abs(cuda_eager.cpu() - cpu_eager))
                )
                finite = bool(torch.isfinite(cuda_eager).all())
                resident = cuda_eager.device == cuda_state.device
                agreement[key] = {
                    "maximum_absolute_difference": difference,
                    "tolerance": 2.0e-11,
                    "finite": finite,
                    "resident": resident,
                    "passed": difference <= 2.0e-11 and finite and resident,
                }
                cuda_cases[key] = compile_case(key, call, cuda_state)

    cpu_passed = all(case["passed"] for case in cpu_cases.values())
    if cuda_visible:
        cuda_passed = all(case["passed"] for case in cuda_cases.values()) and all(
            case["passed"] for case in agreement.values()
        )
        cuda_status = classify_device_admission(
            process_visible=True,
            host_inventory="present",
            admission="passed" if cuda_passed else "failed",
        )
    else:
        cuda_passed = False
        cuda_status = classify_device_admission(
            process_visible=False,
            host_inventory="present",
            admission="not_run",
        )
    return {
        "cpu": {"cases": cpu_cases, "passed": cpu_passed},
        "cuda": {
            "process_visible": cuda_visible,
            "host_inventory": "present",
            "status": cuda_status,
            "agreement": agreement,
            "compiler_cases": cuda_cases,
            "passed": cuda_passed,
        },
        "passed": cpu_passed and cuda_status == "admitted",
    }


def is_movement_event(name: str) -> bool:
    lowered = name.lower()
    return name in {"aten::_to_copy", "aten::copy_"} or any(
        marker in lowered
        for marker in (
            "memcpy",
            "host to device",
            "device to host",
            "h2d",
            "d2h",
        )
    )


def profile_movement(
    method: str,
    rhs: Callable[[torch.Tensor, float, float], torch.Tensor],
    device: str,
) -> dict[str, Any]:
    state = deterministic_state(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        result = rhs(state, 1.0 / 37.0, LF_ALPHA)
        if device == "cuda":
            torch.cuda.synchronize()
    averaged = list(profiler.key_averages())
    movement = sorted(
        event.key for event in averaged if is_movement_event(event.key)
    )
    to_events = []
    for event in averaged:
        if event.key == "aten::to":
            to_events.append(
                {
                    "count": event.count,
                    "cpu_memory_usage": getattr(event, "cpu_memory_usage", None),
                    "self_cpu_memory_usage": getattr(
                        event, "self_cpu_memory_usage", None
                    ),
                    "device_memory_usage": getattr(
                        event, "device_memory_usage", None
                    ),
                    "self_device_memory_usage": getattr(
                        event, "self_device_memory_usage", None
                    ),
                }
            )
    zero_memory_to = all(
        all(
            event[field] in (0, None)
            for field in (
                "cpu_memory_usage",
                "self_cpu_memory_usage",
                "device_memory_usage",
                "self_device_memory_usage",
            )
        )
        for event in to_events
    )
    resident = result.device == state.device and result.dtype == state.dtype
    return {
        "method": method,
        "device": device,
        "movement_events": movement,
        "aten_to_events": to_events,
        "aten_to_zero_memory": zero_memory_to,
        "resident": resident,
        "passed": not movement and zero_memory_to and resident,
    }


def transfer_gate() -> dict[str, Any]:
    source = BURGERS_SOURCE.read_text()
    forbidden_tokens = (
        ".cpu(",
        ".cuda(",
        ".to(",
        ".item(",
        ".numpy(",
        "numpy",
        "triton",
        "torch.library",
        "cpp_extension",
    )
    static_hits = [token for token in forbidden_tokens if token in source.lower()]
    profiles: dict[str, Any] = {}
    for method, configuration in FORMULATIONS.items():
        profiles[f"cpu_{method}"] = profile_movement(
            method, configuration["rhs"], "cpu"
        )
        if torch.cuda.is_available():
            profiles[f"cuda_{method}"] = profile_movement(
                method, configuration["rhs"], "cuda"
            )
    expected_profiles = 4 if torch.cuda.is_available() else 2
    passed = (
        not static_hits
        and len(profiles) == expected_profiles
        and all(profile["passed"] for profile in profiles.values())
        and torch.cuda.is_available()
    )
    return {
        "static_forbidden_hits": static_hits,
        "profiles": profiles,
        "passed": passed,
    }


def environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_process_visible": torch.cuda.is_available(),
        "mps_host_inventory": "absent",
        "mps_status": classify_device_admission(
            process_visible=False,
            host_inventory="absent",
            admission="not_run",
        ),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        payload["cuda"] = {
            "device": properties.name,
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        }
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,uuid",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        payload["cuda"]["driver_uuid_query"] = query.stdout.strip()
        payload["cuda"]["driver_query_returncode"] = query.returncode
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    source_commit = git("rev-parse", "HEAD")
    source_dirty = bool(git("status", "--porcelain"))
    if source_dirty:
        raise RuntimeError("Phase 5B qualification requires a clean source tree")
    predecessor = verify_phase_5a()
    projections = projection_parity()
    conservation = constants_and_conservation()
    spatial = spatial_convergence()
    solves = solve_convergence()
    gradients = differentiation()
    compiler = compiler_and_device()
    transfers = transfer_gate()
    gates = {
        "phase_5a_predecessor": predecessor["passed"],
        "projection_oracle": projections["passed"],
        "constant_and_conservation": conservation["passed"],
        "spatial_convergence": spatial["passed"],
        "complete_solve_convergence": solves["passed"],
        "differentiation": gradients["passed"],
        "cpu_cuda_compiler": compiler["passed"],
        "no_hidden_transfer": transfers["passed"],
    }
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_nonlinear_phase_5b",
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol": "docs/FD_FV_PHASE_5B_PROTOCOL.md",
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                PROTOCOL,
                PHASE5A_PROTOCOL,
                PHASE5A_ORACLE,
                BURGERS_SOURCE,
                WENO_SOURCE,
                FV_SOURCE,
                INFRASTRUCTURE,
                Path(__file__),
            )
        },
        "environment": environment(),
        "predecessor": predecessor,
        "projection_oracle": projections,
        "constant_and_conservation": conservation,
        "spatial_convergence": spatial,
        "complete_solve_convergence": solves,
        "differentiation": gradients,
        "compiler_and_device": compiler,
        "transfer_evidence": transfers,
        "gate_decisions": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "passed": all(gates.values()),
        "performance_measurements_collected": False,
        "explicit_exclusions": [
            "performance_timing",
            "nonlinear_shock",
            "multidimensional_burgers",
            "best_practical_lane",
            "dynamic_alpha",
            "mixed_precision",
            "arbitrary_order_fv",
            "automatic_selection",
            "dveb_changes",
            "publication_claim",
        ],
    }
    output.mkdir(parents=True)
    record_path = output / "qualification.json"
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (output / "SHA256SUMS").write_text(
        f"{sha256(record_path)}  qualification.json\n"
    )
    print(f"wrote Phase 5B qualification to {record_path}")
    print(f"passed={payload['passed']} failed_gates={payload['failed_gates']}")


if __name__ == "__main__":
    main()
