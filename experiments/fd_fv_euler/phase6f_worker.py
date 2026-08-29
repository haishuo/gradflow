#!/usr/bin/env python3
"""One isolated worker for the frozen Phase-6F deployment study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch

from experiments.fd_fv_euler.phase6b_problem import (
    conserved_to_primitive,
    shu_structure,
    sod_wave_metrics,
)
from experiments.fd_fv_euler.phase6c_problem import (
    METHOD_IDS,
    adaptive_solve,
    primitive_error_metrics,
    shock_expected,
    shock_initial,
    stage_function,
    tensor_hash,
)
from experiments.fd_fv_euler.phase6c_shock_worker import eligibility
from experiments.fd_fv_euler.phase6e_aot_model import boundary_and_time
from experiments.fd_fv_euler.phase6e_aot_worker import outputs
from experiments.fd_fv_euler.run_phase6e_repro import comparison


ENDPOINTS = ("cuda_jit", "aot_host", "aot_tensor")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_aot_solve(
    module: Callable[..., object], problem: str, initial: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    _, _, final_time = boundary_and_time(problem)
    state = initial
    time_value = 0.0
    steps = 0
    minimum_density = math.inf
    minimum_pressure = math.inf
    failure_stage = None
    while time_value < final_time:
        remaining = state.new_tensor(final_time - time_value)
        state, dt, density, pressure, finite = outputs(module(state, remaining), 5)
        dt_value = float(dt)
        density_values = density.detach().cpu().tolist()
        pressure_values = pressure.detach().cpu().tolist()
        finite_values = finite.detach().cpu().tolist()
        for index, (rho, pressure_value, valid) in enumerate(
            zip(density_values, pressure_values, finite_values), 1
        ):
            minimum_density = min(minimum_density, rho)
            minimum_pressure = min(minimum_pressure, pressure_value)
            if not valid or rho <= 0.0 or pressure_value <= 0.0:
                failure_stage = f"ssp_rk3_stage_{index}"
                break
        if failure_stage is not None:
            break
        time_value += dt_value
        steps += 1
        if steps > 1_000_000:
            raise RuntimeError("Phase 6F host AOT step guard exceeded")
    return state, {
        "completed": failure_stage is None and time_value >= final_time,
        "failure_stage": failure_stage,
        "steps": steps,
        "simulated_time": time_value,
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
        "cfl_scalar_host_controlled": True,
    }


def tensor_aot_solve(
    module: Callable[..., object], problem: str, initial: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    _, _, final_time = boundary_and_time(problem)
    final, time_tensor, steps_tensor, density, pressure, failed = outputs(
        module(initial), 6
    )
    torch.cuda.synchronize()
    diagnostics = {
        "completed": not bool(failed)
        and float(time_tensor) >= final_time
        and int(steps_tensor) < 1_000_000,
        "failure_stage": "tensor_loop_stage" if bool(failed) else None,
        "steps": int(steps_tensor),
        "simulated_time": float(time_tensor),
        "minimum_density": float(density),
        "minimum_pressure": float(pressure),
        "cfl_scalar_host_controlled": False,
    }
    return final, diagnostics


def jit_solve(
    method: str, problem: str, initial: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    boundary, _, final_time = boundary_and_time(problem)
    stages = torch.compile(
        stage_function(method, 800, boundary), fullgraph=True, dynamic=False
    )
    return adaptive_solve(
        method,
        initial,
        final_time,
        boundary,
        stages,
        check_stages=True,
    )


def oracle(
    problem: str, method: str, final_cpu: torch.Tensor, diagnostics: dict[str, Any]
) -> dict[str, Any]:
    expected_conserved, expected_primitive = shock_expected(method, problem, 800)
    primitive = conserved_to_primitive(final_cpu)
    errors = primitive_error_metrics(primitive, expected_primitive)
    feature = (
        sod_wave_metrics(primitive, 800)
        if problem == "sod"
        else shu_structure(primitive, expected_primitive, 800)
    )
    passed, gates = eligibility(problem, 800, diagnostics, errors, feature)
    return {
        "primitive_errors": errors,
        "conserved_l1_errors": torch.mean(
            torch.abs(final_cpu - expected_conserved), dim=-1
        ).tolist(),
        "feature_metrics": feature,
        "gate_decisions": gates,
        "passed": passed,
    }


def load_package(path: Path) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    module = torch._inductor.aoti_load_package(str(path))
    elapsed = (time.perf_counter_ns() - started) * 1.0e-9
    return module, elapsed


def run_prepare(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.endpoint != "aot_host" or arguments.package is None:
        raise ValueError("preparation requires one host AOT package")
    module, load_seconds = load_package(arguments.package)
    input_started = time.perf_counter_ns()
    initial = shock_initial(arguments.method, arguments.problem, 800).cuda()
    _, _, final_time = boundary_and_time(arguments.problem)
    remaining = initial.new_full((), final_time)
    input_seconds = (time.perf_counter_ns() - input_started) * 1.0e-9
    call_started = time.perf_counter_ns()
    result = outputs(module(initial, remaining), 5)
    torch.cuda.synchronize()
    first_call_seconds = (time.perf_counter_ns() - call_started) * 1.0e-9
    return {
        "status": "completed",
        "kind": "phase6f_cache_preparation",
        "endpoint": arguments.endpoint,
        "problem": arguments.problem,
        "method": arguments.method,
        "package_path": str(arguments.package),
        "package_sha256": sha256(arguments.package),
        "package_load_seconds": load_seconds,
        "input_construction_transfer_seconds": input_seconds,
        "first_call_seconds": first_call_seconds,
        "output_shapes": [list(value.shape) for value in result],
        "eligible": True,
    }


def run_profile(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.endpoint != "aot_tensor" or arguments.package is None:
        raise ValueError("profiling is restricted to the tensor-loop AOT lane")
    module, load_seconds = load_package(arguments.package)
    initial = shock_initial(arguments.method, arguments.problem, 800).cuda()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    ) as profile:
        final, diagnostics = tensor_aot_solve(module, arguments.problem, initial)
        torch.cuda.synchronize()
    events = []
    for event in profile.key_averages():
        lower = event.key.lower()
        if any(
            marker in lower
            for marker in (
                "_local_scalar_dense",
                "memcpy dtoh",
                "device-to-host",
                "cuda memcpy dtoh",
                "synchronize",
            )
        ):
            events.append(
                {
                    "key": event.key,
                    "count": event.count,
                    "cpu_time_total_us": event.cpu_time_total,
                    "device_time_total_us": event.device_time_total,
                }
            )
    return {
        "status": "completed",
        "kind": "phase6f_tensor_loop_profile",
        "endpoint": arguments.endpoint,
        "problem": arguments.problem,
        "method": arguments.method,
        "package_path": str(arguments.package),
        "package_sha256": sha256(arguments.package),
        "package_load_seconds": load_seconds,
        "diagnostics": diagnostics,
        "selected_events": sorted(events, key=lambda item: item["key"]),
        "host_synchronization_observed": bool(events),
        "final_device": final.device.type,
        "eligible": bool(diagnostics["completed"] and final.device.type == "cuda"),
    }


def run_solve(arguments: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to Phase 6F worker")
    torch.cuda.reset_peak_memory_stats()
    load_seconds = None
    module = None
    if arguments.endpoint != "cuda_jit":
        if arguments.package is None:
            raise ValueError("AOT endpoint requires --package")
        module, load_seconds = load_package(arguments.package)

    input_started = time.perf_counter_ns()
    initial = shock_initial(arguments.method, arguments.problem, 800).cuda()
    torch.cuda.synchronize()
    input_seconds = (time.perf_counter_ns() - input_started) * 1.0e-9

    solve_started = time.perf_counter_ns()
    if arguments.endpoint == "cuda_jit":
        final, diagnostics = jit_solve(arguments.method, arguments.problem, initial)
    elif arguments.endpoint == "aot_host":
        final, diagnostics = host_aot_solve(module, arguments.problem, initial)
    else:
        final, diagnostics = tensor_aot_solve(module, arguments.problem, initial)
    torch.cuda.synchronize()
    solve_seconds = (time.perf_counter_ns() - solve_started) * 1.0e-9
    resident_before_materialization = final.device.type == "cuda"

    output_started = time.perf_counter_ns()
    final_cpu = final.detach().cpu().contiguous()
    arguments.array_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(arguments.array_output, final_cpu.numpy(), allow_pickle=False)
    output_seconds = (time.perf_counter_ns() - output_started) * 1.0e-9

    authority = np.load(arguments.authority_array, allow_pickle=False)
    authority_record = json.loads(arguments.authority_record.read_text())
    parity = comparison(
        authority,
        final_cpu.numpy(),
        steps=max(authority_record["diagnostics"]["steps"], diagnostics["steps"]),
        reference_name=arguments.authority_array.name,
        actual_name=arguments.array_output.name,
    )
    parity["step_count_match"] = (
        authority_record["diagnostics"]["steps"] == diagnostics["steps"]
    )
    parity["passed"] = bool(parity["passed"] and parity["step_count_match"])
    oracle_result = oracle(arguments.problem, arguments.method, final_cpu, diagnostics)
    eligible_result = bool(
        parity["passed"]
        and oracle_result["passed"]
        and diagnostics["completed"]
        and diagnostics["minimum_density"] > 0.0
        and diagnostics["minimum_pressure"] > 0.0
        and resident_before_materialization
        and final.dtype == torch.float64
        and tuple(final.shape) == (3, 800)
    )
    return {
        "status": "completed",
        "kind": "phase6f_solve",
        "endpoint": arguments.endpoint,
        "problem": arguments.problem,
        "method": arguments.method,
        "formulation_id": METHOD_IDS[arguments.method],
        "cells": 800,
        "replicate": arguments.replicate,
        "dtype": "float64",
        "shape": [3, 800],
        "package_path": str(arguments.package) if arguments.package else None,
        "package_sha256": sha256(arguments.package) if arguments.package else None,
        "package_load_seconds": load_seconds,
        "input_construction_transfer_seconds": input_seconds,
        "complete_solve_seconds": solve_seconds,
        "final_materialization_serialization_seconds": output_seconds,
        "diagnostics": diagnostics,
        "oracle": oracle_result,
        "authority_parity": parity,
        "terminal_sha256": tensor_hash(final_cpu),
        "array_file": arguments.array_output.name,
        "array_file_sha256": sha256(arguments.array_output),
        "array_bytes": arguments.array_output.stat().st_size,
        "terminal_state_cuda_before_materialization": resident_before_materialization,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "eligible": eligible_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("prepare", "solve", "profile"), required=True)
    parser.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    parser.add_argument("--problem", choices=("sod", "shu_osher"), required=True)
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--authority-array", type=Path)
    parser.add_argument("--authority-record", type=Path)
    parser.add_argument("--array-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("refusing to overwrite Phase 6F worker output")
    try:
        if arguments.action == "prepare":
            result = run_prepare(arguments)
        elif arguments.action == "profile":
            result = run_profile(arguments)
        else:
            if not all(
                (arguments.authority_array, arguments.authority_record, arguments.array_output)
            ):
                raise ValueError("solve action requires authority and array paths")
            result = run_solve(arguments)
    except Exception as error:
        result = {
            "status": "failed",
            "kind": f"phase6f_{arguments.action}",
            "endpoint": arguments.endpoint,
            "problem": arguments.problem,
            "method": arguments.method,
            "replicate": arguments.replicate,
            "error_type": type(error).__name__,
            "error": str(error),
            "eligible": False,
        }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
