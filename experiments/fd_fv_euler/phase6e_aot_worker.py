#!/usr/bin/env python3
"""Qualify one packaged Phase-6E AOT Euler solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any
import zipfile


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
    primitive_error_metrics,
    shock_expected,
    shock_initial,
    tensor_hash,
)
from experiments.fd_fv_euler.phase6c_shock_worker import eligibility
from experiments.fd_fv_euler.phase6e_aot_model import (
    HostControlledAdvance,
    boundary_and_time,
)
from experiments.fd_fv_euler.run_phase6e_repro import comparison


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outputs(value: object, expected: int) -> tuple[torch.Tensor, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != expected:
        raise TypeError(f"expected {expected} AOT outputs, received {type(value)!r}")
    result = tuple(value)
    if not all(isinstance(item, torch.Tensor) for item in result):
        raise TypeError("AOT outputs must all be tensors")
    return result  # type: ignore[return-value]


def profiler_record(call: Any) -> dict[str, Any]:
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        )
    ) as profile:
        call()
        torch.cuda.synchronize()
    keys = sorted({event.key for event in profile.key_averages()})
    forbidden = sorted(
        key
        for key in keys
        if any(
            marker in key.lower()
            for marker in (
                "_local_scalar_dense",
                "memcpy dtoh",
                "device-to-host",
                "cuda memcpy dtoh",
            )
        )
    )
    return {"event_keys": keys, "forbidden_movement_events": forbidden}


def package_inventory(package: Path) -> dict[str, Any]:
    with zipfile.ZipFile(package) as archive:
        names = sorted(archive.namelist())
    return {
        "members": names,
        "shared_objects": [name for name in names if name.endswith(".so")],
        "cuda_sources": [name for name in names if name.endswith(".cu")],
        "cpp_sources": [name for name in names if name.endswith(".cpp")],
    }


def runtime_cache_inventory() -> list[str]:
    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not cache or not Path(cache).exists():
        return []
    return sorted(
        str(path.relative_to(cache))
        for path in Path(cache).rglob("*")
        if path.is_file()
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


def host_solve(
    module: Any,
    method: str,
    problem: str,
    initial: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    _, _, final_time = boundary_and_time(problem)
    state = initial
    time_value = 0.0
    steps = 0
    minimum_density = math.inf
    minimum_pressure = math.inf
    failure_stage = None
    loop_started = time.perf_counter()
    while time_value < final_time:
        remaining = state.new_tensor(final_time - time_value)
        state, dt, density, pressure, finite = outputs(module(state, remaining), 5)
        dt_value = float(dt)
        density_values = density.detach().cpu().tolist()
        pressure_values = pressure.detach().cpu().tolist()
        finite_values = finite.detach().cpu().tolist()
        for index, (rho, p, valid) in enumerate(
            zip(density_values, pressure_values, finite_values), 1
        ):
            minimum_density = min(minimum_density, rho)
            minimum_pressure = min(minimum_pressure, p)
            if not valid or rho <= 0.0 or p <= 0.0:
                failure_stage = f"ssp_rk3_stage_{index}"
                break
        if failure_stage is not None:
            break
        time_value += dt_value
        steps += 1
        if steps > 1_000_000:
            raise RuntimeError("Phase 6E host AOT step guard exceeded")
    torch.cuda.synchronize()
    return state, {
        "completed": failure_stage is None and time_value >= final_time,
        "failure_stage": failure_stage,
        "steps": steps,
        "simulated_time": time_value,
        "minimum_density": minimum_density,
        "minimum_pressure": minimum_pressure,
        "cfl_scalar_host_controlled": True,
    }, {
        "loop_seconds_qualification": time.perf_counter() - loop_started,
        "declared_device_to_host_operations_per_step": 4,
        "declared_device_to_host_scalar_values_per_step": 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("host", "device"), required=True)
    parser.add_argument("--problem", choices=("sod", "shu_osher"), required=True)
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--authority-array", type=Path, required=True)
    parser.add_argument("--authority-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.array_output.exists():
        raise FileExistsError("refusing to overwrite Phase 6E AOT qualification")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible to AOT qualification worker")
        package_hash = sha256(arguments.package)
        inventory = package_inventory(arguments.package)
        cache_before = runtime_cache_inventory()
        load_started = time.perf_counter()
        module = torch._inductor.aoti_load_package(str(arguments.package))
        load_seconds = time.perf_counter() - load_started
        cache_after_load = runtime_cache_inventory()
        initial = shock_initial(arguments.method, arguments.problem, 800).cuda()
        _, _, final_time = boundary_and_time(arguments.problem)

        if arguments.lane == "host":
            remaining = initial.new_full((), final_time)
            eager = HostControlledAdvance(arguments.method, arguments.problem).eval()
            eager_output = outputs(eager(initial, remaining), 5)
            aot_output = outputs(module(initial, remaining), 5)
            torch.cuda.synchronize()
            one_step_differences = [
                float(torch.max(torch.abs(actual - expected)))
                if actual.dtype != torch.bool
                else float(torch.count_nonzero(actual != expected))
                for actual, expected in zip(aot_output, eager_output)
            ]
            movement = profiler_record(lambda: module(initial, remaining))
            final, diagnostics, control = host_solve(
                module, arguments.method, arguments.problem, initial
            )
        else:
            one_step_differences = None
            movement = profiler_record(lambda: module(initial))
            final, time_tensor, steps_tensor, density, pressure, failed = outputs(
                module(initial), 6
            )
            torch.cuda.synchronize()
            diagnostics = {
                "completed": not bool(failed)
                and float(time_tensor) >= final_time
                and int(steps_tensor) < 1_000_000,
                "failure_stage": "device_loop_stage" if bool(failed) else None,
                "steps": int(steps_tensor),
                "simulated_time": float(time_tensor),
                "minimum_density": float(density),
                "minimum_pressure": float(pressure),
                "cfl_scalar_host_controlled": False,
            }
            control = {
                "declared_device_to_host_operations_inside_loop": 0,
                "declared_device_to_host_scalar_values_inside_loop": 0,
            }

        cache_after_call = runtime_cache_inventory()
        final_cpu = final.detach().cpu().contiguous()
        arguments.array_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(arguments.array_output, final_cpu.numpy(), allow_pickle=False)
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
        oracle_result = oracle(
            arguments.problem, arguments.method, final_cpu, diagnostics
        )
        runtime_sources = [
            name
            for name in cache_after_call
            if name.endswith((".cpp", ".cu")) and name not in cache_before
        ]
        one_step_passed = (
            arguments.lane == "device"
            or (
                one_step_differences is not None
                and max(one_step_differences) <= 5.0e-11
            )
        )
        eligible = (
            one_step_passed
            and parity["passed"]
            and oracle_result["passed"]
            and diagnostics["completed"]
            and diagnostics["minimum_density"] > 0.0
            and diagnostics["minimum_pressure"] > 0.0
            and not movement["forbidden_movement_events"]
            and not runtime_sources
            and final.device.type == "cuda"
            and final.dtype == torch.float64
            and tuple(final.shape) == (3, 800)
            and bool(inventory["shared_objects"])
        )
        result = {
            "status": "completed",
            "kind": "phase6e_aot_qualification",
            "lane": arguments.lane,
            "problem": arguments.problem,
            "method": arguments.method,
            "cells": 800,
            "dtype": "float64",
            "shape": [3, 800],
            "package_path": str(arguments.package),
            "package_sha256": package_hash,
            "package_inventory": inventory,
            "package_load_seconds_qualification": load_seconds,
            "runtime_cache_before": cache_before,
            "runtime_cache_after_load": cache_after_load,
            "runtime_cache_after_call": cache_after_call,
            "runtime_compiler_sources_created": runtime_sources,
            "one_step_maximum_absolute_differences": one_step_differences,
            "one_step_tolerance": 5.0e-11,
            "one_step_passed": one_step_passed,
            "movement_probe": movement,
            "diagnostics": diagnostics,
            "control_boundary": control,
            "oracle": oracle_result,
            "authority_parity": parity,
            "terminal_sha256": tensor_hash(final_cpu),
            "array_file": arguments.array_output.name,
            "array_file_sha256": sha256(arguments.array_output),
            "array_bytes": arguments.array_output.stat().st_size,
            "peak_process_rss_bytes": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "eligible": eligible,
        }
    except Exception as error:
        result = {
            "status": "failed",
            "kind": "phase6e_aot_qualification",
            "lane": arguments.lane,
            "problem": arguments.problem,
            "method": arguments.method,
            "error_type": type(error).__name__,
            "error": str(error),
            "eligible": False,
        }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
