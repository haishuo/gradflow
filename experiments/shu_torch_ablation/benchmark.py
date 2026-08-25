#!/usr/bin/env python3
"""One-shot Fortran CPU versus compiled-PyTorch CUDA crossover sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from shu_euler_torch import cfl_timestep, periodic_vortex, ssp_rk3_step


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_FORTRAN_2D = (
    EXPERIMENT_DIR.parent / "fortran_scaling" / "build" / "weno_dynamic"
)
DEFAULT_FORTRAN_3D = EXPERIMENT_DIR / "build" / "shu_euler_3d"


def _run_fortran(dimension: int, size: int, binary: Path) -> float:
    if dimension == 2:
        stdin = f"3\n{size} {size}\n0.1\n1\n0.001\n1\n0\n"
        environment = dict(os.environ, WENO_WRITE_SOLUTION="0")
    else:
        stdin = f"{size} {size} {size}\n0.1\n1\n0.001\n"
        environment = os.environ.copy()

    started = time.perf_counter()
    completed = subprocess.run(
        [str(binary)],
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        check=True,
    )
    elapsed = time.perf_counter() - started
    serious_flags = ("IEEE_DIVIDE_BY_ZERO", "IEEE_INVALID", "IEEE_OVERFLOW")
    if any(flag in completed.stderr for flag in serious_flags):
        raise RuntimeError(f"Fortran signalled a serious exception at N={size}")
    return elapsed


def _initialize_gpu(dimension: int, size: int) -> tuple[torch.Tensor, tuple[float, ...], torch.Tensor]:
    intervals = (size,) * dimension
    state, spacing = periodic_vortex(intervals, device="cuda", dtype=torch.float32)
    dt = torch.minimum(
        cfl_timestep(state, spacing, 0.1),
        torch.tensor(0.001, device="cuda", dtype=torch.float32),
    )
    return state, spacing, dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--fortran", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    fortran_binary = arguments.fortran or (
        DEFAULT_FORTRAN_2D if arguments.dimension == 2 else DEFAULT_FORTRAN_3D
    )
    if not fortran_binary.is_file():
        raise SystemExit(f"missing Fortran binary: {fortran_binary}")

    compiled_step = torch.compile(ssp_rk3_step, fullgraph=True, dynamic=True)
    calibration_size = min(arguments.sizes)
    calibration_state, calibration_spacing, calibration_dt = _initialize_gpu(
        arguments.dimension, calibration_size
    )
    torch.cuda.synchronize()
    compilation_started = time.perf_counter()
    calibration_result = compiled_step(
        calibration_state, calibration_spacing, calibration_dt
    )
    torch.cuda.synchronize()
    compile_and_calibration_seconds = time.perf_counter() - compilation_started
    if not bool(torch.isfinite(calibration_result).all()):
        raise RuntimeError("compiled calibration result is non-finite")
    del calibration_state, calibration_result
    torch.cuda.empty_cache()

    records: list[dict[str, float | int]] = []
    for size in arguments.sizes:
        fortran_seconds = _run_fortran(arguments.dimension, size, fortran_binary)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        initialization_started = time.perf_counter()
        state, spacing, dt = _initialize_gpu(arguments.dimension, size)
        torch.cuda.synchronize()
        initialization_seconds = time.perf_counter() - initialization_started

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        start_event.record()
        result = compiled_step(state, spacing, dt)
        end_event.record()
        torch.cuda.synchronize()
        step_wall_seconds = time.perf_counter() - step_started
        step_device_seconds = start_event.elapsed_time(end_event) / 1000.0
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError(f"non-finite CUDA result at N={size}")

        record = {
            "dimension": arguments.dimension,
            "size": size,
            "fortran_process_seconds": fortran_seconds,
            "gpu_initialization_seconds": initialization_seconds,
            "gpu_step_wall_seconds": step_wall_seconds,
            "gpu_step_device_seconds": step_device_seconds,
            "gpu_initialized_plus_step_seconds": (
                initialization_seconds + step_wall_seconds
            ),
            "fortran_over_gpu_step": fortran_seconds / step_wall_seconds,
            "fortran_over_gpu_initialized_plus_step": fortran_seconds
            / (initialization_seconds + step_wall_seconds),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "dt": float(dt),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del state, result, dt
        torch.cuda.empty_cache()

    output = {
        "schema_version": 1,
        "precision": "float32",
        "runs_per_grid": 1,
        "dimension": arguments.dimension,
        "sizes": arguments.sizes,
        "fortran_binary": str(fortran_binary.resolve()),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compile_and_calibration_seconds": compile_and_calibration_seconds,
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
