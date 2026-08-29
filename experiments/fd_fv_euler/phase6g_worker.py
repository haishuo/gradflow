#!/usr/bin/env python3
"""One isolated worker for the frozen Phase-6G internal-loader study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np
import torch

from experiments.fd_fv_euler.phase6c_problem import (
    METHOD_IDS,
    shock_initial,
    tensor_hash,
)
from experiments.fd_fv_euler.phase6f_worker import (
    host_aot_solve,
    jit_solve,
    oracle,
    sha256,
    tensor_aot_solve,
)
from experiments.fd_fv_euler.run_phase6e_repro import comparison


ENDPOINTS = ("cuda_jit", "aot_host_internal", "aot_tensor_internal")


class InternalAotiModule:
    """Minimal callable adapter around PyTorch's version-locked internal loader."""

    def __init__(self, package: Path) -> None:
        self.loader = torch._C._aoti.AOTIModelPackageLoader(
            str(package), "model", False, 1, -1
        )

    def __call__(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        return self.loader.boxed_run(list(inputs))


def load_internal(path: Path) -> tuple[InternalAotiModule, float]:
    started = time.perf_counter_ns()
    module = InternalAotiModule(path)
    return module, (time.perf_counter_ns() - started) * 1.0e-9


def solve(arguments: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to Phase 6G worker")
    torch.cuda.reset_peak_memory_stats()
    module = None
    load_seconds = None
    if arguments.endpoint != "cuda_jit":
        if arguments.package is None:
            raise ValueError("internal AOTI endpoint requires --package")
        module, load_seconds = load_internal(arguments.package)

    input_started = time.perf_counter_ns()
    initial = shock_initial(arguments.method, arguments.problem, 800).cuda()
    torch.cuda.synchronize()
    input_seconds = (time.perf_counter_ns() - input_started) * 1.0e-9

    solve_started = time.perf_counter_ns()
    if arguments.endpoint == "cuda_jit":
        final, diagnostics = jit_solve(arguments.method, arguments.problem, initial)
    elif arguments.endpoint == "aot_host_internal":
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
    eligible = bool(
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
        "kind": "phase6g_solve",
        "endpoint": arguments.endpoint,
        "loader_boundary": (
            "torch_compile" if arguments.endpoint == "cuda_jit"
            else "torch._C._aoti.AOTIModelPackageLoader.boxed_run"
        ),
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
        "eligible": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=ENDPOINTS, required=True)
    parser.add_argument("--problem", choices=("sod", "shu_osher"), required=True)
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--authority-array", type=Path, required=True)
    parser.add_argument("--authority-record", type=Path, required=True)
    parser.add_argument("--array-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.array_output.exists():
        raise FileExistsError("refusing to overwrite Phase 6G worker output")
    try:
        result = solve(arguments)
    except Exception as error:
        result = {
            "status": "failed",
            "kind": "phase6g_solve",
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
