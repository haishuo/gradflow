#!/usr/bin/env python3
"""Build one frozen Phase-6E AOTInductor package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch
import torch._inductor

from experiments.fd_fv_euler.phase6c_problem import shock_initial
from experiments.fd_fv_euler.phase6e_aot_model import (
    DeviceLoopSolve,
    HostControlledAdvance,
    boundary_and_time,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("host", "device"), required=True)
    parser.add_argument("--problem", choices=("sod", "shu_osher"), required=True)
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.record.exists():
        raise FileExistsError("refusing to overwrite Phase 6E AOT output")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.record.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible to AOT builder")
        state = shock_initial(arguments.method, arguments.problem, 800).cuda()
        _, _, final_time = boundary_and_time(arguments.problem)
        module: torch.nn.Module
        inputs: tuple[torch.Tensor, ...]
        if arguments.lane == "host":
            module = HostControlledAdvance(
                arguments.method, arguments.problem
            ).eval()
            inputs = (state, state.new_full((), final_time))
        else:
            module = DeviceLoopSolve(arguments.method, arguments.problem).eval()
            inputs = (state,)
        torch.cuda.synchronize()
        export_started = time.perf_counter()
        exported = torch.export.export(module, inputs, strict=False)
        export_seconds = time.perf_counter() - export_started
        compile_started = time.perf_counter()
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(arguments.output)
        )
        torch.cuda.synchronize()
        record = {
            "status": "completed",
            "lane": arguments.lane,
            "problem": arguments.problem,
            "method": arguments.method,
            "cells": 800,
            "dtype": "float64",
            "export_seconds": export_seconds,
            "compile_package_seconds": time.perf_counter() - compile_started,
            "total_build_seconds": time.perf_counter() - started,
            "package_path": str(arguments.output),
            "package_sha256": sha256(arguments.output),
            "package_bytes": arguments.output.stat().st_size,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "custom_operator_used": False,
        }
    except Exception as error:
        record = {
            "status": "failed",
            "lane": arguments.lane,
            "problem": arguments.problem,
            "method": arguments.method,
            "cells": 800,
            "dtype": "float64",
            "total_build_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "custom_operator_used": False,
        }
    arguments.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)
    if record["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
