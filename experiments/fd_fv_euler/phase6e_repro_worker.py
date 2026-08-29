#!/usr/bin/env python3
"""Retain one isolated Phase-6E Euler shock terminal array."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("sod", "shu_osher"), required=True)
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.array_output.exists():
        raise FileExistsError("refusing to overwrite Phase 6E worker output")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    try:
        if arguments.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible to the worker")
        initial = shock_initial(arguments.method, arguments.problem, 800).to(
            arguments.device
        )
        boundary = (
            "transmissive_shu_osher"
            if arguments.problem == "shu_osher"
            else "transmissive"
        )
        final_time = 1.8 if arguments.problem == "shu_osher" else 0.2
        stages = stage_function(arguments.method, 800, boundary)
        if arguments.mode == "compiled":
            stages = torch.compile(stages, fullgraph=True, dynamic=False)
        if arguments.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        final, diagnostics = adaptive_solve(
            arguments.method,
            initial,
            final_time,
            boundary,
            stages,
            check_stages=True,
        )
        final_cpu = final.detach().cpu().contiguous()
        expected_conserved, expected_primitive = shock_expected(
            arguments.method, arguments.problem, 800
        )
        actual_primitive = conserved_to_primitive(final_cpu)
        errors = primitive_error_metrics(actual_primitive, expected_primitive)
        feature = (
            sod_wave_metrics(actual_primitive, 800)
            if arguments.problem == "sod"
            else shu_structure(actual_primitive, expected_primitive, 800)
        )
        eligible, gates = eligibility(
            arguments.problem, 800, diagnostics, errors, feature
        )
        conserved_difference = torch.abs(final_cpu - expected_conserved)
        arguments.array_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(arguments.array_output, final_cpu.numpy(), allow_pickle=False)
        result = {
            "status": "completed",
            "kind": "phase6e_reproducibility",
            "problem": arguments.problem,
            "method": arguments.method,
            "formulation_id": METHOD_IDS[arguments.method],
            "device": arguments.device,
            "mode": arguments.mode,
            "cells": 800,
            "replicate": arguments.replicate,
            "dtype": "float64",
            "shape": [3, 800],
            "diagnostics": diagnostics,
            "primitive_errors": errors,
            "conserved_l1_errors": torch.mean(
                conserved_difference, dim=-1
            ).tolist(),
            "feature_metrics": feature,
            "gate_decisions": gates,
            "terminal_sha256": tensor_hash(final_cpu),
            "array_file": arguments.array_output.name,
            "array_file_sha256": sha256(arguments.array_output),
            "array_bytes": arguments.array_output.stat().st_size,
            "host_visible_answer": final_cpu.device.type == "cpu",
            "peak_process_rss_bytes": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated()
                if arguments.device == "cuda"
                else None
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved()
                if arguments.device == "cuda"
                else None
            ),
            "torch_deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "eligible": bool(
                eligible
                and final_cpu.device.type == "cpu"
                and final_cpu.dtype == torch.float64
                and final_cpu.shape == (3, 800)
            ),
        }
    except Exception as error:
        result = {
            "status": "failed",
            "kind": "phase6e_reproducibility",
            "problem": arguments.problem,
            "method": arguments.method,
            "device": arguments.device,
            "mode": arguments.mode,
            "cells": 800,
            "replicate": arguments.replicate,
            "error_type": type(error).__name__,
            "error": str(error),
            "eligible": False,
        }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
