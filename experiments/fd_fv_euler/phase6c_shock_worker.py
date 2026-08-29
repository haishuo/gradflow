#!/usr/bin/env python3
"""Run one launch-to-host-answer Euler shock Phase-6C pilot point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

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


THRESHOLDS = (
    ROOT
    / "experiments/euler_boundary_shock/results/phase_a_20260827/thresholds.json"
)


def eligibility(
    problem: str,
    cells: int,
    diagnostics: dict,
    errors: dict,
    feature: dict,
) -> tuple[bool, dict]:
    thresholds = json.loads(THRESHOLDS.read_text())[problem]
    gates = {
        "completed": diagnostics["completed"],
        "positive_stages": diagnostics["minimum_density"] > 0.0
        and diagnostics["minimum_pressure"] > 0.0,
    }
    if cells == 800 and problem == "sod":
        gates["finest_l1_thresholds"] = all(
            errors["l1"][name] <= thresholds["l1_max"][name]
            for name in ("density", "velocity", "pressure")
        )
        gates["wave_locations"] = all(
            item["error_cells"] <= thresholds["wave_location_error_cells_max"]
            for item in feature.values()
        )
    elif cells == 800 and problem == "shu_osher":
        gates["finest_l1_thresholds"] = all(
            errors["l1"][name] <= thresholds["l1_max_to_n12800"][name]
            for name in ("density", "velocity", "pressure")
        )
        gates["density_correlation"] = (
            feature["density_correlation"]
            >= thresholds["density_correlation_min"]
        )
        gates["density_total_variation_ratio"] = (
            thresholds["density_total_variation_ratio_min"]
            <= feature["density_total_variation_ratio"]
            <= thresholds["density_total_variation_ratio_max"]
        )
    return all(gates.values()), gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("sod", "shu_osher"), required=True)
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--cells", choices=(200, 800), type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    try:
        if arguments.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible to the worker")
        initial = shock_initial(
            arguments.method, arguments.problem, arguments.cells
        ).to(arguments.device)
        boundary = (
            "transmissive_shu_osher"
            if arguments.problem == "shu_osher"
            else "transmissive"
        )
        final_time = 1.8 if arguments.problem == "shu_osher" else 0.2
        stages = stage_function(arguments.method, arguments.cells, boundary)
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
        final_cpu = final.cpu()
        expected_conserved, expected_primitive = shock_expected(
            arguments.method, arguments.problem, arguments.cells
        )
        actual_primitive = conserved_to_primitive(final_cpu)
        errors = primitive_error_metrics(actual_primitive, expected_primitive)
        if arguments.problem == "sod":
            feature = sod_wave_metrics(actual_primitive, arguments.cells)
        else:
            feature = shu_structure(
                actual_primitive, expected_primitive, arguments.cells
            )
        eligible, gates = eligibility(
            arguments.problem, arguments.cells, diagnostics, errors, feature
        )
        conserved_difference = torch.abs(final_cpu - expected_conserved)
        result = {
            "status": "completed",
            "kind": "shock",
            "problem": arguments.problem,
            "method": arguments.method,
            "formulation_id": METHOD_IDS[arguments.method],
            "device": arguments.device,
            "mode": arguments.mode,
            "cells": arguments.cells,
            "diagnostics": diagnostics,
            "primitive_errors": errors,
            "conserved_l1_errors": torch.mean(
                conserved_difference, dim=-1
            ).tolist(),
            "feature_metrics": feature,
            "gate_decisions": gates,
            "terminal_sha256": tensor_hash(final_cpu),
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
            "eligible": eligible and final_cpu.device.type == "cpu",
        }
    except Exception as error:
        result = {
            "status": "failed",
            "kind": "shock",
            "problem": arguments.problem,
            "method": arguments.method,
            "device": arguments.device,
            "mode": arguments.mode,
            "cells": arguments.cells,
            "error_type": type(error).__name__,
            "error": str(error),
            "eligible": False,
        }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
