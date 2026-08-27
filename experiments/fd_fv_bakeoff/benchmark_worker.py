#!/usr/bin/env python3
"""Measure one admitted CPU method/dimension/size Phase-4B cell."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time
from typing import Any, Callable

import torch

from problem import (
    FINAL_TIME,
    METHOD_IDS,
    conservation,
    errors,
    projected_state,
    solve,
    step_function,
)


SOLVE_WARMUPS = 1
SOLVE_REPETITIONS = 5
STEP_WARMUPS = 5
STEP_REPETITIONS = 30


def quantile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]


def sample_statistics(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
        "q1_seconds": quantile(ordered, 0.25),
        "q3_seconds": quantile(ordered, 0.75),
    }


def measure(
    call: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    *,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    output = state
    for _ in range(warmups):
        output = call(state)
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = call(state)
        samples.append((time.perf_counter_ns() - started) * 1.0e-9)
    return sample_statistics(samples), output


def cache_bytes() -> int:
    root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not root:
        return 0
    return sum(
        path.stat().st_size
        for path in Path(root).rglob("*")
        if path.is_file()
    )


def execute(method: str, dimension: int, cells: int) -> dict[str, Any]:
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    initial = projected_state(method, dimension, cells)
    expected = projected_state(method, dimension, cells, time=FINAL_TIME)
    step, steps = step_function(method, dimension, cells)

    def complete(step_call: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return solve(initial, step_call, steps)

    eager_final = complete(step)
    eager_l1, eager_l2 = errors(eager_final, expected)
    eager_mass = conservation(initial, eager_final, dimension, cells)
    eager_solve, eager_measured_final = measure(
        lambda _: complete(step),
        initial,
        warmups=SOLVE_WARMUPS,
        repetitions=SOLVE_REPETITIONS,
    )
    eager_step, _ = measure(
        step,
        initial,
        warmups=STEP_WARMUPS,
        repetitions=STEP_REPETITIONS,
    )

    torch._dynamo.reset()
    compiled_step = torch.compile(step, fullgraph=True, dynamic=False)
    started = time.perf_counter_ns()
    first_compiled_final = complete(compiled_step)
    first_compiled_seconds = (time.perf_counter_ns() - started) * 1.0e-9
    compiled_l1, compiled_l2 = errors(first_compiled_final, expected)
    compiled_mass = conservation(
        initial, first_compiled_final, dimension, cells
    )
    compiled_solve, compiled_measured_final = measure(
        lambda _: complete(compiled_step),
        initial,
        warmups=SOLVE_WARMUPS,
        repetitions=SOLVE_REPETITIONS,
    )
    compiled_step_stats, _ = measure(
        compiled_step,
        initial,
        warmups=STEP_WARMUPS,
        repetitions=STEP_REPETITIONS,
    )

    parity = float(torch.max(torch.abs(first_compiled_final - eager_final)))
    repeated_eager_parity = float(
        torch.max(torch.abs(eager_measured_final - eager_final))
    )
    repeated_compiled_parity = float(
        torch.max(torch.abs(compiled_measured_final - first_compiled_final))
    )
    finite = all(
        math.isfinite(value)
        for value in (eager_l1, eager_l2, compiled_l1, compiled_l2, parity)
    )
    eligible = (
        finite
        and parity <= 2.0e-11
        and repeated_eager_parity == 0.0
        and repeated_compiled_parity == 0.0
        and eager_mass[2]
        and compiled_mass[2]
    )
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "status": "completed",
        "method": method,
        "formulation_id": METHOD_IDS[method],
        "dimension": dimension,
        "cells_per_axis": cells,
        "logical_cells": cells**dimension,
        "steps": steps,
        "dtype": "float64",
        "device": "cpu",
        "persistent_state_bytes": initial.numel() * initial.element_size(),
        "accuracy": {
            "eager_l1_error": eager_l1,
            "eager_l2_error": eager_l2,
            "compiled_l1_error": compiled_l1,
            "compiled_l2_error": compiled_l2,
            "compiled_eager_maximum_absolute_difference": parity,
            "repeated_eager_maximum_absolute_difference": repeated_eager_parity,
            "repeated_compiled_maximum_absolute_difference": repeated_compiled_parity,
        },
        "conservation": {
            "eager_mass_change": eager_mass[0],
            "eager_mass_bound": eager_mass[1],
            "eager_passed": eager_mass[2],
            "compiled_mass_change": compiled_mass[0],
            "compiled_mass_bound": compiled_mass[1],
            "compiled_passed": compiled_mass[2],
        },
        "eager": {
            "complete_solve": eager_solve,
            "ssp_rk3_step": eager_step,
        },
        "compiled": {
            "first_complete_solve_seconds": first_compiled_seconds,
            "complete_solve": compiled_solve,
            "ssp_rk3_step": compiled_step_stats,
        },
        "memory": {
            "peak_process_rss_bytes": peak_rss_bytes,
            "compiler_cache_bytes": cache_bytes(),
        },
        "controls": {
            "solve_warmups": SOLVE_WARMUPS,
            "solve_repetitions": SOLVE_REPETITIONS,
            "step_warmups": STEP_WARMUPS,
            "step_repetitions": STEP_REPETITIONS,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "visible_logical_cpus": os.cpu_count(),
            "process_affinity": affinity,
        },
        "finite": finite,
        "eligible": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    try:
        result = execute(arguments.method, arguments.dimension, arguments.cells)
    except Exception as error:
        result = {
            "status": "failed",
            "method": arguments.method,
            "dimension": arguments.dimension,
            "cells_per_axis": arguments.cells,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
