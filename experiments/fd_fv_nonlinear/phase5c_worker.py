#!/usr/bin/env python3
"""Measure one isolated warm complete-solve or resident-step Phase-5C cell."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from experiments.fd_fv_nonlinear.performance_problem import (
    FINAL_TIME,
    METHOD_IDS,
    conservation,
    errors,
    solve,
    state,
    statistics_record,
    step_function,
    timestep,
)


COMPLETE_WARMUPS = 1
COMPLETE_REPETITIONS = 3
STEP_WARMUPS = 10
STEP_REPETITIONS = 30
TRANSFER_STEP_WARMUPS = 5
TRANSFER_STEP_REPETITIONS = 20


def cache_bytes() -> int:
    root = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not root:
        return 0
    return sum(
        path.stat().st_size for path in Path(root).rglob("*") if path.is_file()
    )


def cuda_memory() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def duration(
    call: Callable[[], torch.Tensor], device: str
) -> tuple[float, torch.Tensor]:
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        output = call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1.0e-3, output
    started = time.perf_counter_ns()
    output = call()
    return (time.perf_counter_ns() - started) * 1.0e-9, output


def resident_samples(
    call: Callable[[], torch.Tensor],
    *,
    device: str,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    output = call()
    for _ in range(warmups - 1):
        output = call()
    if device == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        sample, output = duration(call, device)
        samples.append(sample)
    return statistics_record(samples), output


def transfer_samples(
    call: Callable[[torch.Tensor], torch.Tensor],
    host_state: torch.Tensor,
    *,
    warmups: int,
    repetitions: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    output = host_state
    for _ in range(warmups):
        output = call(host_state.to("cuda")).cpu()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        output = call(host_state.to("cuda")).cpu()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) * 1.0e-9)
    return statistics_record(samples), output


def controls(replicate: int) -> dict[str, Any]:
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "replicate": replicate,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "visible_logical_cpus": os.cpu_count(),
        "process_affinity": affinity,
    }


def output_checks(
    initial: torch.Tensor,
    final: torch.Tensor,
    expected: torch.Tensor,
    cells: int,
) -> dict[str, Any]:
    l1, l2 = errors(final, expected)
    change, bound, conserved = conservation(initial, final, cells)
    return {
        "l1_error": l1,
        "l2_error": l2,
        "mass_change": change,
        "mass_bound": bound,
        "conservation_passed": conserved,
        "finite": bool(torch.isfinite(final).all()),
        "shape": list(final.shape),
        "dtype": str(final.dtype).removeprefix("torch."),
        "device": str(final.device),
    }


def complete_worker(
    method: str, device: str, cells: int, replicate: int
) -> dict[str, Any]:
    host_initial = state(method, cells)
    initial = host_initial.to(device)
    expected = state(method, cells, FINAL_TIME).to(device)
    eager_step = step_function(method, cells)
    _, steps = timestep(cells)

    def eager_solve(values: torch.Tensor) -> torch.Tensor:
        return solve(values, eager_step, steps)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    eager_resident, eager_final = resident_samples(
        lambda: eager_solve(initial),
        device=device,
        warmups=COMPLETE_WARMUPS,
        repetitions=COMPLETE_REPETITIONS,
    )
    eager_memory = cuda_memory()
    eager_transfer = None
    if device == "cuda":
        eager_transfer, _ = transfer_samples(
            eager_solve,
            host_initial,
            warmups=COMPLETE_WARMUPS,
            repetitions=COMPLETE_REPETITIONS,
        )

    torch._dynamo.reset()
    compiled_step = torch.compile(eager_step, fullgraph=True, dynamic=False)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    first_seconds, compiled_first = duration(
        lambda: solve(initial, compiled_step, steps), device
    )

    def compiled_solve(values: torch.Tensor) -> torch.Tensor:
        return solve(values, compiled_step, steps)

    compiled_resident, compiled_final = resident_samples(
        lambda: compiled_solve(initial),
        device=device,
        warmups=COMPLETE_WARMUPS,
        repetitions=COMPLETE_REPETITIONS,
    )
    compiled_memory = cuda_memory()
    compiled_transfer = None
    if device == "cuda":
        compiled_transfer, _ = transfer_samples(
            compiled_solve,
            host_initial,
            warmups=COMPLETE_WARMUPS,
            repetitions=COMPLETE_REPETITIONS,
        )

    eager_checks = output_checks(initial, eager_final, expected, cells)
    compiled_checks = output_checks(initial, compiled_final, expected, cells)
    parity = float(torch.max(torch.abs(compiled_first - eager_final)))
    repeat_parity = float(torch.max(torch.abs(compiled_final - compiled_first)))
    eligible = (
        eager_checks["finite"]
        and compiled_checks["finite"]
        and eager_checks["conservation_passed"]
        and compiled_checks["conservation_passed"]
        and parity <= 2.0e-11
        and repeat_parity == 0.0
        and eager_checks["dtype"] == "float64"
        and compiled_checks["dtype"] == "float64"
    )
    return {
        "status": "completed",
        "kind": "complete",
        "method": method,
        "formulation_id": METHOD_IDS[method],
        "device": device,
        "cells": cells,
        "steps": steps,
        "replicate": replicate,
        "persistent_state_bytes": initial.numel() * initial.element_size(),
        "accuracy": {
            "eager": eager_checks,
            "compiled": compiled_checks,
            "compiled_eager_maximum_absolute_difference": parity,
            "compiled_repeat_maximum_absolute_difference": repeat_parity,
        },
        "eager": {
            "resident_complete_solve": eager_resident,
            "prepared_transfer_complete_solve": eager_transfer,
            "cuda_memory": eager_memory,
        },
        "compiled": {
            "first_complete_solve_seconds": first_seconds,
            "resident_complete_solve": compiled_resident,
            "prepared_transfer_complete_solve": compiled_transfer,
            "cuda_memory": compiled_memory,
        },
        "memory": {
            "peak_process_rss_bytes": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "compiler_cache_bytes": cache_bytes(),
        },
        "controls": {
            **controls(replicate),
            "complete_warmups": COMPLETE_WARMUPS,
            "complete_repetitions": COMPLETE_REPETITIONS,
        },
        "eligible": eligible,
    }


def step_worker(
    method: str, device: str, cells: int, replicate: int
) -> dict[str, Any]:
    host_initial = state(method, cells)
    initial = host_initial.to(device)
    eager_step = step_function(method, cells)
    modes: dict[str, Any] = {}
    eager_reference = eager_step(initial)
    if device == "cuda":
        torch.cuda.synchronize()

    for mode in ("eager", "compiled"):
        if mode == "compiled":
            torch._dynamo.reset()
            call = torch.compile(eager_step, fullgraph=True, dynamic=False)
            first_seconds, first_output = duration(lambda: call(initial), device)
        else:
            call = eager_step
            first_seconds = None
            first_output = eager_reference
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        resident, output = resident_samples(
            lambda call=call: call(initial),
            device=device,
            warmups=STEP_WARMUPS,
            repetitions=STEP_REPETITIONS,
        )
        transfer = None
        if device == "cuda":
            transfer, _ = transfer_samples(
                call,
                host_initial,
                warmups=TRANSFER_STEP_WARMUPS,
                repetitions=TRANSFER_STEP_REPETITIONS,
            )
        parity = float(torch.max(torch.abs(output - eager_reference)))
        first_parity = float(torch.max(torch.abs(first_output - eager_reference)))
        finite = bool(torch.isfinite(output).all())
        modes[mode] = {
            "first_call_seconds": first_seconds,
            "resident_step": resident,
            "transfer_inclusive_step": transfer,
            "maximum_absolute_difference_from_eager": parity,
            "first_maximum_absolute_difference_from_eager": first_parity,
            "finite": finite,
            "dtype": str(output.dtype).removeprefix("torch."),
            "output_device": str(output.device),
            "cuda_memory": cuda_memory(),
            "eligible": finite and parity <= 2.0e-11 and first_parity <= 2.0e-11,
        }
    return {
        "status": "completed",
        "kind": "step",
        "method": method,
        "formulation_id": METHOD_IDS[method],
        "device": device,
        "cells": cells,
        "replicate": replicate,
        "persistent_state_bytes": initial.numel() * initial.element_size(),
        "modes": modes,
        "memory": {
            "peak_process_rss_bytes": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "compiler_cache_bytes": cache_bytes(),
        },
        "controls": {
            **controls(replicate),
            "step_warmups": STEP_WARMUPS,
            "step_repetitions": STEP_REPETITIONS,
            "transfer_step_warmups": TRANSFER_STEP_WARMUPS,
            "transfer_step_repetitions": TRANSFER_STEP_REPETITIONS,
        },
        "eligible": all(mode["eligible"] for mode in modes.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("complete", "step"), required=True)
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    try:
        if arguments.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible to the worker")
        if arguments.kind == "complete":
            result = complete_worker(
                arguments.method,
                arguments.device,
                arguments.cells,
                arguments.replicate,
            )
        else:
            result = step_worker(
                arguments.method,
                arguments.device,
                arguments.cells,
                arguments.replicate,
            )
    except Exception as error:
        result = {
            "status": "failed",
            "kind": arguments.kind,
            "method": arguments.method,
            "device": arguments.device,
            "cells": arguments.cells,
            "replicate": arguments.replicate,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
