#!/usr/bin/env python3
"""Measure one isolated smooth-complete or resident-step Phase-6C cell."""

from __future__ import annotations

import argparse
import json
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

from experiments.fd_fv_euler.phase6c_problem import (
    FINAL_SMOOTH_TIME,
    METHOD_IDS,
    adaptive_solve,
    conservation,
    error_norms,
    fixed_step_function,
    smooth_expected,
    smooth_initial,
    stage_function,
    statistics_record,
    tensor_hash,
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


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def wall_duration(
    call: Callable[[], tuple[torch.Tensor, dict[str, Any]]], device: str
) -> tuple[float, torch.Tensor, dict[str, Any]]:
    synchronize(device)
    started = time.perf_counter_ns()
    output, diagnostics = call()
    synchronize(device)
    return (
        (time.perf_counter_ns() - started) * 1.0e-9,
        output,
        diagnostics,
    )


def step_duration(
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


def complete_samples(
    call: Callable[[], tuple[torch.Tensor, dict[str, Any]]],
    *,
    device: str,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any], list[str]]:
    for _ in range(COMPLETE_WARMUPS):
        call()
    synchronize(device)
    samples = []
    hashes = []
    output = torch.empty(0)
    diagnostics: dict[str, Any] = {}
    for _ in range(COMPLETE_REPETITIONS):
        seconds, output, diagnostics = wall_duration(call, device)
        samples.append(seconds)
        hashes.append(tensor_hash(output))
    return statistics_record(samples), output, diagnostics, hashes


def complete_transfer_samples(
    call: Callable[[torch.Tensor], tuple[torch.Tensor, dict[str, Any]]],
    host_initial: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any], list[str]]:
    for _ in range(COMPLETE_WARMUPS):
        output, _ = call(host_initial.to("cuda"))
        output.cpu()
    torch.cuda.synchronize()
    samples = []
    hashes = []
    final_cpu = host_initial
    diagnostics: dict[str, Any] = {}
    for _ in range(COMPLETE_REPETITIONS):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output, diagnostics = call(host_initial.to("cuda"))
        final_cpu = output.cpu()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) * 1.0e-9)
        hashes.append(tensor_hash(final_cpu))
    return statistics_record(samples), final_cpu, diagnostics, hashes


def step_samples(
    call: Callable[[], torch.Tensor], device: str
) -> tuple[dict[str, Any], torch.Tensor, list[str]]:
    output = call()
    for _ in range(STEP_WARMUPS - 1):
        output = call()
    synchronize(device)
    samples = []
    hashes = []
    for _ in range(STEP_REPETITIONS):
        seconds, output = step_duration(call, device)
        samples.append(seconds)
        hashes.append(tensor_hash(output))
    return statistics_record(samples), output, hashes


def step_transfer_samples(
    call: Callable[[torch.Tensor], torch.Tensor], host_initial: torch.Tensor
) -> tuple[dict[str, Any], torch.Tensor]:
    output = host_initial
    for _ in range(TRANSFER_STEP_WARMUPS):
        output = call(host_initial.to("cuda")).cpu()
    torch.cuda.synchronize()
    samples = []
    for _ in range(TRANSFER_STEP_REPETITIONS):
        started = time.perf_counter_ns()
        output = call(host_initial.to("cuda")).cpu()
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


def complete_checks(
    initial: torch.Tensor,
    final: torch.Tensor,
    expected: torch.Tensor,
    diagnostics: dict[str, Any],
    cells: int,
) -> dict[str, Any]:
    dx = 1.0 / cells
    result = {
        **error_norms(final, expected),
        "conservation": conservation(
            initial, final, dx, diagnostics["steps"]
        ),
        "finite": bool(torch.isfinite(final).all()),
        "shape": list(final.shape),
        "dtype": str(final.dtype).removeprefix("torch."),
        "device": str(final.device),
        "terminal_sha256": tensor_hash(final),
        "solve_diagnostics": diagnostics,
    }
    result["passed"] = (
        result["finite"]
        and result["conservation"]["passed"]
        and result["shape"] == [3, cells]
        and result["dtype"] == "float64"
        and diagnostics["completed"]
    )
    return result


def complete_worker(
    method: str, device: str, cells: int, replicate: int
) -> dict[str, Any]:
    host_initial = smooth_initial(method, cells)
    initial = host_initial.to(device)
    expected = smooth_expected(method, cells).to(device)
    eager_stages = stage_function(method, cells, "periodic")

    def run_eager(
        values: torch.Tensor = initial,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return adaptive_solve(
            method,
            values,
            FINAL_SMOOTH_TIME,
            "periodic",
            eager_stages,
            check_stages=False,
        )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    eager_stats, eager_final, eager_diagnostics, eager_hashes = complete_samples(
        run_eager, device=device
    )
    eager_memory = cuda_memory()
    eager_transfer = None
    eager_transfer_hashes = None
    if device == "cuda":
        (
            eager_transfer,
            _,
            _,
            eager_transfer_hashes,
        ) = complete_transfer_samples(
            lambda values: run_eager(values), host_initial
        )

    torch._dynamo.reset()
    compiled_stages = torch.compile(eager_stages, fullgraph=True, dynamic=False)

    def run_compiled(
        values: torch.Tensor = initial,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return adaptive_solve(
            method,
            values,
            FINAL_SMOOTH_TIME,
            "periodic",
            compiled_stages,
            check_stages=False,
        )

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    first_seconds, first_final, first_diagnostics = wall_duration(
        run_compiled, device
    )
    (
        compiled_stats,
        compiled_final,
        compiled_diagnostics,
        compiled_hashes,
    ) = complete_samples(run_compiled, device=device)
    compiled_memory = cuda_memory()
    compiled_transfer = None
    compiled_transfer_hashes = None
    if device == "cuda":
        (
            compiled_transfer,
            _,
            _,
            compiled_transfer_hashes,
        ) = complete_transfer_samples(
            lambda values: run_compiled(values), host_initial
        )

    eager_checks = complete_checks(
        initial, eager_final, expected, eager_diagnostics, cells
    )
    compiled_checks = complete_checks(
        initial,
        compiled_final,
        expected,
        compiled_diagnostics,
        cells,
    )
    parity = float(torch.max(torch.abs(compiled_final - eager_final)))
    first_parity = float(torch.max(torch.abs(first_final - eager_final)))
    repeat_deterministic = (
        len(set(eager_hashes)) == 1 and len(set(compiled_hashes)) == 1
    )
    transfer_deterministic = device != "cuda" or (
        len(set(eager_transfer_hashes or [])) == 1
        and len(set(compiled_transfer_hashes or [])) == 1
    )
    eligible = (
        eager_checks["passed"]
        and compiled_checks["passed"]
        and parity <= 5.0e-11
        and first_parity <= 5.0e-11
        and repeat_deterministic
        and transfer_deterministic
        and eager_diagnostics["steps"] == compiled_diagnostics["steps"]
        and first_diagnostics["completed"]
    )
    return {
        "status": "completed",
        "kind": "complete",
        "endpoint": "state_resident_host_controlled",
        "method": method,
        "formulation_id": METHOD_IDS[method],
        "device": device,
        "cells": cells,
        "replicate": replicate,
        "persistent_state_bytes": initial.numel() * initial.element_size(),
        "accuracy": {
            "eager": eager_checks,
            "compiled": compiled_checks,
            "compiled_eager_maximum_absolute_difference": parity,
            "compiled_first_eager_maximum_absolute_difference": first_parity,
            "repeat_deterministic": repeat_deterministic,
            "transfer_repeat_deterministic": transfer_deterministic,
        },
        "eager": {
            "resident_complete_solve": eager_stats,
            "prepared_transfer_complete_solve": eager_transfer,
            "terminal_hashes": eager_hashes,
            "transfer_terminal_hashes": eager_transfer_hashes,
            "cuda_memory": eager_memory,
        },
        "compiled": {
            "first_complete_solve_seconds": first_seconds,
            "resident_complete_solve": compiled_stats,
            "prepared_transfer_complete_solve": compiled_transfer,
            "terminal_hashes": compiled_hashes,
            "transfer_terminal_hashes": compiled_transfer_hashes,
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
            "cuda_complete_clock": "synchronized_perf_counter_ns",
        },
        "eligible": eligible,
    }


def step_worker(
    method: str, device: str, cells: int, replicate: int
) -> dict[str, Any]:
    host_initial = smooth_initial(method, cells)
    initial = host_initial.to(device)
    eager_step = fixed_step_function(method, cells)
    eager_reference = eager_step(initial)
    synchronize(device)
    modes: dict[str, Any] = {}
    for mode in ("eager", "compiled"):
        if mode == "compiled":
            torch._dynamo.reset()
            call = torch.compile(eager_step, fullgraph=True, dynamic=False)
            first_seconds, first_output = step_duration(
                lambda: call(initial), device
            )
        else:
            call = eager_step
            first_seconds = None
            first_output = eager_reference
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        resident, output, hashes = step_samples(
            lambda call=call: call(initial), device
        )
        transfer = None
        if device == "cuda":
            transfer, _ = step_transfer_samples(call, host_initial)
        parity = float(torch.max(torch.abs(output - eager_reference)))
        first_parity = float(torch.max(torch.abs(first_output - eager_reference)))
        finite = bool(torch.isfinite(output).all())
        deterministic = len(set(hashes)) == 1
        modes[mode] = {
            "first_call_seconds": first_seconds,
            "resident_step": resident,
            "transfer_inclusive_step": transfer,
            "terminal_hashes": hashes,
            "maximum_absolute_difference_from_eager": parity,
            "first_maximum_absolute_difference_from_eager": first_parity,
            "finite": finite,
            "deterministic": deterministic,
            "dtype": str(output.dtype).removeprefix("torch."),
            "output_device": str(output.device),
            "cuda_memory": cuda_memory(),
            "eligible": finite
            and deterministic
            and parity <= 5.0e-11
            and first_parity <= 5.0e-11,
        }
    return {
        "status": "completed",
        "kind": "step",
        "method": method,
        "formulation_id": METHOD_IDS[method],
        "device": device,
        "cells": cells,
        "replicate": replicate,
        "dt": 0.05 / cells,
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
            "cuda_resident_clock": "cuda_events",
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
