#!/usr/bin/env python3
"""Isolated A3 objective/gradient benchmark worker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import platform
import statistics
import time
from typing import Callable

import torch

from problem import make_problem


WARMUPS = 3
REPETITIONS = 20
REFERENCE_SPEED = 0.9


def summary(values: list[float]) -> dict[str, float | int | list[float]]:
    median = statistics.median(values)
    return {
        "count": len(values),
        "values": values,
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median": median,
        "median_absolute_deviation": statistics.median(
            abs(value - median) for value in values
        ),
        "sample_standard_deviation": statistics.stdev(values),
    }


def value_and_gradient(
    call: Callable[[torch.Tensor], torch.Tensor], speed: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    value = call(speed)
    gradient = torch.autograd.grad(value, speed)[0]
    return value, gradient


def comparison(actual: float, reference: float) -> dict[str, float | bool]:
    absolute = abs(actual - reference)
    normalized = absolute / max(abs(reference), 1.0)
    return {
        "actual": actual,
        "reference": reference,
        "absolute": absolute,
        "normalized": normalized,
        "admitted": math.isfinite(actual) and normalized <= 5.0e-10,
    }


def cpu_timing(
    call: Callable[[torch.Tensor], torch.Tensor],
    speed: torch.Tensor,
    *,
    gradient: bool,
) -> float:
    started = time.perf_counter_ns()
    if gradient:
        value, derivative = value_and_gradient(call, speed)
        _ = float(value.detach()) + float(derivative.detach())
    else:
        value = call(speed)
        _ = float(value.detach())
    return (time.perf_counter_ns() - started) / 1.0e6


def cuda_timing(
    call: Callable[[torch.Tensor], torch.Tensor],
    speed: torch.Tensor,
    *,
    gradient: bool,
) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if gradient:
        value, derivative = value_and_gradient(call, speed)
        _ = value, derivative
    else:
        value = call(speed)
        _ = value
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def timed_lane(
    call: Callable[[torch.Tensor], torch.Tensor],
    speed: torch.Tensor,
    *,
    device: str,
) -> dict[str, object]:
    timer = cuda_timing if device == "cuda" else cpu_timing
    for _ in range(WARMUPS):
        timer(call, speed, gradient=False)
        timer(call, speed, gradient=True)
    forward = [timer(call, speed, gradient=False) for _ in range(REPETITIONS)]
    objective_gradient = [timer(call, speed, gradient=True) for _ in range(REPETITIONS)]
    forward_summary = summary(forward)
    gradient_summary = summary(objective_gradient)
    return {
        "warmups": WARMUPS,
        "repetitions": REPETITIONS,
        "forward_ms": forward_summary,
        "objective_and_gradient_ms": gradient_summary,
        "reverse_mode_over_forward_median": (
            gradient_summary["median"] / forward_summary["median"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    arguments = parser.parse_args()
    if arguments.device == "cuda" and not torch.cuda.is_available():
        print(
            json.dumps(
                {
                    "schema": "gradflow-academic-a3-benchmark-v1",
                    "status": "unavailable",
                    "device": "cuda",
                }
            )
        )
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    cpu_problem = make_problem(128)
    cpu_speed = torch.tensor(REFERENCE_SPEED, dtype=torch.float64, requires_grad=True)
    cpu_value, cpu_gradient = value_and_gradient(cpu_problem.objective, cpu_speed)
    reference_value = float(cpu_value.detach())
    reference_gradient = float(cpu_gradient.detach())

    problem = make_problem(128, device=arguments.device)
    speed = torch.tensor(
        REFERENCE_SPEED,
        dtype=torch.float64,
        device=arguments.device,
        requires_grad=True,
    )
    if arguments.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    eager_started = time.perf_counter()
    eager_value, eager_gradient = value_and_gradient(problem.objective, speed)
    if arguments.device == "cuda":
        torch.cuda.synchronize()
    eager_first_seconds = time.perf_counter() - eager_started
    eager_correctness = {
        "objective": comparison(float(eager_value.detach()), reference_value),
        "gradient": comparison(float(eager_gradient.detach()), reference_gradient),
    }
    eager_admitted = all(item["admitted"] for item in eager_correctness.values())

    explanation = torch._dynamo.explain(problem.objective)(speed)
    graph = {
        "unique_graphs": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "graph_break_reasons": [str(reason) for reason in explanation.break_reasons],
    }
    compiled = torch.compile(problem.objective, fullgraph=True, dynamic=False)
    compiled_error = None
    compiled_first_seconds = None
    compiled_correctness = None
    try:
        compiled_started = time.perf_counter()
        compiled_value, compiled_gradient = value_and_gradient(compiled, speed)
        if arguments.device == "cuda":
            torch.cuda.synchronize()
        compiled_first_seconds = time.perf_counter() - compiled_started
        compiled_correctness = {
            "objective": comparison(float(compiled_value.detach()), reference_value),
            "gradient": comparison(
                float(compiled_gradient.detach()), reference_gradient
            ),
        }
    except Exception as error:  # pragma: no cover - records compiler failures
        compiled_error = f"{type(error).__name__}: {error}"
    compiled_admitted = bool(
        compiled_correctness is not None
        and graph["unique_graphs"] == 1
        and graph["graph_break_count"] == 0
        and all(item["admitted"] for item in compiled_correctness.values())
    )

    timings = {}
    if eager_admitted:
        timings["eager"] = timed_lane(problem.objective, speed, device=arguments.device)
    if compiled_admitted:
        timings["compiled"] = timed_lane(compiled, speed, device=arguments.device)

    memory = None
    if arguments.device == "cuda":
        torch.cuda.synchronize()
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
    print(
        json.dumps(
            {
                "schema": "gradflow-academic-a3-benchmark-v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                "device": arguments.device,
                "n": 128,
                "order": 11,
                "dtype": "float64",
                "reference_speed": REFERENCE_SPEED,
                "environment": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "gpu": (
                        torch.cuda.get_device_name()
                        if arguments.device == "cuda"
                        else None
                    ),
                },
                "reference": {
                    "objective": reference_value,
                    "gradient": reference_gradient,
                },
                "eager": {
                    "first_objective_and_gradient_seconds": eager_first_seconds,
                    "correctness": eager_correctness,
                    "admitted": eager_admitted,
                },
                "compiled": {
                    "first_objective_and_gradient_seconds": compiled_first_seconds,
                    "correctness": compiled_correctness,
                    "graph": graph,
                    "error": compiled_error,
                    "admitted": compiled_admitted,
                },
                "timings": timings,
                "memory": memory,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
