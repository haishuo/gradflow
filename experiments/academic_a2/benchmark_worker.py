#!/usr/bin/env python3
"""Isolated correctness and warm-timing worker for one A2 configuration."""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import statistics
import subprocess
import time
from typing import Any, Callable

import torch

from gradflow import WENOJS, euler_weno_rhs, periodic_vortex


WARMUPS = 5
REPETITIONS = 20
BOOTSTRAPS = 20_000
SEED = 20260830
THERMAL_STOP_C = 80.0


def smooth_input(n: int, dimensions: int, dtype: torch.dtype) -> torch.Tensor:
    shape = (n,) * dimensions
    result = torch.zeros(shape, dtype=dtype)
    for axis in range(dimensions):
        coordinate = 2.0 * math.pi * torch.arange(n, dtype=dtype) / n
        view = [1] * dimensions
        view[axis] = n
        coordinate = coordinate.reshape(view)
        result = result + torch.sin((axis + 1) * coordinate)
        result = result + 0.17 * torch.cos((axis + 2) * coordinate)
    return result / dimensions


def make_problem(
    subject: str, order: int, dtype: torch.dtype, dimensions: int, n: int
) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    if subject == "scalar":
        state = smooth_input(n, dimensions, dtype)
        scheme = WENOJS(order)
        dx = 2.0 * math.pi / n

        def rhs(values: torch.Tensor) -> torch.Tensor:
            result = scheme.rhs(
                values,
                dx,
                lambda q: 0.5 * q.square(),
                lambda q: q,
                axis=0,
            )
            for axis in range(1, dimensions):
                result = result + scheme.rhs(
                    values,
                    dx,
                    lambda q: 0.5 * q.square(),
                    lambda q: q,
                    axis=axis,
                )
            return result

        return state, rhs
    if dimensions != 3:
        raise ValueError("the characteristic subject is three-dimensional")
    state, spacing = periodic_vortex((n, n, n), dtype=dtype)

    def rhs(values: torch.Tensor) -> torch.Tensor:
        return euler_weno_rhs(values, spacing, order=order)

    return state, rhs


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    difference = actual.to(torch.float64) - reference.to(torch.float64)
    scale = max(float(torch.amax(torch.abs(reference))), 1.0)
    return {
        "bitwise_identical": bool(torch.equal(actual, reference)),
        "maximum_normalized": float(torch.amax(torch.abs(difference))) / scale,
        "rms_normalized": float(torch.sqrt(torch.mean(difference.square()))) / scale,
    }


def bounds(subject: str, dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return (5.0e-5, 5.0e-6) if subject == "scalar" else (3.0e-4, 3.0e-5)
    return (5.0e-11, 5.0e-12)


def comparison_passes(metrics: dict[str, Any], limit: tuple[float, float]) -> bool:
    return metrics["maximum_normalized"] <= limit[0] and metrics["rms_normalized"] <= limit[1]


def health(output: torch.Tensor, subject: str) -> dict[str, Any]:
    values = output
    if subject == "characteristic":
        values = output[(slice(None),) + (slice(None, -1),) * 3]
        flattened = values.reshape(values.shape[0], -1)
        absolute_sum = torch.sum(torch.abs(flattened), dim=1, dtype=torch.float64)
        residual = torch.abs(torch.sum(flattened, dim=1, dtype=torch.float64))
        bound = 32.0 * torch.finfo(output.dtype).eps * absolute_sum
        passed = bool(torch.all(residual <= bound))
        maximum_residual = float(torch.amax(residual))
        maximum_bound = float(torch.amax(bound))
    else:
        absolute_sum = torch.sum(torch.abs(values), dtype=torch.float64)
        residual = torch.abs(torch.sum(values, dtype=torch.float64))
        bound = 32.0 * torch.finfo(output.dtype).eps * absolute_sum
        passed = bool(residual <= bound)
        maximum_residual = float(residual)
        maximum_bound = float(bound)
    return {
        "finite": bool(torch.isfinite(output).all()),
        "conservation_passed": passed,
        "maximum_conservation_absolute": maximum_residual,
        "maximum_conservation_bound": maximum_bound,
        "checksum_float64": float(torch.sum(output, dtype=torch.float64)),
        "maximum_absolute": float(torch.amax(torch.abs(output))),
    }


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def call_wall(
    function: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor, device: str
) -> tuple[float, torch.Tensor]:
    synchronize(device)
    started = time.perf_counter()
    output = function(state)
    synchronize(device)
    return time.perf_counter() - started, output


def call_milliseconds(
    function: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor, device: str
) -> tuple[float, torch.Tensor]:
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function(state)
        end.record()
        end.synchronize()
        return start.elapsed_time(end), output
    started = time.perf_counter_ns()
    output = function(state)
    return (time.perf_counter_ns() - started) / 1.0e6, output


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def statistics_record(values: list[float]) -> dict[str, Any]:
    median = statistics.median(values)
    return {
        "count": len(values),
        "values": values,
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median": median,
        "median_absolute_deviation": statistics.median(abs(value - median) for value in values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def paired_analysis(eager: list[float], compiled: list[float], seed: int) -> dict[str, Any]:
    ratios = [compiled_value / eager_value for eager_value, compiled_value in zip(eager, compiled)]
    generator = random.Random(seed)
    bootstrapped = []
    for _ in range(BOOTSTRAPS):
        sample = [ratios[generator.randrange(len(ratios))] for _ in ratios]
        bootstrapped.append(statistics.median(sample))
    interval = [quantile(bootstrapped, 0.025), quantile(bootstrapped, 0.975)]
    median = statistics.median(ratios)
    if median < 0.95 and interval[1] < 1.0:
        decision = "compiled_win"
    elif median > 1.05 and interval[0] > 1.0:
        decision = "eager_win"
    else:
        decision = "unresolved"
    return {
        "compiled_over_eager": {**statistics_record(ratios), "bootstrap_median_95_ci": interval},
        "decision": decision,
    }


def telemetry() -> dict[str, Any]:
    command = (
        "timestamp,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,"
        "power.limit,utilization.gpu,memory.used"
    )
    completed = subprocess.run(
        ("nvidia-smi", f"--query-gpu={command}", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    return {
        "timestamp": values[0],
        "temperature_c": float(values[1]),
        "pstate": values[2],
        "sm_clock_mhz": float(values[3]),
        "memory_clock_mhz": float(values[4]),
        "power_w": float(values[5]),
        "power_limit_w": float(values[6]),
        "utilization_percent": float(values[7]),
        "memory_used_mib": float(values[8]),
    }


def peak_cuda(
    function: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor
) -> dict[str, int]:
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    output = function(state)
    torch.cuda.synchronize()
    record = {
        "incremental_peak_allocated_bytes": max(0, torch.cuda.max_memory_allocated() - allocated),
        "incremental_peak_reserved_bytes": max(0, torch.cuda.max_memory_reserved() - reserved),
    }
    del output
    return record


def time_resident_pair(
    functions: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    admitted: dict[str, bool],
    state: torch.Tensor,
    device: str,
    seed: int,
) -> dict[str, Any]:
    active = [name for name in ("eager", "compiled") if admitted[name]]
    for name in active:
        for _ in range(WARMUPS):
            call_milliseconds(functions[name], state, device)
    samples = {name: [] for name in active}
    blocks = []
    generator = random.Random(seed)
    for repetition in range(REPETITIONS):
        order = active[:]
        generator.shuffle(order)
        milliseconds = {}
        for name in order:
            elapsed, output = call_milliseconds(functions[name], state, device)
            milliseconds[name] = elapsed
            samples[name].append(elapsed)
            del output
        blocks.append({"repetition": repetition, "order": order, "milliseconds": milliseconds})
    record: dict[str, Any] = {
        "warmups_per_lane": WARMUPS,
        "randomized_complete_pair_blocks": REPETITIONS,
        "blocks": blocks,
        "lanes": {name: statistics_record(values) for name, values in samples.items()},
    }
    if len(active) == 2:
        record["paired_analysis"] = paired_analysis(samples["eager"], samples["compiled"], seed + 1)
    return record


def transfer_call(
    function: Callable[[torch.Tensor], torch.Tensor], state_cpu: torch.Tensor
) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    state = state_cpu.to("cuda")
    output = function(state)
    result = output.cpu()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - started) / 1.0e6, result


def time_transfer_pair(
    functions: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    admitted: dict[str, bool],
    state_cpu: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    active = [name for name in ("eager", "compiled") if admitted[name]]
    for name in active:
        for _ in range(WARMUPS):
            transfer_call(functions[name], state_cpu)
    samples = {name: [] for name in active}
    blocks = []
    generator = random.Random(seed)
    for repetition in range(REPETITIONS):
        order = active[:]
        generator.shuffle(order)
        milliseconds = {}
        for name in order:
            elapsed, result = transfer_call(functions[name], state_cpu)
            milliseconds[name] = elapsed
            samples[name].append(elapsed)
            del result
        blocks.append({"repetition": repetition, "order": order, "milliseconds": milliseconds})
    record: dict[str, Any] = {
        "warmups_per_lane": WARMUPS,
        "randomized_complete_pair_blocks": REPETITIONS,
        "blocks": blocks,
        "lanes": {name: statistics_record(values) for name, values in samples.items()},
    }
    if len(active) == 2:
        record["paired_analysis"] = paired_analysis(samples["eager"], samples["compiled"], seed + 2)
    return record


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=("scalar", "characteristic"), required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    dtype = getattr(torch, arguments.dtype)
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    state_cpu, function_cpu = make_problem(
        arguments.subject, arguments.order, dtype, arguments.dimensions, arguments.size
    )
    reference_seconds, reference = call_wall(function_cpu, state_cpu, "cpu")
    reference_health = health(reference, arguments.subject)
    limit = bounds(arguments.subject, dtype)

    state = state_cpu.to(arguments.device)
    _, function = make_problem(
        arguments.subject, arguments.order, dtype, arguments.dimensions, arguments.size
    )
    if arguments.device == "cuda":
        # The closure contains no tensors, but moving the input establishes the device.
        telemetry_before = telemetry()
        if telemetry_before["temperature_c"] >= THERMAL_STOP_C:
            raise RuntimeError("thermal stop before A2 worker")
    else:
        telemetry_before = None

    eager_first_seconds, eager_output = call_wall(function, state, arguments.device)
    eager_comparison = error_metrics(eager_output.cpu(), reference)
    eager_health = health(eager_output, arguments.subject)
    eager_admitted = (
        comparison_passes(eager_comparison, limit)
        and eager_health["finite"]
        and eager_health["conservation_passed"]
    )

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(function, fullgraph=True, dynamic=False)
    compiled_error = None
    compiled_output = None
    compiled_first_seconds = None
    compiled_comparison = None
    compiled_health = None
    try:
        compiled_first_seconds, compiled_output = call_wall(compiled, state, arguments.device)
        compiled_comparison = error_metrics(compiled_output.cpu(), reference)
        compiled_health = health(compiled_output, arguments.subject)
    except Exception as error:  # noqa: BLE001 - failure is experimental evidence
        compiled_error = {"type": type(error).__name__, "message": str(error)}
    stats = torch._dynamo.utils.counters.get("stats", {})
    graph_breaks = torch._dynamo.utils.counters.get("graph_break", {})
    graph = {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "calls_captured": int(stats.get("calls_captured", 0)),
        "graph_break_count": int(sum(graph_breaks.values())),
        "graph_break_reasons": {str(key): int(value) for key, value in graph_breaks.items()},
    }
    compiled_admitted = bool(
        compiled_comparison is not None
        and comparison_passes(compiled_comparison, limit)
        and compiled_health is not None
        and compiled_health["finite"]
        and compiled_health["conservation_passed"]
        and graph["unique_graphs"] == 1
        and graph["graph_break_count"] == 0
    )
    functions = {"eager": function, "compiled": compiled}

    result: dict[str, Any] = {
        "schema": "gradflow-academic-a2-worker-v1",
        "subject": arguments.subject,
        "order": arguments.order,
        "dtype": arguments.dtype,
        "dimensions": arguments.dimensions,
        "n": arguments.size,
        "device": arguments.device,
        "cells": arguments.size**arguments.dimensions,
        "reference_first_call_seconds": reference_seconds,
        "reference_health": reference_health,
        "bounds": {"maximum_normalized": limit[0], "rms_normalized": limit[1]},
        "first_call_seconds": {"eager": eager_first_seconds, "compiled": compiled_first_seconds},
        "correctness": {
            "eager": {"comparison": eager_comparison, "health": eager_health, "admitted": eager_admitted},
            "compiled": {
                "comparison": compiled_comparison,
                "health": compiled_health,
                "graph": graph,
                "error": compiled_error,
                "admitted": compiled_admitted,
            },
        },
        "telemetry_before": telemetry_before,
    }

    if arguments.device == "cpu":
        result["cpu"] = {}
        for threads in (1, 6):
            torch.set_num_threads(threads)
            admitted = {"eager": eager_admitted, "compiled": compiled_admitted}
            per_thread_correctness = {}
            for name in ("eager", "compiled"):
                if not admitted[name]:
                    continue
                output = functions[name](state)
                metrics = error_metrics(output, reference)
                lane_health = health(output, arguments.subject)
                lane_passed = comparison_passes(metrics, limit) and lane_health["finite"] and lane_health["conservation_passed"]
                admitted[name] = lane_passed
                per_thread_correctness[name] = {"comparison": metrics, "health": lane_health, "admitted": lane_passed}
            result["cpu"][str(threads)] = {
                "correctness": per_thread_correctness,
                "resident_timing": time_resident_pair(
                    functions,
                    admitted,
                    state,
                    "cpu",
                    SEED + arguments.order + threads + arguments.size,
                ),
            }
        result["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    else:
        admitted = {"eager": eager_admitted, "compiled": compiled_admitted}
        result["cuda"] = {
            "resident_timing": time_resident_pair(
                functions, admitted, state, "cuda", SEED + arguments.order + arguments.size
            ),
            "transfer_inclusive_timing": time_transfer_pair(
                functions, admitted, state_cpu, SEED + arguments.order + arguments.size + 17
            ),
            "memory": {
                name: peak_cuda(functions[name], state)
                for name in ("eager", "compiled")
                if admitted[name]
            },
        }
        telemetry_after = telemetry()
        if telemetry_after["temperature_c"] >= THERMAL_STOP_C:
            raise RuntimeError("thermal stop after A2 worker")
        result["telemetry_after"] = telemetry_after

    result["status"] = "complete"
    result["canonical_source_changed"] = False
    result["performance_measured_without_parity"] = False
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
