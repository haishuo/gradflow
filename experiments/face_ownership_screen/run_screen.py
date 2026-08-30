#!/usr/bin/env python3
"""Run the frozen ordinary-PyTorch face-ownership screening matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
from typing import Any, Callable

import torch

from gradflow.weno_js import WENOJS


WARMUPS = 5
REPETITIONS = 20
SEED = 20260830
BOOTSTRAPS = 20_000
THERMAL_STOP_C = 80.0
REPRESENTATIONS = ("face_once", "cell_recompute")
MODES = ("eager", "compiled")


def configurations() -> list[dict[str, Any]]:
    records = []
    for order in (5, 15):
        for dtype in ("float32", "float64"):
            for dimensions, n in ((1, 1_048_576), (3, 96)):
                records.append(
                    {
                        "order": order,
                        "dtype": dtype,
                        "dimensions": dimensions,
                        "n": n,
                        "role": "large_factorial",
                    }
                )
    for n in (16, 32, 64, 128):
        records.append(
            {
                "order": 5,
                "dtype": "float32",
                "dimensions": 3,
                "n": n,
                "role": "scale_slice",
            }
        )
    return records


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def telemetry() -> dict[str, Any]:
    fields = (
        "timestamp,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,"
        "power.limit,utilization.gpu,memory.used"
    )
    values = [
        value.strip()
        for value in command_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        ).split(",")
    ]
    if len(values) != 9:
        raise RuntimeError(f"unexpected telemetry: {values}")
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


def check_temperature(record: dict[str, Any]) -> None:
    if record["temperature_c"] >= THERMAL_STOP_C:
        raise RuntimeError(
            f"thermal stop at {record['temperature_c']} C "
            f"(limit {THERMAL_STOP_C} C)"
        )


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


def burgers_flux(state: torch.Tensor) -> torch.Tensor:
    return 0.5 * state.square()


def burgers_speed(state: torch.Tensor) -> torch.Tensor:
    return state


def cell_recompute_axis_rhs(
    state: torch.Tensor,
    dx: float,
    scheme: WENOJS,
    axis: int,
) -> torch.Tensor:
    """Reconstruct both adjacent faces independently from shared split fluxes."""
    physical_flux = burgers_flux(state)
    alpha = torch.amax(torch.abs(burgers_speed(state)))
    positive = 0.5 * (physical_flux + alpha * state)
    negative = 0.5 * (physical_flux - alpha * state)
    current = scheme.reconstruct(
        positive, bias="left", axis=axis
    ) + scheme.reconstruct(negative, bias="right", axis=axis)
    shifted_positive = torch.roll(positive, shifts=1, dims=axis)
    shifted_negative = torch.roll(negative, shifts=1, dims=axis)
    previous = scheme.reconstruct(
        shifted_positive, bias="left", axis=axis
    ) + scheme.reconstruct(shifted_negative, bias="right", axis=axis)
    return (previous - current) / dx


def make_rhs(
    representation: str,
    order: int,
    dimensions: int,
    dx: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    scheme = WENOJS(order)
    axes = tuple(range(dimensions))
    if representation == "face_once":

        def directional(state: torch.Tensor, axis: int) -> torch.Tensor:
            return scheme.rhs(
                state,
                dx,
                burgers_flux,
                burgers_speed,
                axis=axis,
            )

    elif representation == "cell_recompute":

        def directional(state: torch.Tensor, axis: int) -> torch.Tensor:
            return cell_recompute_axis_rhs(state, dx, scheme, axis)

    else:
        raise ValueError(f"unknown representation: {representation}")

    def rhs(state: torch.Tensor) -> torch.Tensor:
        result = directional(state, axes[0])
        for axis in axes[1:]:
            result = result + directional(state, axis)
        return result

    return rhs


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    difference = actual.to(torch.float64) - reference.to(torch.float64)
    scale = max(float(torch.amax(torch.abs(reference))), 1.0)
    return {
        "bitwise_identical": bool(torch.equal(actual, reference)),
        "maximum_normalized": float(torch.amax(torch.abs(difference))) / scale,
        "rms_normalized": float(torch.sqrt(torch.mean(difference.square()))) / scale,
    }


def threshold(dtype: torch.dtype, compiled: bool) -> tuple[float, float]:
    if dtype == torch.float32:
        return (5.0e-5, 5.0e-6) if compiled else (2.0e-5, 2.0e-6)
    return (5.0e-11, 5.0e-12) if compiled else (2.0e-12, 2.0e-13)


def comparison_passes(metrics: dict[str, Any], bounds: tuple[float, float]) -> bool:
    return (
        metrics["maximum_normalized"] <= bounds[0]
        and metrics["rms_normalized"] <= bounds[1]
    )


def output_health(output: torch.Tensor) -> dict[str, Any]:
    epsilon = torch.finfo(output.dtype).eps
    absolute_sum = torch.sum(torch.abs(output), dtype=torch.float64)
    conservation = torch.abs(torch.sum(output, dtype=torch.float64))
    bound = 32.0 * epsilon * absolute_sum
    return {
        "finite": bool(torch.isfinite(output).all()),
        "conservation_absolute": float(conservation),
        "conservation_bound": float(bound),
        "conservation_passed": bool(conservation <= bound),
        "checksum_float64": float(torch.sum(output, dtype=torch.float64)),
        "maximum_absolute": float(torch.amax(torch.abs(output))),
    }


def compile_function(
    function: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], torch.Tensor, dict[str, Any]]:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(function, fullgraph=True)
    started = time.perf_counter()
    output = compiled(state)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    stats = torch._dynamo.utils.counters.get("stats", {})
    graph_breaks = torch._dynamo.utils.counters.get("graph_break", {})
    record = {
        "first_call_wall_seconds": elapsed,
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "calls_captured": int(stats.get("calls_captured", 0)),
        "graph_break_count": int(sum(graph_breaks.values())),
        "graph_break_reasons": {str(key): int(value) for key, value in graph_breaks.items()},
    }
    return compiled, output, record


def event_call(
    function: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor
) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function(state)
    end.record()
    end.synchronize()
    return start.elapsed_time(end), output


def incremental_peak_bytes(
    function: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor
) -> int:
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = function(state)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    return max(0, peak - baseline)


def quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def sample_statistics(values: list[float]) -> dict[str, Any]:
    median = statistics.median(values)
    return {
        "values": values,
        "count": len(values),
        "median": median,
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "median_absolute_deviation": statistics.median(
            abs(value - median) for value in values
        ),
    }


def paired_analysis(
    blocks: list[dict[str, Any]], rng: random.Random
) -> dict[str, Any]:
    face = [block["milliseconds"]["face_once"] for block in blocks]
    cell = [block["milliseconds"]["cell_recompute"] for block in blocks]
    ratios = [left / right for left, right in zip(face, cell, strict=True)]
    bootstrap_medians = []
    for _ in range(BOOTSTRAPS):
        resample = [ratios[rng.randrange(len(ratios))] for _ in ratios]
        bootstrap_medians.append(statistics.median(resample))
    bootstrap_medians.sort()
    ratio_stats = sample_statistics(ratios)
    ratio_stats["bootstrap_median_95_ci"] = [
        quantile(bootstrap_medians, 0.025),
        quantile(bootstrap_medians, 0.975),
    ]
    median = ratio_stats["median"]
    lower, upper = ratio_stats["bootstrap_median_95_ci"]
    if median < 0.95 and upper < 1.0:
        decision = "face_once_win"
    elif median > 1.05 and lower > 1.0:
        decision = "cell_recompute_win"
    else:
        decision = "unresolved"
    return {
        "face_once_milliseconds": sample_statistics(face),
        "cell_recompute_milliseconds": sample_statistics(cell),
        "paired_face_over_cell_ratio": ratio_stats,
        "decision": decision,
    }


def benchmark_mode(
    functions: dict[str, Callable[[torch.Tensor], torch.Tensor]],
    state: torch.Tensor,
    order_rng: random.Random,
    bootstrap_rng: random.Random,
) -> dict[str, Any]:
    for representation in REPRESENTATIONS:
        for _ in range(WARMUPS):
            _, output = event_call(functions[representation], state)
            del output
    torch.cuda.synchronize()
    memory = {
        representation: incremental_peak_bytes(functions[representation], state)
        for representation in REPRESENTATIONS
    }
    blocks = []
    for repetition in range(REPETITIONS):
        order = list(REPRESENTATIONS)
        order_rng.shuffle(order)
        before = telemetry()
        check_temperature(before)
        milliseconds = {}
        for representation in order:
            elapsed, output = event_call(functions[representation], state)
            milliseconds[representation] = elapsed
            del output
        after = telemetry()
        check_temperature(after)
        blocks.append(
            {
                "repetition": repetition,
                "order": order,
                "milliseconds": milliseconds,
                "telemetry_before": before,
                "telemetry_after": after,
            }
        )
    return {
        "warmups_per_representation": WARMUPS,
        "randomized_pair_blocks": REPETITIONS,
        "incremental_peak_allocated_bytes": memory,
        "blocks": blocks,
        "analysis": paired_analysis(blocks, bootstrap_rng),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def machine_record() -> dict[str, Any]:
    return {
        "hostname": command_output(["hostname"]),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "driver": command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def run_configuration(
    configuration: dict[str, Any],
    order_rng: random.Random,
    bootstrap_rng: random.Random,
) -> dict[str, Any]:
    order = configuration["order"]
    dimensions = configuration["dimensions"]
    n = configuration["n"]
    dtype = getattr(torch, configuration["dtype"])
    dx = 2.0 * math.pi / n
    cpu_state = smooth_input(n, dimensions, dtype)
    input_sha256 = sha256_bytes(cpu_state.numpy().tobytes(order="C"))
    state = cpu_state.to(device="cuda")
    del cpu_state
    functions = {
        representation: make_rhs(representation, order, dimensions, dx)
        for representation in REPRESENTATIONS
    }
    eager_outputs = {
        representation: functions[representation](state)
        for representation in REPRESENTATIONS
    }
    torch.cuda.synchronize()
    parity = error_metrics(eager_outputs["cell_recompute"], eager_outputs["face_once"])
    parity_bounds = threshold(dtype, compiled=False)
    eager_health = {
        representation: output_health(eager_outputs[representation])
        for representation in REPRESENTATIONS
    }
    gate_passed = comparison_passes(parity, parity_bounds) and all(
        item["finite"] and item["conservation_passed"]
        for item in eager_health.values()
    )
    record: dict[str, Any] = {
        **configuration,
        "shape": list(state.shape),
        "cells": state.numel(),
        "input_sha256": input_sha256,
        "eager_representation_parity": {
            "metrics": parity,
            "bounds": {"maximum_normalized": parity_bounds[0], "rms_normalized": parity_bounds[1]},
            "passed": comparison_passes(parity, parity_bounds),
        },
        "eager_health": eager_health,
        "passed_precompile_gate": gate_passed,
    }
    if not gate_passed:
        record["status"] = "failed_precompile_gate"
        return record

    compiled_functions = {}
    compiled_outputs = {}
    compile_records = {}
    for representation in REPRESENTATIONS:
        compiled, output, compile_record = compile_function(
            functions[representation], state
        )
        compiled_functions[representation] = compiled
        compiled_outputs[representation] = output
        compile_records[representation] = compile_record
    compiled_bounds = threshold(dtype, compiled=True)
    compiled_parity = {
        representation: {
            "metrics": error_metrics(
                compiled_outputs[representation], eager_outputs[representation]
            ),
            "bounds": {
                "maximum_normalized": compiled_bounds[0],
                "rms_normalized": compiled_bounds[1],
            },
        }
        for representation in REPRESENTATIONS
    }
    for item in compiled_parity.values():
        item["passed"] = comparison_passes(item["metrics"], compiled_bounds)
    compiled_cross = error_metrics(
        compiled_outputs["cell_recompute"], compiled_outputs["face_once"]
    )
    compiled_health = {
        representation: output_health(compiled_outputs[representation])
        for representation in REPRESENTATIONS
    }
    compile_gate = all(
        compile_records[representation]["unique_graphs"] == 1
        and compile_records[representation]["graph_break_count"] == 0
        and compiled_parity[representation]["passed"]
        and compiled_health[representation]["finite"]
        and compiled_health[representation]["conservation_passed"]
        for representation in REPRESENTATIONS
    )
    record.update(
        {
            "compilation": compile_records,
            "compiled_versus_eager": compiled_parity,
            "compiled_representation_parity": compiled_cross,
            "compiled_health": compiled_health,
            "passed_compile_gate": compile_gate,
        }
    )
    if not compile_gate:
        record["status"] = "failed_compile_gate"
        return record

    del eager_outputs, compiled_outputs
    torch.cuda.synchronize()
    record["timing"] = {
        "eager": benchmark_mode(
            functions, state, order_rng, bootstrap_rng
        ),
        "compiled": benchmark_mode(
            compiled_functions, state, order_rng, bootstrap_rng
        ),
    }
    record["status"] = "complete"
    del state, functions, compiled_functions
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise RuntimeError("output exists; refusing to overwrite")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this execution context")

    order_rng = random.Random(SEED)
    bootstrap_rng = random.Random(SEED + 1)
    records = []
    payload: dict[str, Any] = {
        "schema": "gradflow-face-ownership-screen-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_backend_admitted": False,
        "canonical_source_changed": False,
        "protocol": {
            "warmups_per_representation": WARMUPS,
            "randomized_pair_blocks": REPETITIONS,
            "random_seed": SEED,
            "bootstrap_resamples": BOOTSTRAPS,
            "thermal_stop_c": THERMAL_STOP_C,
            "representations": list(REPRESENTATIONS),
            "modes": list(MODES),
            "configurations": configurations(),
        },
        "machine": machine_record(),
        "configurations": records,
    }
    for index, configuration in enumerate(configurations()):
        try:
            record = run_configuration(configuration, order_rng, bootstrap_rng)
        except (torch.OutOfMemoryError, RuntimeError) as error:
            torch._dynamo.reset()
            torch.cuda.empty_cache()
            record = {
                **configuration,
                "status": "recorded_failure",
                "exception_type": type(error).__name__,
                "exception": str(error),
            }
        records.append(record)
        payload["completed_configurations"] = index + 1
        write_json(arguments.output, payload)
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "total": len(configurations()),
                    "configuration": configuration,
                    "status": record["status"],
                    "decisions": {
                        mode: record.get("timing", {})
                        .get(mode, {})
                        .get("analysis", {})
                        .get("decision")
                        for mode in MODES
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    payload["complete"] = True
    payload["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(arguments.output, payload)


if __name__ == "__main__":
    main()
