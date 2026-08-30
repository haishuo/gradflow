#!/usr/bin/env python3
"""Qualify and time one prepared A2 package against JIT-compiled CUDA."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from typing import Any, Callable

import torch

from benchmark_worker import (
    BOOTSTRAPS,
    REPETITIONS,
    SEED,
    WARMUPS,
    error_metrics,
    health,
    make_problem,
    quantile,
    statistics_record,
    telemetry,
)


def output_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"unexpected AOT output type: {type(value).__name__}")


def event_call(
    function: Callable[[torch.Tensor], Any], state: torch.Tensor
) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = output_tensor(function(state))
    end.record()
    end.synchronize()
    return start.elapsed_time(end), output


def transfer_call(
    function: Callable[[torch.Tensor], Any], state_cpu: torch.Tensor
) -> tuple[float, torch.Tensor]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    state = state_cpu.cuda()
    output = output_tensor(function(state))
    result = output.cpu()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - started) / 1.0e6, result


def paired_analysis(compiled: list[float], aot: list[float], seed: int) -> dict[str, Any]:
    ratios = [aot_value / compiled_value for compiled_value, aot_value in zip(compiled, aot)]
    generator = random.Random(seed)
    medians = []
    for _ in range(BOOTSTRAPS):
        medians.append(statistics.median(ratios[generator.randrange(len(ratios))] for _ in ratios))
    interval = [quantile(medians, 0.025), quantile(medians, 0.975)]
    median = statistics.median(ratios)
    if median < 0.95 and interval[1] < 1.0:
        decision = "aot_win"
    elif median > 1.05 and interval[0] > 1.0:
        decision = "jit_win"
    else:
        decision = "unresolved"
    return {
        "aot_over_jit": {**statistics_record(ratios), "bootstrap_median_95_ci": interval},
        "decision": decision,
    }


def time_pair(
    functions: dict[str, Callable[[torch.Tensor], Any]],
    call: Callable[[Callable[[torch.Tensor], Any], torch.Tensor], tuple[float, torch.Tensor]],
    state: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    for name in ("jit", "aot"):
        for _ in range(WARMUPS):
            call(functions[name], state)
    samples = {"jit": [], "aot": []}
    blocks = []
    generator = random.Random(seed)
    for repetition in range(REPETITIONS):
        order = ["jit", "aot"]
        generator.shuffle(order)
        milliseconds = {}
        for name in order:
            elapsed, output = call(functions[name], state)
            samples[name].append(elapsed)
            milliseconds[name] = elapsed
            del output
        blocks.append({"repetition": repetition, "order": order, "milliseconds": milliseconds})
    return {
        "warmups_per_lane": WARMUPS,
        "randomized_complete_pair_blocks": REPETITIONS,
        "blocks": blocks,
        "lanes": {name: statistics_record(values) for name, values in samples.items()},
        "paired_analysis": paired_analysis(samples["jit"], samples["aot"], seed + 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=(5, 11, 15), required=True)
    parser.add_argument("--package", required=True)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    state_cpu, function = make_problem("scalar", arguments.order, torch.float32, 3, 64)
    reference = function(state_cpu)
    state = state_cpu.cuda()

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    jit = torch.compile(function, fullgraph=True, dynamic=False)
    started = time.perf_counter()
    jit_output = jit(state)
    torch.cuda.synchronize()
    jit_first_seconds = time.perf_counter() - started
    stats = torch._dynamo.utils.counters.get("stats", {})
    breaks = torch._dynamo.utils.counters.get("graph_break", {})

    load_started = time.perf_counter()
    aot = torch._inductor.aoti_load_package(arguments.package)
    load_seconds = time.perf_counter() - load_started
    first_started = time.perf_counter()
    aot_output = output_tensor(aot(state))
    torch.cuda.synchronize()
    aot_first_seconds = time.perf_counter() - first_started

    limits = (5.0e-5, 5.0e-6)
    correctness = {}
    for name, output in (("jit", jit_output), ("aot", aot_output)):
        metrics = error_metrics(output.cpu(), reference)
        lane_health = health(output, "scalar")
        correctness[name] = {
            "comparison": metrics,
            "health": lane_health,
            "admitted": (
                metrics["maximum_normalized"] <= limits[0]
                and metrics["rms_normalized"] <= limits[1]
                and lane_health["finite"]
                and lane_health["conservation_passed"]
            ),
        }
    graph = {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "graph_break_count": int(sum(breaks.values())),
    }
    if not all(item["admitted"] for item in correctness.values()):
        raise RuntimeError(f"AOT/JIT correctness gate failed: {correctness}")
    if graph != {"unique_graphs": 1, "graph_break_count": 0}:
        raise RuntimeError(f"JIT graph gate failed: {graph}")

    functions = {"jit": jit, "aot": aot}
    before = telemetry()
    resident = time_pair(functions, event_call, state, SEED + arguments.order)
    transfer = time_pair(functions, transfer_call, state_cpu, SEED + arguments.order + 100)
    after = telemetry()
    print(
        json.dumps(
            {
                "schema": "gradflow-academic-a2-aot-worker-v1",
                "status": "complete",
                "order": arguments.order,
                "dtype": "float32",
                "dimensions": 3,
                "n": 64,
                "correctness": correctness,
                "jit_graph": graph,
                "jit_first_call_seconds": jit_first_seconds,
                "aot_load_seconds": load_seconds,
                "aot_first_call_after_load_seconds": aot_first_seconds,
                "resident_timing": resident,
                "transfer_inclusive_timing": transfer,
                "telemetry_before": before,
                "telemetry_after": after,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
