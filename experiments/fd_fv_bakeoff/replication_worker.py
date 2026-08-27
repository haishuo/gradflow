#!/usr/bin/env python3
"""Measure one isolated CPU Phase-4R method/size replicate."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time
from typing import Any, Callable

import torch
import torch._inductor.metrics as inductor_metrics

from problem import METHOD_IDS, projected_state, step_function


PRIMARY_THREADS = 6
EAGER_WARMUPS = 10
EAGER_REPETITIONS = 30
COMPILED_WARMUPS = 10
COMPILED_REPETITIONS = 50
THREAD_COUNTS = (1, 2, 3, 6, 12)
THREAD_WARMUPS = 5
THREAD_REPETITIONS = 30


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return (1.0 - weight) * ordered[lower] + weight * ordered[upper]


def sample_statistics(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    median = statistics.median(samples)
    mean = statistics.fmean(samples)
    deviations = [abs(value - median) for value in samples]
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "mean_seconds": mean,
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
        "q1_seconds": quantile(ordered, 0.25),
        "q3_seconds": quantile(ordered, 0.75),
        "median_absolute_deviation_seconds": statistics.median(deviations),
        "coefficient_of_variation": (
            statistics.pstdev(samples) / mean if mean else 0.0
        ),
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


def metric_value(name: str) -> Any:
    value = getattr(inductor_metrics, name, None)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [
            dataclasses.asdict(item) if dataclasses.is_dataclass(item) else str(item)
            for item in value
        ]
    return str(value)


def compiler_metrics() -> dict[str, Any]:
    names = (
        "generated_kernel_count",
        "generated_cpp_vec_kernel_count",
        "ir_nodes_pre_fusion",
        "num_bytes_accessed",
        "cpp_outer_loop_fused_inner_counts",
        "num_loop_reordering",
        "num_auto_chunking",
        "parallel_reduction_count",
        "cpp_to_dtype_count",
    )
    return {name: metric_value(name) for name in names}


def cache_evidence() -> dict[str, Any]:
    cache_text = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not cache_text:
        return {"status": "unavailable", "files": []}
    cache = Path(cache_text)
    files = []
    for path in sorted(cache.rglob("*.cpp")):
        source = path.read_text(errors="replace")
        files.append(
            {
                "relative_path": str(path.relative_to(cache)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "lines": len(source.splitlines()),
                "text_counts": {
                    "parallel_for": source.count("parallel_for"),
                    "openmp_pragma": source.count("#pragma omp"),
                    "vectorized_type": source.count("Vectorized<"),
                    "loadu": source.count("loadu"),
                    "store": source.count("store"),
                    "cpp_fused": source.count("cpp_fused_"),
                },
            }
        )
    all_files = [path for path in cache.rglob("*") if path.is_file()]
    return {
        "status": "recorded",
        "cpp_file_count": len(files),
        "cpp_total_bytes": sum(item["bytes"] for item in files),
        "cpp_total_lines": sum(item["lines"] for item in files),
        "total_cache_bytes": sum(path.stat().st_size for path in all_files),
        "files": files,
    }


def profile_call(
    call: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor
) -> list[dict[str, Any]]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        call(state)
    return sorted(
        (
            {
                "key": event.key,
                "count": event.count,
                "cpu_time_total_us": event.cpu_time_total,
                "self_cpu_time_total_us": event.self_cpu_time_total,
            }
            for event in profiler.key_averages()
        ),
        key=lambda item: item["key"],
    )


def thread_sweep(
    eager: Callable[[torch.Tensor], torch.Tensor],
    compiled: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
) -> dict[str, Any]:
    records = {}
    for threads in THREAD_COUNTS:
        torch.set_num_threads(threads)
        eager_stats, _ = measure(
            eager,
            state,
            warmups=THREAD_WARMUPS,
            repetitions=THREAD_REPETITIONS,
        )
        compiled_stats, _ = measure(
            compiled,
            state,
            warmups=THREAD_WARMUPS,
            repetitions=THREAD_REPETITIONS,
        )
        records[str(threads)] = {
            "eager": eager_stats,
            "compiled": compiled_stats,
        }
    torch.set_num_threads(PRIMARY_THREADS)
    return records


def execute(method: str, cells: int, replicate: int) -> dict[str, Any]:
    torch.set_num_threads(PRIMARY_THREADS)
    torch.set_num_interop_threads(1)
    state = projected_state(method, 3, cells)
    eager_step, _ = step_function(method, 3, cells)
    eager_reference = eager_step(state)

    torch._dynamo.reset()
    explanation = torch._dynamo.explain(eager_step)(state)
    graph = explanation.graphs[0].graph
    graph_record = {
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "operation_count": explanation.op_count,
        "fx_node_count": sum(1 for _ in graph.nodes),
        "break_reasons": [str(reason) for reason in explanation.break_reasons],
    }

    torch._dynamo.reset()
    inductor_metrics.reset()
    compiled_step = torch.compile(eager_step, fullgraph=True, dynamic=False)
    started = time.perf_counter_ns()
    compiled_reference = compiled_step(state)
    first_call_seconds = (time.perf_counter_ns() - started) * 1.0e-9
    metrics = compiler_metrics()
    cache = cache_evidence()
    parity = float(torch.max(torch.abs(compiled_reference - eager_reference)))

    eager_stats, eager_output = measure(
        eager_step,
        state,
        warmups=EAGER_WARMUPS,
        repetitions=EAGER_REPETITIONS,
    )
    compiled_stats, compiled_output = measure(
        compiled_step,
        state,
        warmups=COMPILED_WARMUPS,
        repetitions=COMPILED_REPETITIONS,
    )
    eager_repeat = float(torch.max(torch.abs(eager_output - eager_reference)))
    compiled_repeat = float(
        torch.max(torch.abs(compiled_output - compiled_reference))
    )
    sweep = (
        thread_sweep(eager_step, compiled_step, state) if cells == 27 else None
    )
    torch.set_num_threads(PRIMARY_THREADS)
    profile = profile_call(compiled_step, state)
    eligible = (
        explanation.graph_count == 1
        and explanation.graph_break_count == 0
        and parity <= 2.0e-11
        and eager_repeat == 0.0
        and compiled_repeat == 0.0
        and bool(torch.isfinite(compiled_reference).all())
        and compiled_reference.shape == state.shape
        and compiled_reference.dtype == state.dtype
        and compiled_reference.device == state.device
    )
    return {
        "status": "completed",
        "method": method,
        "formulation_id": METHOD_IDS[method],
        "dimension": 3,
        "cells_per_axis": cells,
        "logical_cells": cells**3,
        "replicate": replicate,
        "device": "cpu",
        "dtype": "float64",
        "graph": graph_record,
        "correctness": {
            "compiled_eager_maximum_absolute_difference": parity,
            "eager_repeat_maximum_absolute_difference": eager_repeat,
            "compiled_repeat_maximum_absolute_difference": compiled_repeat,
            "finite": bool(torch.isfinite(compiled_reference).all()),
            "shape_preserved": compiled_reference.shape == state.shape,
            "dtype_preserved": compiled_reference.dtype == state.dtype,
            "device_preserved": compiled_reference.device == state.device,
        },
        "timing": {
            "eager": eager_stats,
            "compiled": compiled_stats,
            "first_compiled_call_seconds": first_call_seconds,
        },
        "compiler_metrics": metrics,
        "cache_evidence": cache,
        "thread_sweep": sweep,
        "compiled_profile": profile,
        "controls": {
            "primary_intraop_threads": PRIMARY_THREADS,
            "interop_threads": torch.get_num_interop_threads(),
            "eager_warmups": EAGER_WARMUPS,
            "eager_repetitions": EAGER_REPETITIONS,
            "compiled_warmups": COMPILED_WARMUPS,
            "compiled_repetitions": COMPILED_REPETITIONS,
            "thread_counts": THREAD_COUNTS if cells == 27 else None,
            "thread_warmups": THREAD_WARMUPS if cells == 27 else None,
            "thread_repetitions": THREAD_REPETITIONS if cells == 27 else None,
            "visible_logical_cpus": os.cpu_count(),
            "process_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
        },
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "eligible": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    try:
        result = execute(arguments.method, arguments.cells, arguments.replicate)
    except Exception as error:
        result = {
            "status": "failed",
            "method": arguments.method,
            "cells_per_axis": arguments.cells,
            "replicate": arguments.replicate,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
