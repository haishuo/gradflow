#!/usr/bin/env python3
"""Measure one isolated, device-resident Phase-4R CUDA replicate."""

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
import subprocess
import time
import traceback
from typing import Any, Callable

import torch
import torch._inductor.metrics as inductor_metrics

from problem import METHOD_IDS, conservation, projected_state, step_function


WARMUPS = 10
REPETITIONS = 50


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
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "mean_seconds": mean,
        "minimum_seconds": ordered[0],
        "maximum_seconds": ordered[-1],
        "q1_seconds": quantile(ordered, 0.25),
        "q3_seconds": quantile(ordered, 0.75),
        "median_absolute_deviation_seconds": statistics.median(
            abs(value - median) for value in samples
        ),
        "coefficient_of_variation": statistics.pstdev(samples) / mean,
    }


def measure_cuda_events(
    call: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor
) -> tuple[dict[str, Any], torch.Tensor]:
    output = state
    for _ in range(WARMUPS):
        output = call(state)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPETITIONS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPETITIONS)]
    for start, end in zip(starts, ends):
        start.record()
        output = call(state)
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) * 1.0e-3 for start, end in zip(starts, ends)]
    record = sample_statistics(samples)
    record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return record, output


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
    for path in sorted(cache.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(cache))
        item: dict[str, Any] = {
            "relative_path": relative,
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "suffix": path.suffix,
        }
        if path.suffix in {".py", ".cpp", ".cu"}:
            source = path.read_text(errors="replace")
            item["lines"] = len(source.splitlines())
            item["text_counts"] = {
                "triton_jit": source.count("triton.jit"),
                "triton_heuristics": source.count("@triton_heuristics"),
                "tl_load": source.count("tl.load"),
                "tl_store": source.count("tl.store"),
                "num_warps": source.count("num_warps"),
                "cpp_fused": source.count("cpp_fused_"),
            }
        files.append(item)
    text_files = [item for item in files if "text_counts" in item]
    return {
        "status": "recorded",
        "file_count": len(files),
        "total_cache_bytes": sum(item["bytes"] for item in files),
        "text_file_count": len(text_files),
        "text_total_bytes": sum(item["bytes"] for item in text_files),
        "files": files,
    }


def device_environment() -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    driver = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=False,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver or None,
        "device": torch.cuda.get_device_name(0),
        "device_uuid": str(getattr(properties, "uuid", "unknown")),
        "device_total_memory_bytes": properties.total_memory,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "multiprocessor_count": properties.multi_processor_count,
    }


def execute(method: str, cells: int, replicate: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to the worker")
    torch.cuda.set_device(0)
    cpu_state = projected_state(method, 3, cells)
    state = cpu_state.cuda()
    eager_step, _ = step_function(method, 3, cells)
    cpu_reference = eager_step(cpu_state)
    gpu_eager_reference = eager_step(state)
    torch.cuda.synchronize()

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
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    compiled_reference = compiled_step(state)
    torch.cuda.synchronize()
    first_call_seconds = (time.perf_counter_ns() - started) * 1.0e-9
    metrics = compiler_metrics()
    cache = cache_evidence()

    eager_stats, eager_output = measure_cuda_events(eager_step, state)
    compiled_stats, compiled_output = measure_cuda_events(compiled_step, state)
    cpu_cuda_difference = float(
        torch.max(torch.abs(gpu_eager_reference.cpu() - cpu_reference))
    )
    compiled_eager_difference = float(
        torch.max(torch.abs(compiled_reference - gpu_eager_reference))
    )
    eager_repeat = float(torch.max(torch.abs(eager_output - gpu_eager_reference)))
    compiled_repeat = float(
        torch.max(torch.abs(compiled_output - compiled_reference))
    )
    mass_change, mass_bound, mass_passed = conservation(
        state, compiled_output, 3, cells
    )
    finite = bool(torch.isfinite(compiled_output).all())
    eligible = (
        explanation.graph_count == 1
        and explanation.graph_break_count == 0
        and cpu_cuda_difference <= 2.0e-11
        and compiled_eager_difference <= 2.0e-11
        and eager_repeat == 0.0
        and compiled_repeat == 0.0
        and finite
        and mass_passed
        and compiled_output.shape == state.shape
        and compiled_output.dtype == state.dtype
        and compiled_output.device == state.device
    )
    return {
        "status": "completed",
        "method": method,
        "formulation_id": METHOD_IDS[method],
        "dimension": 3,
        "cells_per_axis": cells,
        "logical_cells": cells**3,
        "replicate": replicate,
        "device": "cuda:0",
        "dtype": "float64",
        "resident_timed_region": True,
        "timing_method": "cuda_events",
        "controls": {"warmups": WARMUPS, "repetitions": REPETITIONS},
        "environment": device_environment(),
        "graph": graph_record,
        "correctness": {
            "cpu_eager_gpu_eager_maximum_absolute_difference": cpu_cuda_difference,
            "compiled_eager_maximum_absolute_difference": compiled_eager_difference,
            "eager_repeat_maximum_absolute_difference": eager_repeat,
            "compiled_repeat_maximum_absolute_difference": compiled_repeat,
            "finite": finite,
            "shape_preserved": compiled_output.shape == state.shape,
            "dtype_preserved": compiled_output.dtype == state.dtype,
            "device_preserved": compiled_output.device == state.device,
            "conservation_mass_change": mass_change,
            "conservation_bound": mass_bound,
            "conservation_passed": mass_passed,
        },
        "timing": {
            "eager": eager_stats,
            "compiled": compiled_stats,
            "first_compiled_call_seconds": first_call_seconds,
        },
        "compiler_metrics": metrics,
        "cache_evidence": cache,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
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
        raise SystemExit(f"refusing existing output: {arguments.output}")
    try:
        payload = execute(arguments.method, arguments.cells, arguments.replicate)
    except Exception as error:  # failures are experimental evidence
        payload = {
            "status": "failed",
            "method": arguments.method,
            "cells_per_axis": arguments.cells,
            "replicate": arguments.replicate,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
