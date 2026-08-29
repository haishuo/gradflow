#!/usr/bin/env python3
"""Measure one isolated CPU thread/codegen characterization cell."""

from __future__ import annotations

import argparse
import hashlib
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
from torch._inductor import metrics

from experiments.fd_fv_euler.phase6c_problem import (
    METHOD_IDS,
    fixed_step_function,
    smooth_initial,
    statistics_record,
    tensor_hash,
)


WARMUPS = 10
REPETITIONS = 30


def duration(call: Callable[[], torch.Tensor]) -> tuple[float, torch.Tensor]:
    started = time.perf_counter_ns()
    output = call()
    return (time.perf_counter_ns() - started) * 1.0e-9, output


def samples(
    call: Callable[[], torch.Tensor],
) -> tuple[dict[str, Any], torch.Tensor, list[str]]:
    output = call()
    for _ in range(WARMUPS - 1):
        output = call()
    values = []
    hashes = []
    for _ in range(REPETITIONS):
        seconds, output = duration(call)
        values.append(seconds)
        hashes.append(tensor_hash(output))
    return statistics_record(values), output, hashes


def source_inventory() -> dict[str, Any]:
    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not cache:
        return {"cache_present": False, "files": [], "totals": {}}
    files = []
    totals = {
        "cpp_files": 0,
        "cpp_bytes": 0,
        "cpp_lines": 0,
        "openmp_pragmas": 0,
        "parallel_for_markers": 0,
        "vectorized_markers": 0,
        "gcc_ivdep_pragmas": 0,
    }
    for path in sorted(Path(cache).rglob("*.cpp")):
        content = path.read_text(errors="replace")
        record = {
            "relative_path": str(path.relative_to(cache)),
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "bytes": path.stat().st_size,
            "lines": len(content.splitlines()),
            "openmp_pragmas": content.count("#pragma omp"),
            "parallel_for_markers": content.count("parallel_for"),
            "vectorized_markers": content.count("Vectorized")
            + content.count("at::vec"),
            "gcc_ivdep_pragmas": content.count("#pragma GCC ivdep"),
        }
        files.append(record)
        totals["cpp_files"] += 1
        totals["cpp_bytes"] += record["bytes"]
        totals["cpp_lines"] += record["lines"]
        for name in (
            "openmp_pragmas",
            "parallel_for_markers",
            "vectorized_markers",
            "gcc_ivdep_pragmas",
        ):
            totals[name] += record[name]
    return {"cache_present": True, "files": files, "totals": totals}


def compiler_metrics() -> dict[str, Any]:
    return {
        "generated_kernel_count": metrics.generated_kernel_count,
        "generated_cpp_vec_kernel_count": metrics.generated_cpp_vec_kernel_count,
        "ir_nodes_pre_fusion": metrics.ir_nodes_pre_fusion,
        "num_bytes_accessed": metrics.num_bytes_accessed,
        "cpp_to_dtype_count": metrics.cpp_to_dtype_count,
        "rejected_mix_order_reduction_fusion": (
            metrics.rejected_mix_order_reduction_fusion
        ),
    }


def worker(method: str, cells: int, threads: int, replicate: int) -> dict[str, Any]:
    initial = smooth_initial(method, cells)
    eager_step = fixed_step_function(method, cells)
    eager_statistics, eager_output, eager_hashes = samples(
        lambda: eager_step(initial)
    )
    metrics.reset()
    torch._dynamo.reset()
    compiled_step = torch.compile(eager_step, fullgraph=True, dynamic=False)
    first_seconds, first_output = duration(lambda: compiled_step(initial))
    metrics_after_first = compiler_metrics()
    compiled_statistics, compiled_output, compiled_hashes = samples(
        lambda: compiled_step(initial)
    )
    parity = float(torch.max(torch.abs(compiled_output - eager_output)))
    first_parity = float(torch.max(torch.abs(first_output - eager_output)))
    finite = bool(torch.isfinite(eager_output).all()) and bool(
        torch.isfinite(compiled_output).all()
    )
    deterministic = len(set(eager_hashes)) == 1 and len(set(compiled_hashes)) == 1
    source = source_inventory()
    eligible = (
        finite
        and deterministic
        and parity <= 5.0e-11
        and first_parity <= 5.0e-11
        and eager_output.dtype == compiled_output.dtype == torch.float64
        and eager_output.shape == compiled_output.shape == (3, cells)
    )
    affinity = (
        sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "status": "completed",
        "kind": "cpu_regime",
        "method": method,
        "formulation_id": METHOD_IDS[method],
        "device": "cpu",
        "cells": cells,
        "threads": threads,
        "replicate": replicate,
        "eager": {
            "resident_step": eager_statistics,
            "terminal_hashes": eager_hashes,
        },
        "compiled": {
            "first_call_seconds": first_seconds,
            "resident_step": compiled_statistics,
            "terminal_hashes": compiled_hashes,
        },
        "compiled_eager_maximum_absolute_difference": parity,
        "compiled_first_eager_maximum_absolute_difference": first_parity,
        "finite": finite,
        "deterministic": deterministic,
        "dtype": "float64",
        "shape": [3, cells],
        "compiler_metrics": metrics_after_first,
        "generated_cpp_inventory": source,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "compiler_cache_bytes": sum(
            path.stat().st_size
            for path in Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).rglob("*")
            if path.is_file()
        ),
        "controls": {
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "visible_logical_cpus": os.cpu_count(),
            "process_affinity": affinity,
            "warmups": WARMUPS,
            "repetitions": REPETITIONS,
        },
        "eligible": eligible,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--threads", choices=(1, 2, 4, 6), type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    torch.set_num_threads(arguments.threads)
    torch.set_num_interop_threads(1)
    try:
        result = worker(
            arguments.method,
            arguments.cells,
            arguments.threads,
            arguments.replicate,
        )
    except Exception as error:
        result = {
            "status": "failed",
            "kind": "cpu_regime",
            "method": arguments.method,
            "device": "cpu",
            "cells": arguments.cells,
            "threads": arguments.threads,
            "replicate": arguments.replicate,
            "error_type": type(error).__name__,
            "error": str(error),
            "eligible": False,
        }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
