#!/usr/bin/env python3
"""One fresh-process A2 start-to-finish deployment observation."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import torch

from benchmark_worker import make_problem


def output_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], torch.Tensor)
    ):
        return value[0]
    raise TypeError(f"unexpected output type: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=(5, 15), required=True)
    parser.add_argument("--dimensions", type=int, choices=(1, 3), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument(
        "--lane", choices=("cpu_compiled", "cuda_compiled", "cuda_aot"), required=True
    )
    parser.add_argument("--package")
    arguments = parser.parse_args()
    if arguments.lane.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.set_num_threads(6 if arguments.lane == "cpu_compiled" else 1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    state_cpu, function = make_problem(
        "scalar", arguments.order, torch.float32, arguments.dimensions, arguments.size
    )
    if arguments.lane == "cpu_compiled":
        state = state_cpu
        call = torch.compile(function, fullgraph=True, dynamic=False)
    elif arguments.lane == "cuda_compiled":
        state = state_cpu.cuda()
        call = torch.compile(function, fullgraph=True, dynamic=False)
    else:
        if arguments.package is None:
            raise SystemExit("cuda_aot requires --package")
        state = state_cpu.cuda()
        call = torch._inductor.aoti_load_package(arguments.package)
    if state.device.type == "cuda":
        torch.cuda.synchronize()
    execution_started = time.perf_counter()
    with torch.inference_mode():
        output = output_tensor(call(state))
    if state.device.type == "cuda":
        torch.cuda.synchronize()
        output = output.cpu()
        torch.cuda.synchronize()
    execution_and_return_seconds = time.perf_counter() - execution_started
    print(
        json.dumps(
            {
                "schema": "gradflow-academic-a2-deployment-worker-v1",
                "status": "complete",
                "order": arguments.order,
                "dimensions": arguments.dimensions,
                "n": arguments.size,
                "lane": arguments.lane,
                "after_import_total_seconds": time.perf_counter() - started,
                "execution_and_return_seconds": execution_and_return_seconds,
                "finite": bool(torch.isfinite(output).all()),
                "checksum_float64": float(torch.sum(output, dtype=torch.float64)),
                "maximum_absolute": float(torch.amax(torch.abs(output))),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
