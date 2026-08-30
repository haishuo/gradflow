#!/usr/bin/env python3
"""One isolated GradFlow U4-C qualification or resident-timing worker."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch

from gradflow import weno5_rhs


WARMUPS = 5
SAMPLES = 20


def state(size: int, device: str, path: Path | None) -> torch.Tensor:
    if path is not None:
        values = np.fromfile(path, dtype=np.float64)
        if values.shape != (size,):
            raise RuntimeError("frozen U4-C input shape mismatch")
        return torch.from_numpy(values.copy()).to(device)
    x = torch.arange(size, dtype=torch.float64) / size
    return (
        0.4
        + torch.sin(2.0 * math.pi * 37.0 * x)
        + 0.1 * torch.cos(2.0 * math.pi * 91.0 * x)
    ).to(device)


def rhs(values: torch.Tensor, spacing: float) -> torch.Tensor:
    return weno5_rhs(values, spacing, lambda value: value, alpha=1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=("qualify", "resident"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    values = state(args.size, args.device, args.input)
    spacing = 1.0 / args.size

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(
        lambda current: rhs(current, spacing), fullgraph=True, dynamic=False
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    first_started = time.perf_counter()
    output = compiled(values)
    if args.device == "cuda":
        torch.cuda.synchronize()
    first_seconds = time.perf_counter() - first_started

    stats = torch._dynamo.utils.counters.get("stats", {})
    breaks = torch._dynamo.utils.counters.get("graph_break", {})
    graph = {
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "graph_break_count": int(sum(breaks.values())),
    }
    samples: list[float] = []
    if args.mode == "resident":
        for _ in range(WARMUPS):
            output = compiled(values)
            if args.device == "cuda":
                torch.cuda.synchronize()
        for _ in range(SAMPLES):
            if args.device == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = compiled(values)
                end.record()
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            else:
                started = time.perf_counter_ns()
                output = compiled(values)
                samples.append((time.perf_counter_ns() - started) / 1.0e6)

    host = output.detach().cpu().numpy()
    if args.output is not None:
        host.tofile(args.output)
    payload = {
        "schema": "gradflow.academic_u4c.gradflow_worker.v1",
        "status": "complete",
        "size": args.size,
        "device": args.device,
        "mode": args.mode,
        "dtype": "float64",
        "first_call_seconds": first_seconds,
        "graph": graph,
        "samples_milliseconds": samples,
        "finite": bool(np.all(np.isfinite(host))),
        "checksum_float64": float(np.sum(host, dtype=np.float64)),
        "maximum_absolute": float(np.max(np.abs(host))),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None
        ),
    }
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
