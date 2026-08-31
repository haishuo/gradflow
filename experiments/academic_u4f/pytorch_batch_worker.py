#!/usr/bin/env python3
"""Isolated PyTorch/TorchInductor worker for the frozen U4-F batch surface."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from gradflow import weno5_rhs


WARMUPS = 5
SAMPLES = 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=("qualify", "resident"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    values_array = np.fromfile(args.input, dtype=np.float64)
    if values_array.shape != (args.batch * args.size,):
        raise RuntimeError("frozen U4-F input shape mismatch")
    values_cpu = torch.from_numpy(values_array.copy()).reshape(args.batch, args.size)
    values = values_cpu.to(args.device)
    spacing = 1.0 / args.size

    def rhs(current: torch.Tensor) -> torch.Tensor:
        return weno5_rhs(current, spacing, lambda value: value, alpha=1.0)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(rhs, fullgraph=True, dynamic=False)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = compiled(values)
    if args.device == "cuda":
        torch.cuda.synchronize()
    first_call_seconds = time.perf_counter() - started

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
                sample_started = time.perf_counter_ns()
                output = compiled(values)
                samples.append((time.perf_counter_ns() - sample_started) / 1.0e6)

    host = output.detach().cpu().numpy()
    if args.output is not None:
        host.tofile(args.output)
    payload = {
        "schema": "gradflow.academic_u4f.pytorch_worker.v1",
        "status": "complete",
        "size": args.size,
        "batch": args.batch,
        "device": args.device,
        "mode": args.mode,
        "dtype": "float64",
        "first_call_seconds": first_call_seconds,
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
