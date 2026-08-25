#!/usr/bin/env python3
"""Measure one CPU-preparation/H2D/D2H pass for existing crossover records."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from shu_euler_torch import cfl_timestep, periodic_vortex


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    benchmark = json.loads(arguments.input.read_text())
    dimension = benchmark["dimension"]
    records = []
    for benchmark_record in benchmark["records"]:
        size = benchmark_record["size"]
        intervals = (size,) * dimension

        started = time.perf_counter()
        cpu_state, spacing = periodic_vortex(intervals, device="cpu")
        cpu_dt = torch.minimum(
            cfl_timestep(cpu_state, spacing, 0.1),
            torch.tensor(0.001, dtype=torch.float32),
        )
        cpu_preparation_seconds = time.perf_counter() - started

        torch.cuda.synchronize()
        started = time.perf_counter()
        gpu_state = cpu_state.to("cuda")
        gpu_dt = cpu_dt.to("cuda")
        torch.cuda.synchronize()
        host_to_device_seconds = time.perf_counter() - started

        torch.cuda.synchronize()
        started = time.perf_counter()
        returned_state = gpu_state.cpu()
        torch.cuda.synchronize()
        device_to_host_seconds = time.perf_counter() - started
        if returned_state.shape != cpu_state.shape or not bool(torch.isfinite(gpu_dt)):
            raise RuntimeError("transfer validation failed")

        transfer_inclusive_estimate = (
            cpu_preparation_seconds
            + host_to_device_seconds
            + benchmark_record["gpu_step_wall_seconds"]
            + device_to_host_seconds
        )
        record = {
            "dimension": dimension,
            "size": size,
            "cpu_preparation_seconds": cpu_preparation_seconds,
            "host_to_device_seconds": host_to_device_seconds,
            "device_to_host_seconds": device_to_host_seconds,
            "gpu_step_wall_seconds_from_benchmark": benchmark_record[
                "gpu_step_wall_seconds"
            ],
            "transfer_inclusive_estimate_seconds": transfer_inclusive_estimate,
            "fortran_process_seconds": benchmark_record[
                "fortran_process_seconds"
            ],
            "fortran_over_transfer_inclusive_estimate": benchmark_record[
                "fortran_process_seconds"
            ]
            / transfer_inclusive_estimate,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del cpu_state, gpu_state, returned_state, cpu_dt, gpu_dt
        torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "method": "sum of separately measured one-shot CPU preparation, H2D, recorded GPU step, and D2H",
        "source_benchmark": str(arguments.input),
        "records": records,
    }
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
