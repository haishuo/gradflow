#!/usr/bin/env python3
"""Reduce the G5 Nsight Compute Basic CSV to its frozen pencil metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any


PENCIL_IDS = (3, 5, 7, 9, 11, 13, 15, 17, 19)
METRICS = (
    "Duration",
    "Memory Throughput",
    "DRAM Throughput",
    "L1/TEX Cache Throughput",
    "L2 Cache Throughput",
    "Compute (SM) Throughput",
    "Registers Per Thread",
    "Theoretical Occupancy",
    "Achieved Occupancy",
    "Block Limit Registers",
    "Block Limit Shared Mem",
    "Block Limit Warps",
)


def numeric(value: str) -> float:
    return float(value.replace(",", ""))


def run(source: Path) -> dict[str, Any]:
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    launches = []
    for position, kernel_id in enumerate(PENCIL_IDS):
        selected = [row for row in rows if int(row["ID"]) == kernel_id]
        values = {
            row["Metric Name"]: {
                "value": numeric(row["Metric Value"]),
                "unit": row["Metric Unit"],
            }
            for row in selected
            if row["Metric Name"] in METRICS
        }
        duration = values["Duration"]
        duration_ms = duration["value"] * (0.001 if duration["unit"] == "us" else 1.0)
        launches.append(
            {
                "id": kernel_id,
                "stage": position // 3 + 1,
                "axis": ("x", "y", "z")[position % 3],
                "duration_ms": duration_ms,
                "metrics": values,
            }
        )

    axes = {}
    for axis in ("x", "y", "z"):
        group = [launch for launch in launches if launch["axis"] == axis]
        axes[axis] = {
            "total_duration_ms": sum(item["duration_ms"] for item in group),
            "median_memory_throughput_percent": statistics.median(
                item["metrics"]["Memory Throughput"]["value"] for item in group
            ),
            "median_dram_throughput_percent": statistics.median(
                item["metrics"]["DRAM Throughput"]["value"] for item in group
            ),
            "median_l1_tex_throughput_percent": statistics.median(
                item["metrics"]["L1/TEX Cache Throughput"]["value"]
                for item in group
            ),
            "median_l2_throughput_percent": statistics.median(
                item["metrics"]["L2 Cache Throughput"]["value"] for item in group
            ),
            "median_compute_sm_throughput_percent": statistics.median(
                item["metrics"]["Compute (SM) Throughput"]["value"]
                for item in group
            ),
            "median_achieved_occupancy_percent": statistics.median(
                item["metrics"]["Achieved Occupancy"]["value"] for item in group
            ),
        }
    first = launches[0]["metrics"]
    return {
        "schema": "gradflow-g5-ncu-basic-summary-v1",
        "source": source.name,
        "pencil_launches": launches,
        "axis_summary": axes,
        "launch_invariants": {
            "registers_per_thread": first["Registers Per Thread"]["value"],
            "theoretical_occupancy_percent": first["Theoretical Occupancy"]["value"],
            "register_block_limit": first["Block Limit Registers"]["value"],
            "shared_memory_block_limit": first["Block Limit Shared Mem"]["value"],
            "warp_block_limit": first["Block Limit Warps"]["value"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.source)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.write_text(serialized)


if __name__ == "__main__":
    main()
