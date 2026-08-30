#!/usr/bin/env python3
"""Reduce the paired G6 Nsight Compute Basic CSV exports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any


FACE_IDS = (5, 10, 15)
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


def duration_ms(metric: dict[str, Any]) -> float:
    factors = {"ns": 1.0e-6, "us": 1.0e-3, "ms": 1.0}
    return metric["value"] * factors[metric["unit"]]


def summarize_lane(source: Path) -> dict[str, Any]:
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    launches = []
    for stage, kernel_id in enumerate(FACE_IDS, start=1):
        selected = [row for row in rows if int(row["ID"]) == kernel_id]
        values = {
            row["Metric Name"]: {
                "value": numeric(row["Metric Value"]),
                "unit": row["Metric Unit"],
            }
            for row in selected
            if row["Metric Name"] in METRICS
        }
        if set(values) != set(METRICS):
            missing = sorted(set(METRICS) - set(values))
            raise RuntimeError(f"missing face metrics for ID {kernel_id}: {missing}")
        launches.append(
            {
                "id": kernel_id,
                "stage": stage,
                "duration_ms": duration_ms(values["Duration"]),
                "metrics": values,
            }
        )
    first = launches[0]["metrics"]
    return {
        "source": source.name,
        "face_launches": launches,
        "face_summary": {
            "total_duration_ms": sum(item["duration_ms"] for item in launches),
            "median_duration_ms": statistics.median(
                item["duration_ms"] for item in launches
            ),
            "median_memory_throughput_percent": statistics.median(
                item["metrics"]["Memory Throughput"]["value"] for item in launches
            ),
            "median_dram_throughput_percent": statistics.median(
                item["metrics"]["DRAM Throughput"]["value"] for item in launches
            ),
            "median_l1_tex_throughput_percent": statistics.median(
                item["metrics"]["L1/TEX Cache Throughput"]["value"]
                for item in launches
            ),
            "median_l2_throughput_percent": statistics.median(
                item["metrics"]["L2 Cache Throughput"]["value"]
                for item in launches
            ),
            "median_compute_sm_throughput_percent": statistics.median(
                item["metrics"]["Compute (SM) Throughput"]["value"]
                for item in launches
            ),
            "median_achieved_occupancy_percent": statistics.median(
                item["metrics"]["Achieved Occupancy"]["value"]
                for item in launches
            ),
        },
        "launch_invariants": {
            "registers_per_thread": first["Registers Per Thread"]["value"],
            "theoretical_occupancy_percent": first["Theoretical Occupancy"]["value"],
            "register_block_limit": first["Block Limit Registers"]["value"],
            "shared_memory_block_limit": first["Block Limit Shared Mem"]["value"],
            "warp_block_limit": first["Block Limit Warps"]["value"],
        },
    }


def run(frozen_source: Path, candidate_source: Path) -> dict[str, Any]:
    frozen = summarize_lane(frozen_source)
    candidate = summarize_lane(candidate_source)
    frozen_summary = frozen["face_summary"]
    candidate_summary = candidate["face_summary"]
    return {
        "schema": "gradflow-g6-ncu-basic-comparison-v1",
        "lanes": {"frozen_r6q": frozen, "b256_r112": candidate},
        "candidate_over_frozen": {
            "face_total_duration_ratio": candidate_summary["total_duration_ms"]
            / frozen_summary["total_duration_ms"],
            "achieved_occupancy_difference_percentage_points": (
                candidate_summary["median_achieved_occupancy_percent"]
                - frozen_summary["median_achieved_occupancy_percent"]
            ),
            "compute_sm_throughput_difference_percentage_points": (
                candidate_summary["median_compute_sm_throughput_percent"]
                - frozen_summary["median_compute_sm_throughput_percent"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frozen_source", type=Path)
    parser.add_argument("candidate_source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = run(arguments.frozen_source, arguments.candidate_source)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
