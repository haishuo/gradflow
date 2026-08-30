#!/usr/bin/env python3
"""Derive compact, deterministic A2 analysis tables from immutable records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ORDERS = (5, 7, 9, 11, 13, 15)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker_id(
    subject: str,
    order: int,
    dtype: str,
    dimensions: int,
    n: int,
    device: str,
) -> str:
    return f"{subject}_o{order}_{dtype}_d{dimensions}_n{n}_{device}"


def worker(
    campaign: dict[str, Any],
    subject: str,
    order: int,
    dtype: str,
    dimensions: int,
    n: int,
    device: str,
) -> dict[str, Any]:
    return campaign["workers"][worker_id(subject, order, dtype, dimensions, n, device)][
        "record"
    ]


def lane_summary(timing: dict[str, Any] | None, lane: str) -> dict[str, Any] | None:
    if timing is None or lane not in timing.get("lanes", {}):
        return None
    record = timing["lanes"][lane]
    return {
        "median_ms": record["median"],
        "mean_ms": record["mean"],
        "mad_ms": record["median_absolute_deviation"],
        "sample_standard_deviation_ms": record["sample_standard_deviation"],
        "count": record["count"],
    }


def pair_summary(timing: dict[str, Any] | None) -> dict[str, Any] | None:
    if timing is None or "paired_analysis" not in timing:
        return None
    comparison = timing["paired_analysis"]["compiled_over_eager"]
    return {
        "compiled_over_eager_median": comparison["median"],
        "bootstrap_median_95_ci": comparison["bootstrap_median_95_ci"],
        "decision": timing["paired_analysis"]["decision"],
    }


def correctness(record: dict[str, Any], lane: str) -> dict[str, Any] | None:
    result = record["correctness"].get(lane)
    if result is None:
        return None
    return {
        "admitted": result["admitted"],
        "comparison": result.get("comparison"),
        "health": result.get("health"),
        "graph": result.get("graph"),
        "error": result.get("error"),
    }


def CPU_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_health": record["reference_health"],
        "untimed_correctness": {
            lane: correctness(record, lane) for lane in ("eager", "compiled")
        },
        "threads": {
            threads: {
                "correctness": {
                    lane: correctness(thread_record, lane)
                    for lane in ("eager", "compiled")
                },
                "resident": {
                    lane: lane_summary(thread_record.get("resident_timing"), lane)
                    for lane in ("eager", "compiled")
                },
                "paired": pair_summary(thread_record.get("resident_timing")),
            }
            for threads, thread_record in record["cpu"].items()
        },
    }


def CUDA_summary(record: dict[str, Any]) -> dict[str, Any]:
    cuda = record["cuda"]
    return {
        "reference_health": record["reference_health"],
        "correctness": {
            lane: {
                "raw": correctness(record, lane),
                "analysis_admitted": bool(
                    record["reference_health"]["conservation_passed"]
                    and record["correctness"][lane]["admitted"]
                ),
            }
            for lane in ("eager", "compiled")
        },
        "first_call_seconds": record["first_call_seconds"],
        "resident": {
            lane: lane_summary(cuda.get("resident_timing"), lane)
            for lane in ("eager", "compiled")
        },
        "resident_paired": pair_summary(cuda.get("resident_timing")),
        "transfer_inclusive": {
            lane: lane_summary(cuda.get("transfer_inclusive_timing"), lane)
            for lane in ("eager", "compiled")
        },
        "transfer_inclusive_paired": pair_summary(
            cuda.get("transfer_inclusive_timing")
        ),
        "memory": cuda["memory"],
    }


def best_cpu(record: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for threads, thread_record in record["cpu"].items():
        for lane in ("eager", "compiled"):
            result = thread_record["correctness"].get(lane)
            if result is None or not result["admitted"]:
                continue
            timing = lane_summary(thread_record.get("resident_timing"), lane)
            if timing is not None:
                candidates.append((timing["median_ms"], lane, int(threads)))
    if not candidates:
        return None
    median, lane, threads = min(candidates)
    return {"median_ms": median, "lane": lane, "threads": threads}


def best_cuda(record: dict[str, Any], endpoint: str) -> dict[str, Any] | None:
    if not record["reference_health"]["conservation_passed"]:
        return None
    timing_key = (
        "resident_timing" if endpoint == "resident" else "transfer_inclusive_timing"
    )
    timing = record["cuda"].get(timing_key)
    candidates = []
    for lane in ("eager", "compiled"):
        if record["correctness"][lane]["admitted"]:
            summary = lane_summary(timing, lane)
            if summary is not None:
                candidates.append((summary["median_ms"], lane))
    if not candidates:
        return None
    median, lane = min(candidates)
    return {"median_ms": median, "lane": lane}


def scale_row(
    campaign: dict[str, Any], order: int, dimensions: int, n: int
) -> dict[str, Any]:
    cpu = worker(campaign, "scalar", order, "float32", dimensions, n, "cpu")
    cuda = worker(campaign, "scalar", order, "float32", dimensions, n, "cuda")
    cpu_best = best_cpu(cpu)
    resident_best = best_cuda(cuda, "resident")
    transfer_best = best_cuda(cuda, "transfer")
    row: dict[str, Any] = {
        "order": order,
        "dimensions": dimensions,
        "n": n,
        "cells": n**dimensions,
        "best_cpu_resident": cpu_best,
        "best_cuda_resident": resident_best,
        "best_cuda_transfer_inclusive": transfer_best,
    }
    if cpu_best is not None and resident_best is not None:
        row["cuda_resident_over_cpu_resident"] = (
            resident_best["median_ms"] / cpu_best["median_ms"]
        )
    if cpu_best is not None and transfer_best is not None:
        row["cuda_transfer_over_cpu_resident"] = (
            transfer_best["median_ms"] / cpu_best["median_ms"]
        )
    return row


def exclusion_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    exclusions = []
    for identifier, wrapper in sorted(campaign["workers"].items()):
        if not wrapper.get("protocol_eligible", True):
            continue
        record = wrapper["record"]
        if record["device"] == "cpu":
            if not record["reference_health"]["conservation_passed"]:
                for lane in ("eager", "compiled"):
                    result = record["correctness"][lane]
                    exclusions.append(
                        {
                            "worker": identifier,
                            "endpoint": f"cpu_{lane}_all_threads",
                            "comparison": result.get("comparison"),
                            "health": result.get("health"),
                            "error": result.get("error"),
                        }
                    )
                continue
            for threads, thread_record in record["cpu"].items():
                for lane in ("eager", "compiled"):
                    result = thread_record["correctness"][lane]
                    if not result["admitted"]:
                        exclusions.append(
                            {
                                "worker": identifier,
                                "endpoint": f"cpu_{lane}_{threads}_threads",
                                "comparison": result.get("comparison"),
                                "health": result.get("health"),
                                "error": result.get("error"),
                            }
                        )
        else:
            if not record["reference_health"]["conservation_passed"]:
                for lane in ("eager", "compiled"):
                    result = record["correctness"][lane]
                    exclusions.append(
                        {
                            "worker": identifier,
                            "endpoint": f"cuda_{lane}_reference_precondition",
                            "comparison": result.get("comparison"),
                            "health": result.get("health"),
                            "error": result.get("error"),
                        }
                    )
                continue
            for lane in ("eager", "compiled"):
                result = record["correctness"][lane]
                if not result["admitted"]:
                    exclusions.append(
                        {
                            "worker": identifier,
                            "endpoint": f"cuda_{lane}",
                            "comparison": result.get("comparison"),
                            "health": result.get("health"),
                            "error": result.get("error"),
                        }
                    )
    return exclusions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--aot", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--deployment-isolated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing existing output: {arguments.output}")
    campaign = read_json(arguments.core)
    aot = read_json(arguments.aot)
    deployment = read_json(arguments.deployment)
    deployment_isolated = read_json(arguments.deployment_isolated)

    cross_order = []
    for order in ORDERS:
        for dtype in ("float32", "float64"):
            for dimensions, n in ((1, 8192), (3, 64)):
                cpu = worker(campaign, "scalar", order, dtype, dimensions, n, "cpu")
                cuda = worker(campaign, "scalar", order, dtype, dimensions, n, "cuda")
                cross_order.append(
                    {
                        "order": order,
                        "dtype": dtype,
                        "dimensions": dimensions,
                        "n": n,
                        "cells": n**dimensions,
                        "cpu": CPU_summary(cpu),
                        "cuda": CUDA_summary(cuda),
                    }
                )

    scale = []
    for order in (5, 15):
        for n in (128, 512, 2048, 8192, 32768):
            scale.append(scale_row(campaign, order, 1, n))
        for n in (16, 32, 64, 96):
            scale.append(scale_row(campaign, order, 3, n))

    characteristic = []
    for order in (5, 11, 15):
        for dtype in ("float32", "float64"):
            cpu = worker(campaign, "characteristic", order, dtype, 3, 32, "cpu")
            cuda = worker(campaign, "characteristic", order, dtype, 3, 32, "cuda")
            characteristic.append(
                {
                    "order": order,
                    "dtype": dtype,
                    "dimensions": 3,
                    "n": 32,
                    "cpu": CPU_summary(cpu),
                    "cuda": CUDA_summary(cuda),
                }
            )
    for order in (5, 15):
        cuda = worker(campaign, "characteristic", order, "float32", 3, 64, "cuda")
        characteristic.append(
            {
                "order": order,
                "dtype": "float32",
                "dimensions": 3,
                "n": 64,
                "cpu": None,
                "cuda": CUDA_summary(cuda),
            }
        )

    aot_rows = []
    for order in (5, 11, 15):
        entry = aot["orders"][str(order)]
        build = entry["build_record"]
        qualification = entry["qualification"]
        aot_rows.append(
            {
                "order": order,
                "export_seconds": build["export_seconds"],
                "compile_package_seconds": build["compile_package_seconds"],
                "total_build_seconds": build["total_build_seconds"],
                "package_bytes": build["package_bytes"],
                "package_sha256": build["package_sha256"],
                "load_seconds": qualification["aot_load_seconds"],
                "first_call_after_load_seconds": qualification[
                    "aot_first_call_after_load_seconds"
                ],
                "correctness": qualification["correctness"],
                "resident_timing": qualification["resident_timing"],
                "transfer_inclusive_timing": qualification["transfer_inclusive_timing"],
            }
        )

    input_paths = tuple(
        path.resolve()
        for path in (
            arguments.core,
            arguments.aot,
            arguments.deployment,
            arguments.deployment_isolated,
        )
    )
    inputs = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    source_paths = (
        ROOT / "docs/ACADEMIC_A2_PROTOCOL.md",
        ROOT / "experiments/academic_a2/benchmark_worker.py",
        ROOT / "experiments/academic_a2/run_campaign.py",
        ROOT / "experiments/academic_a2/build_aot.py",
        ROOT / "experiments/academic_a2/aot_worker.py",
        ROOT / "experiments/academic_a2/run_aot.py",
        ROOT / "experiments/academic_a2/deployment_worker.py",
        ROOT / "experiments/academic_a2/run_deployment.py",
        Path(__file__).resolve(),
    )
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    document = {
        "schema": "gradflow-academic-a2-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete_inputs": {
            "core": campaign["complete"],
            "aot": aot["complete"],
            "deployment_prepared_cache": deployment["complete"],
            "deployment_isolated_cache": deployment_isolated["complete"],
        },
        "environment": campaign["environment"],
        "core_worker_counts": {
            "total": len(campaign["workers"]),
            "protocol_eligible": sum(
                wrapper.get("protocol_eligible", True)
                for wrapper in campaign["workers"].values()
            ),
            "unregistered_excluded": len(campaign["excluded_unregistered_workers"]),
        },
        "cross_order": cross_order,
        "scale": scale,
        "characteristic": characteristic,
        "correctness_exclusions": exclusion_rows(campaign),
        "aot": aot_rows,
        "deployment_prepared_cache": deployment["configurations"],
        "deployment_isolated_cache": deployment_isolated["configurations"],
        "input_sha256": inputs,
        "source_sha256": sources,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
