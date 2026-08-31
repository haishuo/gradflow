#!/usr/bin/env python3
"""Build the deterministic stable/development PyTorch U5 comparison."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def best_times(row: dict[str, Any]) -> dict[str, Any]:
    cpu = []
    for threads, record in row["cpu"]["threads"].items():
        for lane, timing in record["resident"].items():
            if timing is not None and record["correctness"][lane]["admitted"]:
                cpu.append((timing["median_ms"], lane, int(threads)))
    cuda = []
    transfer = []
    for lane, timing in row["cuda"]["resident"].items():
        if timing is not None and row["cuda"]["correctness"][lane]["analysis_admitted"]:
            cuda.append((timing["median_ms"], lane))
    for lane, timing in row["cuda"]["transfer_inclusive"].items():
        if timing is not None and row["cuda"]["correctness"][lane]["analysis_admitted"]:
            transfer.append((timing["median_ms"], lane))
    cpu_best = min(cpu)
    cuda_best = min(cuda)
    transfer_best = min(transfer)
    return {
        "cpu_ms": cpu_best[0],
        "cpu_lane": cpu_best[1],
        "cpu_threads": cpu_best[2],
        "cuda_resident_ms": cuda_best[0],
        "cuda_resident_lane": cuda_best[1],
        "cuda_transfer_ms": transfer_best[0],
        "cuda_transfer_lane": transfer_best[1],
        "cpu_over_cuda_resident": cpu_best[0] / cuda_best[0],
        "cpu_over_cuda_transfer": cpu_best[0] / transfer_best[0],
    }


def exclusion_ids(document: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (record["worker"], record["endpoint"])
        for record in document["correctness_exclusions"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()

    stable_a1 = load(evidence / "a1/numerical_limits.json")
    old_a1 = load(ROOT / "experiments/academic_a1/evidence/a1_20260830/numerical_limits.json")
    stable_a2 = load(evidence / "a2/analysis.json")
    old_a2 = load(ROOT / "experiments/academic_a2/evidence/a2_20260830/analysis.json")
    stable_a3 = load(evidence / "a3/campaign.json")
    old_a3 = load(ROOT / "experiments/academic_a3/evidence/a3_20260830/campaign.json")
    stable_u4f = load(evidence / "u4f/campaign.json")
    old_u4f = load(ROOT / "experiments/academic_u4f/evidence/u4f_20260831/campaign.json")

    coefficient_relative_differences = []
    fields = (
        "maximum_candidate_moment_condition_2",
        "full_moment_condition_2",
        "maximum_smoothness_restricted_condition_2",
    )
    for stable, old in zip(
        stable_a1["coefficient_diagnostics"], old_a1["coefficient_diagnostics"]
    ):
        for field in fields:
            coefficient_relative_differences.append(
                abs(stable[field] - old[field]) / max(abs(old[field]), 1.0e-300)
            )

    cross_order = []
    stable_speedups: dict[str, list[float]] = {"float32": [], "float64": []}
    stable_transfer_speedups: dict[str, list[float]] = {
        "float32": [], "float64": []
    }
    old_rows = {
        (row["order"], row["dtype"], row["dimensions"], row["n"]): row
        for row in old_a2["cross_order"]
    }
    for row in stable_a2["cross_order"]:
        if row["dimensions"] != 3:
            continue
        key = (row["order"], row["dtype"], row["dimensions"], row["n"])
        stable = best_times(row)
        old = best_times(old_rows[key])
        stable_speedups[row["dtype"]].append(stable["cpu_over_cuda_resident"])
        stable_transfer_speedups[row["dtype"]].append(
            stable["cpu_over_cuda_transfer"]
        )
        cross_order.append(
            {
                "order": row["order"],
                "dtype": row["dtype"],
                "stable": stable,
                "development": old,
                "stable_over_development_cpu": stable["cpu_ms"] / old["cpu_ms"],
                "stable_over_development_cuda": (
                    stable["cuda_resident_ms"] / old["cuda_resident_ms"]
                ),
            }
        )

    old_aot = {record["order"]: record for record in old_a2["aot"]}
    aot = []
    for stable in stable_a2["aot"]:
        old = old_aot[stable["order"]]
        aot.append(
            {
                "order": stable["order"],
                "stable_build_seconds": stable["total_build_seconds"],
                "development_build_seconds": old["total_build_seconds"],
                "stable_jit_ms": stable["resident_timing"]["lanes"]["jit"]["median"],
                "stable_aot_ms": stable["resident_timing"]["lanes"]["aot"]["median"],
                "stable_decision": stable["resident_timing"]["paired_analysis"]["decision"],
                "development_jit_ms": old["resident_timing"]["lanes"]["jit"]["median"],
                "development_aot_ms": old["resident_timing"]["lanes"]["aot"]["median"],
                "development_decision": old["resident_timing"]["paired_analysis"]["decision"],
            }
        )

    deployment = {}
    for name in ("deployment_prepared_cache", "deployment_isolated_cache"):
        deployment[name] = [
            {
                "order": row["order"],
                "dimensions": row["dimensions"],
                "lane": row["lane"],
                "median_seconds": row["median_parent_start_to_finish_seconds"],
            }
            for row in stable_a2[name]
            if row["eligible"]
        ]

    a3 = {
        "stable_inverse_speed": stable_a3["inverse_gate"]["autograd"]["speed"],
        "development_inverse_speed": old_a3["inverse_gate"]["autograd"]["speed"],
        "stable_golden_speed": stable_a3["inverse_gate"]["golden_section"]["speed"],
        "stable_best_gradient_relative_error": min(
            record["relative_error"]
            for record in stable_a3["derivative_gate"]["records"]
        ),
        "devices": {},
    }
    for device in ("cpu", "cuda"):
        stable = stable_a3["benchmarks"][device]["record"]
        old = old_a3["benchmarks"][device]["record"]
        a3["devices"][device] = {
            "stable_compile_seconds": stable["compiled"]["first_objective_and_gradient_seconds"],
            "development_compile_seconds": old["compiled"]["first_objective_and_gradient_seconds"],
            "stable_compiled_forward_ms": stable["timings"]["compiled"]["forward_ms"]["median"],
            "stable_compiled_gradient_ms": stable["timings"]["compiled"]["objective_and_gradient_ms"]["median"],
            "development_compiled_forward_ms": old["timings"]["compiled"]["forward_ms"]["median"],
            "development_compiled_gradient_ms": old["timings"]["compiled"]["objective_and_gradient_ms"]["median"],
        }

    u4f = []
    for batch in stable_u4f["batches"]:
        stable_cell = stable_u4f["cells"][str(batch)]
        old_cell = old_u4f["cells"][str(batch)]
        record: dict[str, Any] = {"batch": batch, "devices": {}}
        for device in ("cpu", "cuda"):
            stable_resident = stable_cell["resident"][device]
            stable_analysis = stable_resident["analysis"]
            old_resident = old_cell.get("resident", {}).get(device)
            record["devices"][device] = {
                "stable_admitted": stable_cell["admitted"][device],
                "development_admitted": old_cell["admitted"][device],
                "stable_dveb_ms": stable_analysis["lanes"]["dveb_native"]["worker_medians"]["median"],
                "stable_pytorch_ms": stable_analysis["lanes"]["pytorch_inductor"]["worker_medians"]["median"],
                "stable_ratio": stable_analysis["paired_worker_median_ratio_pytorch_over_dveb"]["median"],
                "stable_interval": stable_analysis["paired_worker_median_ratio_pytorch_over_dveb"]["bootstrap_median_95_ci"],
                "stable_decision": stable_analysis["decision"],
                "development_decision": (
                    None if old_resident is None else old_resident["analysis"]["decision"]
                ),
            }
        u4f.append(record)

    stable_exclusions = exclusion_ids(stable_a2)
    old_exclusions = exclusion_ids(old_a2)
    document = {
        "schema": "gradflow.academic_u5.comparison.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "a1": {
            "qualified_orders_identical": stable_a1["qualified_orders"] == old_a1["qualified_orders"],
            "roundoff_sweeps_identical": stable_a1["roundoff_sweeps"] == old_a1["roundoff_sweeps"],
            "epsilon_sweeps_identical": stable_a1["epsilon_sweeps"] == old_a1["epsilon_sweeps"],
            "maximum_selected_condition_diagnostic_relative_difference": max(
                coefficient_relative_differences
            ),
        },
        "a2": {
            "stable_environment": stable_a2["environment"],
            "development_environment": old_a2["environment"],
            "stable_worker_counts": stable_a2["core_worker_counts"],
            "cross_order_64cube": cross_order,
            "stable_speedup_ranges": {
                dtype: {
                    "resident": [min(values), max(values)],
                    "transfer_inclusive": [
                        min(stable_transfer_speedups[dtype]),
                        max(stable_transfer_speedups[dtype]),
                    ],
                }
                for dtype, values in stable_speedups.items()
            },
            "stable_exclusion_count": len(stable_exclusions),
            "development_exclusion_count": len(old_exclusions),
            "exclusions_removed": sorted(old_exclusions - stable_exclusions),
            "exclusions_added": sorted(stable_exclusions - old_exclusions),
            "aot": aot,
            "deployment": deployment,
        },
        "a3": a3,
        "u4f": {
            "batched_cpu_compiler_failure_fixed": all(
                stable_u4f["cells"][str(batch)]["admitted"]["cpu"]
                for batch in stable_u4f["batches"]
            ),
            "cells": u4f,
        },
        "claim_boundary": {
            "one_machine": True,
            "cross_version_timings_are_paired": False,
            "compiler_change_causally_isolated": False,
            "universal_backend_winner_claimed": False,
            "canonical_math_changed": False,
        },
    }
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()

