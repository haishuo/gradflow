#!/usr/bin/env python3
"""Export the frozen GradFlow academic dataset for downstream papers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "experiments/academic_a4/exports/academic-v0.1.0-rc1"
ORDERS = (5, 7, 9, 11, 13, 15)
RELEASE_TAG = "academic-v0.1.0-rc1"
RELEASE_COMMIT = "99a2a806fdaedb6cc32cdad2d621144d014865de"
INPUTS = {
    "a1": ROOT / "experiments/academic_a1/evidence/a1_20260830/numerical_limits.json",
    "a2": ROOT / "experiments/academic_a2/evidence/a2_20260830/analysis.json",
    "a3": ROOT / "experiments/academic_a3/evidence/a3_20260830/campaign.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def best_cpu_compiled(record: dict[str, Any]) -> float:
    admitted = [
        lane["resident"]["compiled"]["median_ms"]
        for lane in record["cpu"]["threads"].values()
        if lane["correctness"]["compiled"]["admitted"]
    ]
    assert admitted
    return min(admitted)


def build_data() -> dict[str, Any]:
    a1, a2, a3 = (load(INPUTS[key]) for key in ("a1", "a2", "a3"))

    roundoff = {(item["order"], item["dtype"]): item for item in a1["roundoff_sweeps"]}
    epsilon = {item["order"]: item for item in a1["epsilon_sweeps"]}
    numerical = []
    for coefficient in a1["coefficient_diagnostics"]:
        order = coefficient["order"]
        numerical.append(
            {
                "order": order,
                "substencil_width": coefficient["substencil_width"],
                "full_moment_condition_2": coefficient["full_moment_condition_2"],
                "optimal_weight_dynamic_range": coefficient[
                    "optimal_weight_dynamic_range"
                ],
                "maximum_numerator_bits": coefficient["maximum_numerator_bits"],
                "maximum_denominator_bits": coefficient["maximum_denominator_bits"],
                "float32_roundoff_onset_n": roundoff[(order, "float32")][
                    "first_sampled_roundoff_onset_n"
                ],
                "float64_roundoff_onset_n": roundoff[(order, "float64")][
                    "first_sampled_roundoff_onset_n"
                ],
                "epsilon_material_change_count": epsilon[order][
                    "material_change_count"
                ],
            }
        )

    performance = []
    cross_lookup = {
        (item["order"], item["dtype"], item["dimensions"], item["n"]): item
        for item in a2["cross_order"]
    }
    for dtype in ("float32", "float64"):
        for order in ORDERS:
            record = cross_lookup[(order, dtype, 3, 64)]
            cpu = best_cpu_compiled(record)
            cuda = record["cuda"]["resident"]["compiled"]["median_ms"]
            copy = record["cuda"]["transfer_inclusive"]["compiled"]["median_ms"]
            assert record["cuda"]["correctness"]["compiled"]["analysis_admitted"]
            performance.append(
                {
                    "order": order,
                    "dtype": dtype,
                    "cpu_compiled_ms": cpu,
                    "cuda_compiled_resident_ms": cuda,
                    "cuda_compiled_with_copy_ms": copy,
                    "resident_speedup": cpu / cuda,
                    "copy_inclusive_speedup": cpu / copy,
                }
            )

    crossover = [
        {
            "order": item["order"],
            "n": item["n"],
            "cells": item["cells"],
            "cuda_resident_over_cpu": item["cuda_resident_over_cpu_resident"],
            "cuda_with_copy_over_cpu": item["cuda_transfer_over_cpu_resident"],
        }
        for item in a2["scale"]
        if item["dimensions"] == 3 and item["order"] in (5, 15)
    ]

    isolated = {
        (item["order"], item["dimensions"], item["n"], item["lane"]): item
        for item in a2["deployment_isolated_cache"]
        if item["eligible"]
    }
    deployment = []
    for order in (5, 15):
        for lane in ("cpu_compiled", "cuda_compiled", "cuda_aot"):
            item = isolated[(order, 3, 64, lane)]
            deployment.append(
                {
                    "order": order,
                    "lane": lane,
                    "median_launch_to_answer_seconds": item[
                        "median_parent_start_to_finish_seconds"
                    ],
                }
            )

    gradient = [
        {
            "step": item["step"],
            "relative_error": item["relative_error"],
            "absolute_error": item["absolute_error"],
        }
        for item in a3["derivative_gate"]["records"]
    ]
    resolution = [
        {
            "n": item["n"],
            "steps": item["steps"],
            "recovered_speed": item["speed"],
            "truth_error": item["truth_error"],
            "objective": item["objective"],
        }
        for item in a3["resolution_study"]
    ]

    return {
        "schema": "gradflow-academic-paper-data-v1",
        "release_candidate": RELEASE_TAG,
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()
        },
        "numerical_limits": numerical,
        "performance_64cube": performance,
        "crossover_3d": crossover,
        "isolated_cache_deployment": deployment,
        "gradient_validation": gradient,
        "inverse_resolution": resolution,
        "fixed_summary": {
            "eligible_a2_workers": a2["core_worker_counts"]["protocol_eligible"],
            "a2_graph_breaks": 0,
            "inverse_truth": 1.1,
            "inverse_autograd": a3["inverse_gate"]["autograd"]["speed"],
            "inverse_derivative_free": a3["inverse_gate"]["golden_section"]["speed"],
        },
    }


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    data = build_data()
    dataset = EXPORT / "paper_data.json"
    dataset.write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "schema": "gradflow-academic-export-v1",
        "release_tag": RELEASE_TAG,
        "release_commit": RELEASE_COMMIT,
        "generation_date_utc": "2026-08-30",
        "generator": "experiments/academic_a4/export_paper_data.py",
        "generator_sha256": sha256(Path(__file__)),
        "input_sha256": data["input_sha256"],
        "outputs": {
            "paper_data.json": {
                "bytes": dataset.stat().st_size,
                "sha256": sha256(dataset),
            }
        },
    }
    manifest_path = EXPORT / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        "exported GradFlow academic dataset; "
        f"manifest sha256={sha256(manifest_path)}"
    )


if __name__ == "__main__":
    main()
