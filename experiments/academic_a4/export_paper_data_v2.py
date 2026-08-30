#!/usr/bin/env python3
"""Export the reporting-complete GradFlow academic dataset for the paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from export_paper_data import INPUTS, ROOT, build_data, load, sha256


EXPORT_ID = "academic-v0.1.0-rc1-paper-v2"
EXPORT = ROOT / "experiments/academic_a4/exports" / EXPORT_ID
SOURCE_RELEASE_TAG = "academic-v0.1.0-rc1"
SOURCE_RELEASE_COMMIT = "99a2a806fdaedb6cc32cdad2d621144d014865de"
SUMMARY_KEYS = (
    "count",
    "values",
    "minimum",
    "maximum",
    "mean",
    "median",
    "median_ms",
    "mean_ms",
    "median_absolute_deviation",
    "mad_ms",
    "sample_standard_deviation",
    "sample_standard_deviation_ms",
)


def timing_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Copy the retained observations and descriptive statistics if present."""
    return {key: record[key] for key in SUMMARY_KEYS if key in record}


def performance_with_dispersion(a2: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in a2["cross_order"]:
        if record["dimensions"] != 3 or record["n"] != 64:
            continue
        admitted_cpu = []
        for thread, lane in record["cpu"]["threads"].items():
            if lane["correctness"]["compiled"]["admitted"]:
                admitted_cpu.append((int(thread), lane["resident"]["compiled"]))
        cpu_threads, cpu = min(admitted_cpu, key=lambda item: item[1]["median_ms"])
        cuda = record["cuda"]["resident"]["compiled"]
        cuda_copy = record["cuda"]["transfer_inclusive"]["compiled"]
        assert record["cuda"]["correctness"]["compiled"]["analysis_admitted"]
        rows.append(
            {
                "order": record["order"],
                "dtype": record["dtype"],
                "cpu_threads": cpu_threads,
                "cpu_compiled": timing_summary(cpu),
                "cuda_compiled_resident": timing_summary(cuda),
                "cuda_compiled_with_copy": timing_summary(cuda_copy),
                "resident_speedup": cpu["median_ms"] / cuda["median_ms"],
                "copy_inclusive_speedup": (
                    cpu["median_ms"] / cuda_copy["median_ms"]
                ),
            }
        )
    return rows


def deployment_with_dispersion(a2: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in a2["deployment_isolated_cache"]:
        if not item["eligible"] or item["dimensions"] != 3 or item["n"] != 64:
            continue
        if item["order"] not in (5, 15):
            continue
        values = [record["parent_start_to_finish_seconds"] for record in item["records"]]
        rows.append(
            {
                "order": item["order"],
                "lane": item["lane"],
                "count": len(values),
                "values_seconds": values,
                "minimum_seconds": min(values),
                "median_seconds": item["median_parent_start_to_finish_seconds"],
                "maximum_seconds": max(values),
            }
        )
    return rows


def aot_summary(a2: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in a2["aot"]:
        rows.append(
            {
                "order": item["order"],
                "export_seconds": item["export_seconds"],
                "compile_package_seconds": item["compile_package_seconds"],
                "total_build_seconds": item["total_build_seconds"],
                "package_bytes": item["package_bytes"],
                "package_sha256": item["package_sha256"],
                "load_seconds": item["load_seconds"],
                "first_call_after_load_seconds": item[
                    "first_call_after_load_seconds"
                ],
                "resident_jit": timing_summary(
                    item["resident_timing"]["lanes"]["jit"]
                ),
                "resident_aot": timing_summary(
                    item["resident_timing"]["lanes"]["aot"]
                ),
                "resident_aot_over_jit": item["resident_timing"][
                    "paired_analysis"
                ]["aot_over_jit"],
                "resident_decision": item["resident_timing"]["paired_analysis"][
                    "decision"
                ],
                "with_copy_jit": timing_summary(
                    item["transfer_inclusive_timing"]["lanes"]["jit"]
                ),
                "with_copy_aot": timing_summary(
                    item["transfer_inclusive_timing"]["lanes"]["aot"]
                ),
                "with_copy_aot_over_jit": item["transfer_inclusive_timing"][
                    "paired_analysis"
                ]["aot_over_jit"],
                "with_copy_decision": item["transfer_inclusive_timing"][
                    "paired_analysis"
                ]["decision"],
            }
        )
    return rows


def inverse_summary(a3: dict[str, Any]) -> dict[str, Any]:
    inverse = a3["inverse_gate"]
    autograd = inverse["autograd"]
    golden = inverse["golden_section"]
    truth = 1.1
    return {
        "n": inverse["n"],
        "order": inverse["order"],
        "initial_speed": inverse["initial_speed"],
        "initial_objective": inverse["initial_objective"],
        "truth": truth,
        "autograd": {
            "speed": autograd["speed"],
            "truth_error": abs(autograd["speed"] - truth),
            "objective": autograd["objective"],
            "speed_gradient": autograd["speed_gradient"],
            "closure_evaluations": autograd["closure_evaluations"],
        },
        "golden_section": {
            "speed": golden["speed"],
            "truth_error": abs(golden["speed"] - truth),
            "objective": golden["objective"],
            "iterations": golden["iterations"],
            "objective_evaluations": golden["objective_evaluations"],
            "final_interval": golden["final_interval"],
        },
    }


def differentiation_benchmarks(a3: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for device, wrapper in a3["benchmarks"].items():
        record = wrapper["record"]
        lane_rows = {}
        for lane in ("eager", "compiled"):
            lane_rows[lane] = {
                "forward_ms": timing_summary(record["timings"][lane]["forward_ms"]),
                "objective_and_gradient_ms": timing_summary(
                    record["timings"][lane]["objective_and_gradient_ms"]
                ),
                "reverse_mode_over_forward_median": record["timings"][lane][
                    "reverse_mode_over_forward_median"
                ],
                "first_objective_and_gradient_seconds": record[lane][
                    "first_objective_and_gradient_seconds"
                ],
            }
        eager = lane_rows["eager"]["objective_and_gradient_ms"]["median"]
        compiled = lane_rows["compiled"]["objective_and_gradient_ms"]["median"]
        rows.append(
            {
                "device": device,
                "lanes": lane_rows,
                "warm_objective_and_gradient_speedup": eager / compiled,
            }
        )
    return rows


def build_data_v2() -> dict[str, Any]:
    data = build_data()
    a2 = load(INPUTS["a2"])
    a3 = load(INPUTS["a3"])
    derivative = a3["derivative_gate"]
    data.update(
        {
            "schema": "gradflow-academic-paper-data-v2",
            "export_id": EXPORT_ID,
            "source_release_candidate": SOURCE_RELEASE_TAG,
            "environment": {"a2": a2["environment"], "a3": a3["environment"]},
            "performance_64cube": performance_with_dispersion(a2),
            "isolated_cache_deployment": deployment_with_dispersion(a2),
            "aot_packages": aot_summary(a2),
            "gradient_validation": {
                "evaluation_speed": derivative["evaluation_speed"],
                "objective": derivative["objective"],
                "autograd_derivative": derivative["autograd_derivative"],
                "records": derivative["records"],
            },
            "inverse_problem": inverse_summary(a3),
            "differentiation_benchmarks": differentiation_benchmarks(a3),
        }
    )
    return data


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    data = build_data_v2()
    dataset = EXPORT / "paper_data.json"
    dataset.write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "schema": "gradflow-academic-export-v2",
        "export_id": EXPORT_ID,
        "source_release_tag": SOURCE_RELEASE_TAG,
        "source_release_commit": SOURCE_RELEASE_COMMIT,
        "generation_date_utc": "2026-08-30",
        "generator": "experiments/academic_a4/export_paper_data_v2.py",
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
    print(f"exported {EXPORT_ID}; manifest sha256={sha256(manifest_path)}")
if __name__ == "__main__":
    main()
