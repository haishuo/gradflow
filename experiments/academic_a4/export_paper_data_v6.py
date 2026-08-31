#!/usr/bin/env python3
"""Extend paper export v5 with the frozen Moody modern-GPU replication."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

from export_paper_data import ROOT, load, sha256


EXPORT_ID = "academic-3992e06-moody-paper-v6"
EXPORT = ROOT / "experiments/academic_a4/exports" / EXPORT_ID
SOURCE_RELEASE_TAG = "academic-v0.1.0-rc2"
SOURCE_REVISION = "3992e06939005d27b2d99017992e68d383b5034f"
INPUTS = {
    "paper_v5": ROOT
    / "experiments/academic_a4/exports/academic-33c469b-unity-paper-v5/paper_data.json",
    "moody_record": ROOT
    / "experiments/academic_a4/evidence/moody_20260831/second_machine.json",
    "moody_pytest": ROOT
    / "experiments/academic_a4/evidence/moody_20260831/raw/pytest.stdout",
    "moody_checksums": ROOT
    / "experiments/academic_a4/evidence/moody_20260831/SHA256SUMS",
    "moody_results": ROOT / "docs/ACADEMIC_A4_MOODY_RESULTS.md",
}


def cpu_model(record: dict) -> str:
    text = record["environment"]["commands"]["lscpu"]["stdout"]
    for line in text.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("Moody lscpu record has no model name")


def pytest_summary(text: str) -> dict[str, int]:
    match = re.search(r"(\d+) failed, (\d+) passed, (\d+) skipped, .* in ", text)
    if match is None:
        raise AssertionError("Moody pytest summary is missing")
    failed, passed, skipped = (int(value) for value in match.groups())
    return {"passed": passed, "skipped": skipped, "failed": failed}


def lane_summary(lanes: dict) -> dict:
    return {
        name: {
            "worker_medians_ms": lane["worker_medians_ms"],
            "median_of_worker_medians_ms": lane["median_of_worker_medians_ms"],
        }
        for name, lane in lanes.items()
    }


def summarize_a2(record: dict) -> dict:
    cells = []
    for cell in record["a2_analysis"]["cells"]:
        devices = {
            name: lane_summary(device["lanes"])
            for name, device in cell["devices"].items()
        }
        cpu_lane, cpu = min(
            devices["cpu"].items(),
            key=lambda item: item[1]["median_of_worker_medians_ms"],
        )
        cuda_lane, cuda = min(
            devices["cuda"].items(),
            key=lambda item: item[1]["median_of_worker_medians_ms"],
        )
        cpu_ms = cpu["median_of_worker_medians_ms"]
        cuda_ms = cuda["median_of_worker_medians_ms"]
        cells.append(
            {
                "order": cell["order"],
                "dtype": cell["dtype"],
                "devices": devices,
                "fastest_cpu_lane": cpu_lane,
                "fastest_cpu_median_ms": cpu_ms,
                "fastest_cuda_lane": cuda_lane,
                "fastest_cuda_median_ms": cuda_ms,
                "fastest_cuda_over_fastest_cpu": cuda_ms / cpu_ms,
                "fastest_cpu_over_fastest_cuda": cpu_ms / cuda_ms,
            }
        )
    return {
        "workers": len(record["a2_workers"]),
        "all_expected_workers_parsed": record["a2_analysis"][
            "all_expected_workers_parsed"
        ],
        "admission_failures": record["a2_analysis"]["admission_failures"],
        "graph_failures": record["a2_analysis"]["graph_failures"],
        "graph_contract_passed": record["qualification"][
            "a2_graph_contract_passed"
        ],
        "binary32_cuda_materially_useful": record["qualification"][
            "binary32_cuda_materially_useful"
        ],
        "cells": cells,
    }


def summarize_a3(record: dict) -> dict:
    campaign = record["a3"]["record"]
    devices = {}
    for device, wrapper in campaign["benchmarks"].items():
        benchmark = wrapper["record"]
        compiled = benchmark["compiled"]
        timing = benchmark["timings"]["compiled"]
        devices[device] = {
            "compiled_admitted": compiled["admitted"],
            "compiled_first_objective_and_gradient_seconds": compiled[
                "first_objective_and_gradient_seconds"
            ],
            "compiled_objective_absolute_error": compiled["correctness"][
                "objective"
            ]["absolute"],
            "compiled_gradient_absolute_error": compiled["correctness"][
                "gradient"
            ]["absolute"],
            "compiled_unique_graphs": compiled["graph"]["unique_graphs"],
            "compiled_graph_break_count": compiled["graph"]["graph_break_count"],
            "compiled_forward_median_ms": timing["forward_ms"]["median"],
            "compiled_objective_and_gradient_median_ms": timing[
                "objective_and_gradient_ms"
            ]["median"],
        }
    return {
        "campaign_returncode": record["a3"]["returncode"],
        "derivative_gate_passed": campaign["derivative_gate"][
            "registered_window_passed"
        ],
        "inverse_gate_passed": campaign["inverse_gate"]["passed"],
        "devices": devices,
    }


def summarize_moody(record: dict, pytest_text: str) -> dict:
    created = datetime.fromisoformat(record["created_utc"])
    completed = datetime.fromisoformat(record["completed_utc"])
    qualification = record["qualification"]
    scientific_keys = (
        "a1_completed",
        "a3_agreement_passed",
        "a2_worker_surface_complete",
        "a2_graph_contract_passed",
        "binary32_cuda_materially_useful",
    )
    scientific_complete = all(qualification[key] for key in scientific_keys) and not (
        qualification["admission_failures"]
    )
    return {
        "schema": "gradflow-academic-moody-replication-paper-v1",
        "status": record["status"],
        "formal_all_sentinels_gate_passed": qualification["sentinels_passed"],
        "scientific_replication_complete": scientific_complete,
        "interpretation": "modern_gpu_scientific_replication_with_packet_limits",
        "elapsed_seconds": (completed - created).total_seconds(),
        "source_tag": record["source_tag"],
        "source_commit": record["source_commit"],
        "machine": {
            "hostname": record["environment"]["hostname"],
            "cpu": cpu_model(record),
            "gpu": record["environment"]["gpu"],
            "gpu_capability": record["environment"]["gpu_capability"],
            "gpu_memory_bytes": record["environment"]["gpu_memory_bytes"],
            "python": record["environment"]["python"],
            "torch": record["environment"]["torch"],
            "torch_commit": record["environment"]["torch_commit"],
            "cuda_runtime": record["environment"]["cuda_runtime"],
            "execution_context": record["environment"]["execution_context"],
        },
        "pytest": pytest_summary(pytest_text),
        "sentinels": [
            {
                "name": sentinel["name"],
                "returncode": sentinel["returncode"],
                "duration_seconds": sentinel["duration_seconds"],
            }
            for sentinel in record["sentinels"]
        ],
        "a1_completed": qualification["a1_completed"],
        "a2": summarize_a2(record),
        "a3": summarize_a3(record),
        "packet_limits": {
            "missing_rc1_tag_failures": 2,
            "untransported_machine_specific_aot_failures": 2,
            "scientific_worker_failures": 0,
        },
    }


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    data = load(INPUTS["paper_v5"])
    moody = load(INPUTS["moody_record"])
    pytest_text = INPUTS["moody_pytest"].read_text()
    second_machine = dict(data["second_machine"])
    second_machine["moody"] = summarize_moody(moody, pytest_text)
    data.update(
        {
            "schema": "gradflow-academic-paper-data-v6",
            "export_id": EXPORT_ID,
            "source_revision": SOURCE_REVISION,
            "second_machine": second_machine,
        }
    )
    data["input_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()
    }

    dataset = EXPORT / "paper_data.json"
    dataset.write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "schema": "gradflow-academic-export-v6",
        "export_id": EXPORT_ID,
        "source_revision": SOURCE_REVISION,
        "source_release_tag": SOURCE_RELEASE_TAG,
        "generation_date_utc": "2026-08-31",
        "generator": "experiments/academic_a4/export_paper_data_v6.py",
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
    (EXPORT / "README.md").write_text(
        f"# GradFlow paper export `{EXPORT_ID}`\n\n"
        f"Evidence commit: `{SOURCE_REVISION}`  \n"
        f"Scientific source tag: `{SOURCE_RELEASE_TAG}`\n\n"
        "This immutable export extends paper export v5 with the prospectively "
        "frozen Moody modern-GPU replication. The complete A1/A2/A3 scientific "
        "surface and its timings are exported alongside the unchanged aggregate "
        "controller status and four documented transport-packet limitations. "
        "External review, rights, and licensing results are not included.\n"
    )
    print(f"exported {EXPORT_ID}; manifest sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
