#!/usr/bin/env python3
"""Extend paper export v4 with the frozen Unity portability observation."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re

from export_paper_data import ROOT, load, sha256


EXPORT_ID = "academic-33c469b-unity-paper-v5"
EXPORT = ROOT / "experiments/academic_a4/exports" / EXPORT_ID
SOURCE_RELEASE_TAG = "academic-v0.1.0-rc2"
SOURCE_REVISION = "33c469b98e6b1dab8aa93026580e56c5fe33af6c"
INPUTS = {
    "paper_v4": ROOT / "experiments/academic_a4/exports/academic-v0.1.0-rc2-paper-v4/paper_data.json",
    "unity_record": ROOT / "experiments/academic_a4/evidence/unity_20260831/second_machine.json",
    "unity_pytest": ROOT / "experiments/academic_a4/evidence/unity_20260831/raw/pytest.stdout",
    "unity_import_manifest": ROOT / "experiments/academic_a4/evidence/unity_20260831/IMPORT_SHA256SUMS",
    "unity_results": ROOT / "docs/ACADEMIC_A4_UNITY_RESULTS.md",
}


def cpu_model(record: dict) -> str:
    text = record["environment"]["commands"]["lscpu"]["stdout"]
    for line in text.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    raise AssertionError("Unity lscpu record has no model name")


def pytest_summary(text: str) -> dict[str, int]:
    match = re.search(
        r"(\d+) failed, (\d+) passed, (\d+) skipped, .* in ",
        text,
    )
    if match is None:
        raise AssertionError("Unity pytest summary is missing")
    failed, passed, skipped = (int(value) for value in match.groups())
    return {"passed": passed, "skipped": skipped, "failed": failed}


def summarize_unity(record: dict, pytest_text: str) -> dict:
    created = datetime.fromisoformat(record["created_utc"])
    completed = datetime.fromisoformat(record["completed_utc"])
    a3 = record["a3"]["record"]
    a2_cells = []
    for cell in record["a2_analysis"]["cells"]:
        cpu = cell["devices"]["cpu"]["lanes"]["compiled"]
        cuda = cell["devices"]["cuda"]["lanes"]["eager"]
        a2_cells.append(
            {
                "order": cell["order"],
                "dtype": cell["dtype"],
                "compiled_cpu_worker_medians_ms": cpu["worker_medians_ms"],
                "compiled_cpu_median_ms": cpu["median_of_worker_medians_ms"],
                "eager_cuda_worker_medians_ms": cuda["worker_medians_ms"],
                "eager_cuda_median_ms": cuda["median_of_worker_medians_ms"],
                "fastest_cuda_over_fastest_cpu": cell[
                    "fastest_cuda_over_fastest_cpu"
                ],
            }
        )
    cuda_a3 = a3["benchmarks"]["cuda"]["record"]
    first_cuda_worker = next(
        worker for worker in record["a2_workers"] if worker["device"] == "cuda"
    )
    return {
        "schema": "gradflow-academic-unity-portability-paper-v1",
        "status": record["status"],
        "closes_second_machine_gate": False,
        "interpretation": "negative_legacy_gpu_compiler_portability_boundary",
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
            "slurm_job_id": record["environment"]["slurm"]["SLURM_JOB_ID"],
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
        "a1_completed": record["qualification"]["a1_completed"],
        "a3": {
            "campaign_returncode": record["a3"]["returncode"],
            "derivative_gate_passed": a3["derivative_gate"][
                "registered_window_passed"
            ],
            "inverse_gate_passed": a3["inverse_gate"]["passed"],
            "cuda_eager_admitted": cuda_a3["eager"]["admitted"],
            "cuda_eager_objective_absolute_error": cuda_a3["eager"][
                "correctness"
            ]["objective"]["absolute"],
            "cuda_eager_gradient_absolute_error": cuda_a3["eager"][
                "correctness"
            ]["gradient"]["absolute"],
            "cuda_compiled_admitted": cuda_a3["compiled"]["admitted"],
            "cuda_compiled_error": cuda_a3["compiled"]["error"],
        },
        "a2": {
            "workers": len(record["a2_workers"]),
            "compiled_cuda_admission_failures": len(
                record["qualification"]["admission_failures"]
            ),
            "graph_contract_passed": record["qualification"][
                "a2_graph_contract_passed"
            ],
            "binary32_cuda_materially_useful": record["qualification"][
                "binary32_cuda_materially_useful"
            ],
            "compiler_error": first_cuda_worker["record"]["correctness"][
                "compiled"
            ]["error"],
            "cells": a2_cells,
        },
        "packet_limits": {
            "missing_rc1_tag_failures": 2,
            "untransported_machine_specific_aot_failures": 2,
            "direct_compiled_cuda_test_failures": 7,
        },
    }


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    data = load(INPUTS["paper_v4"])
    unity = load(INPUTS["unity_record"])
    pytest_text = INPUTS["unity_pytest"].read_text()
    data.update(
        {
            "schema": "gradflow-academic-paper-data-v5",
            "export_id": EXPORT_ID,
            "source_revision": SOURCE_REVISION,
            "second_machine": {"unity": summarize_unity(unity, pytest_text)},
        }
    )
    data["input_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()
    }

    dataset = EXPORT / "paper_data.json"
    dataset.write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "schema": "gradflow-academic-export-v5",
        "export_id": EXPORT_ID,
        "source_revision": SOURCE_REVISION,
        "source_release_tag": SOURCE_RELEASE_TAG,
        "generation_date_utc": "2026-08-31",
        "generator": "experiments/academic_a4/export_paper_data_v5.py",
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
        "This immutable export extends paper export v4 with the frozen Unity "
        "negative portability observation. It does not claim that Unity closes "
        "the suitable modern-GPU second-machine gate. Moody, external-review, "
        "rights, and licensing results are not included.\n"
    )
    print(f"exported {EXPORT_ID}; manifest sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
