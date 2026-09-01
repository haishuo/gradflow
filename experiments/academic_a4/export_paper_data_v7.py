#!/usr/bin/env python3
"""Extend paper export v6 with CPU-thread and cross-machine dispositions."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

from export_paper_data import ROOT, load, sha256


EXPORT_ID = "academic-f65803e-moody-paper-v7"
EXPORT = ROOT / "experiments/academic_a4/exports" / EXPORT_ID
SOURCE_RELEASE_TAG = "academic-v0.1.0-rc2"
SOURCE_REVISION = "f65803e6f0dc12b773b7a762c7625cf690bc65d4"
INPUTS = {
    "paper_v6": ROOT
    / "experiments/academic_a4/exports/academic-3992e06-moody-paper-v6/paper_data.json",
    "moody_record": ROOT
    / "experiments/academic_a4/evidence/moody_20260831/second_machine.json",
    "moody_results": ROOT / "docs/ACADEMIC_A4_MOODY_RESULTS.md",
}


def cpu_thread_surface(record: dict, order: int, dtype: str) -> dict:
    records = [
        worker["record"]
        for worker in record["a2_workers"]
        if worker["device"] == "cpu"
        and worker["order"] == order
        and worker["dtype"] == dtype
    ]
    assert len(records) == 3
    threads = {}
    for thread_count in ("1", "6"):
        lanes = {}
        for lane in ("eager", "compiled"):
            values = [
                worker["cpu"][thread_count]["resident_timing"]["lanes"][lane][
                    "median"
                ]
                for worker in records
            ]
            lanes[lane] = {
                "worker_medians_ms": values,
                "median_of_worker_medians_ms": statistics.median(values),
            }
        threads[thread_count] = {"lanes": lanes}
    selected_by_worker = {}
    for lane in ("eager", "compiled"):
        selected_by_worker[lane] = [
            int(
                min(
                    (worker["cpu"][thread]["resident_timing"]["lanes"][lane]["median"], thread)
                    for thread in ("1", "6")
                )[1]
            )
            for worker in records
        ]
    return {
        "tested_intraop_threads": [1, 6],
        "threads": threads,
        "selected_threads_by_worker": selected_by_worker,
    }


def winner(cpu_over_cuda: float) -> str:
    if cpu_over_cuda > 1.0:
        return "cuda"
    if cpu_over_cuda < 1.0:
        return "cpu"
    return "tie"


def enrich(data: dict, record: dict) -> None:
    moody = data["second_machine"]["moody"]
    for cell in moody["a2"]["cells"]:
        surface = cpu_thread_surface(record, cell["order"], cell["dtype"])
        cell["devices"]["cpu"]["thread_surface"] = surface
        selected = surface["selected_threads_by_worker"][cell["fastest_cpu_lane"]]
        assert len(set(selected)) == 1
        cell["fastest_cpu_threads"] = selected[0]
        selected_lane = surface["threads"][str(cell["fastest_cpu_threads"])][
            "lanes"
        ][cell["fastest_cpu_lane"]]
        assert selected_lane["worker_medians_ms"] == cell["devices"]["cpu"][
            cell["fastest_cpu_lane"]
        ]["worker_medians_ms"]

    primary = {
        (row["order"], row["dtype"]): row["stable"]
        for row in data["stable_toolchain"]["performance_64cube"]
    }
    comparisons = []
    for moody_cell in moody["a2"]["cells"]:
        key = (moody_cell["order"], moody_cell["dtype"])
        forge = primary[key]
        forge_ratio = forge["cpu_over_cuda_resident"]
        moody_ratio = moody_cell["fastest_cpu_over_fastest_cuda"]
        forge_winner = winner(forge_ratio)
        moody_winner = winner(moody_ratio)
        comparisons.append(
            {
                "order": key[0],
                "dtype": key[1],
                "primary": {
                    "cpu_median_ms": forge["cpu_ms"],
                    "cpu_lane": forge["cpu_lane"],
                    "cpu_threads": forge["cpu_threads"],
                    "cuda_median_ms": forge["cuda_resident_ms"],
                    "cuda_lane": forge["cuda_resident_lane"],
                    "cpu_over_cuda": forge_ratio,
                    "winner": forge_winner,
                },
                "moody": {
                    "cpu_median_ms": moody_cell["fastest_cpu_median_ms"],
                    "cpu_lane": moody_cell["fastest_cpu_lane"],
                    "cpu_threads": moody_cell["fastest_cpu_threads"],
                    "cuda_median_ms": moody_cell["fastest_cuda_median_ms"],
                    "cuda_lane": moody_cell["fastest_cuda_lane"],
                    "cpu_over_cuda": moody_ratio,
                    "winner": moody_winner,
                },
                "winner_reproduced": forge_winner == moody_winner,
                "cpu_contract_matched": (
                    forge["cpu_threads"] == moody_cell["fastest_cpu_threads"]
                    and forge["cpu_lane"] == moody_cell["fastest_cpu_lane"]
                ),
                "cuda_lane_reproduced": (
                    forge["cuda_resident_lane"] == moody_cell["fastest_cuda_lane"]
                ),
            }
        )
    reversals = [row for row in comparisons if not row["winner_reproduced"]]
    moody["cross_machine"] = {
        "schema": "gradflow-academic-moody-cross-machine-paper-v1",
        "interpretation": "partial_performance_ordering_reproduction",
        "matched_cells": len(comparisons),
        "winner_reproduced_cells": len(comparisons) - len(reversals),
        "winner_reversal_cells": len(reversals),
        "binary32_cuda_advantage_reproduced": all(
            row["primary"]["winner"] == row["moody"]["winner"] == "cuda"
            for row in comparisons
            if row["dtype"] == "float32"
        ),
        "all_cpu_contracts_matched": all(
            row["cpu_contract_matched"] for row in comparisons
        ),
        "all_moody_selected_cuda_lanes_compiled": all(
            row["moody"]["cuda_lane"] == "compiled" for row in comparisons
        ),
        "comparisons": comparisons,
        "reversals": reversals,
    }


def main() -> None:
    EXPORT.mkdir(parents=True, exist_ok=True)
    data = load(INPUTS["paper_v6"])
    record = load(INPUTS["moody_record"])
    enrich(data, record)
    data.update(
        {
            "schema": "gradflow-academic-paper-data-v7",
            "export_id": EXPORT_ID,
            "source_revision": SOURCE_REVISION,
        }
    )
    data["input_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in INPUTS.values()
    }

    dataset = EXPORT / "paper_data.json"
    dataset.write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        "schema": "gradflow-academic-export-v7",
        "export_id": EXPORT_ID,
        "source_revision": SOURCE_REVISION,
        "source_release_tag": SOURCE_RELEASE_TAG,
        "generation_date_utc": "2026-09-01",
        "generator": "experiments/academic_a4/export_paper_data_v7.py",
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
        f"Evidence interpretation commit: `{SOURCE_REVISION}`  \n"
        f"Scientific source tag: `{SOURCE_RELEASE_TAG}`\n\n"
        "This immutable export supersedes v6 for paper rendering. It preserves "
        "the same Moody observations while adding the explicit one-/six-thread "
        "CPU surface, the selected six-thread identity, matched primary/Moody "
        "lane identities, and the binary64 WENO-15 CPU/CUDA winner reversal. "
        "The source measurements are unchanged.\n"
    )
    print(f"exported {EXPORT_ID}; manifest sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
