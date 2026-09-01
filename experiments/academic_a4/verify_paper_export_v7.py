#!/usr/bin/env python3
"""Verify the thread-explicit Moody academic export."""

from __future__ import annotations

import json

from export_paper_data import ROOT, sha256
from export_paper_data_v7 import (
    EXPORT,
    EXPORT_ID,
    INPUTS,
    SOURCE_RELEASE_TAG,
    SOURCE_REVISION,
)


def main() -> None:
    manifest_path = EXPORT / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "gradflow-academic-export-v7"
    assert manifest["export_id"] == EXPORT_ID
    assert manifest["source_revision"] == SOURCE_REVISION
    assert manifest["source_release_tag"] == SOURCE_RELEASE_TAG
    for path in INPUTS.values():
        relative = str(path.relative_to(ROOT))
        assert sha256(path) == manifest["input_sha256"][relative]

    dataset = EXPORT / "paper_data.json"
    output = manifest["outputs"]["paper_data.json"]
    assert dataset.stat().st_size == output["bytes"]
    assert sha256(dataset) == output["sha256"]
    data = json.loads(dataset.read_text())
    assert data["schema"] == "gradflow-academic-paper-data-v7"
    assert data["export_id"] == EXPORT_ID
    assert data["source_revision"] == SOURCE_REVISION

    moody = data["second_machine"]["moody"]
    assert moody["scientific_replication_complete"]
    for cell in moody["a2"]["cells"]:
        surface = cell["devices"]["cpu"]["thread_surface"]
        assert surface["tested_intraop_threads"] == [1, 6]
        assert cell["fastest_cpu_threads"] == 6
        assert all(
            selected == 6
            for values in surface["selected_threads_by_worker"].values()
            for selected in values
        )
        for lane in ("eager", "compiled"):
            one = surface["threads"]["1"]["lanes"][lane][
                "median_of_worker_medians_ms"
            ]
            six = surface["threads"]["6"]["lanes"][lane][
                "median_of_worker_medians_ms"
            ]
            assert six < one

    cross = moody["cross_machine"]
    assert cross["interpretation"] == "partial_performance_ordering_reproduction"
    assert cross["matched_cells"] == 6
    assert cross["winner_reproduced_cells"] == 5
    assert cross["winner_reversal_cells"] == 1
    assert cross["binary32_cuda_advantage_reproduced"]
    assert cross["all_cpu_contracts_matched"]
    assert cross["all_moody_selected_cuda_lanes_compiled"]
    reversal = cross["reversals"][0]
    assert (reversal["order"], reversal["dtype"]) == (15, "float64")
    assert reversal["primary"]["winner"] == "cpu"
    assert reversal["primary"]["cuda_lane"] == "eager"
    assert reversal["moody"]["winner"] == "cuda"
    assert reversal["moody"]["cuda_lane"] == "compiled"
    assert sum(
        row["primary"]["cuda_lane"] == "eager"
        for row in cross["comparisons"]
    ) == 2
    assert all(row["primary"]["cpu_threads"] == 6 for row in cross["comparisons"])
    assert all(row["moody"]["cpu_threads"] == 6 for row in cross["comparisons"])
    print(f"GradFlow academic export verified: {EXPORT_ID}")


if __name__ == "__main__":
    main()
