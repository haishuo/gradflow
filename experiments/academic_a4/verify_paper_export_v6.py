#!/usr/bin/env python3
"""Verify the Moody-extended downstream academic export."""

from __future__ import annotations

import json

from export_paper_data import ROOT, sha256
from export_paper_data_v6 import (
    EXPORT,
    EXPORT_ID,
    INPUTS,
    SOURCE_RELEASE_TAG,
    SOURCE_REVISION,
)


EXPECTED = {
    (5, "float32"): (2.2413629999999998, 0.24319999665021896),
    (5, "float64"): (5.5076979999999995, 1.7028799653053284),
    (11, "float32"): (9.315638, 0.9917599856853485),
    (11, "float64"): (19.6736755, 5.955423831939697),
    (15, "float32"): (15.5638755, 2.6505759954452515),
    (15, "float64"): (29.556987499999998, 20.22553539276123),
}


def main() -> None:
    manifest_path = EXPORT / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "gradflow-academic-export-v6"
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
    assert data["schema"] == "gradflow-academic-paper-data-v6"
    assert data["export_id"] == EXPORT_ID
    assert data["source_revision"] == SOURCE_REVISION
    assert data["second_machine"]["unity"]["machine"]["gpu"] == "Tesla M40 24GB"

    moody = data["second_machine"]["moody"]
    assert moody["status"] == "fail_needs_investigation"
    assert not moody["formal_all_sentinels_gate_passed"]
    assert moody["scientific_replication_complete"]
    assert moody["machine"]["gpu"] == "NVIDIA GeForce RTX 4070 SUPER"
    assert moody["machine"]["gpu_capability"] == [8, 9]
    assert moody["pytest"] == {"passed": 351, "skipped": 12, "failed": 4}
    assert moody["a2"]["workers"] == 36
    assert not moody["a2"]["admission_failures"]
    assert not moody["a2"]["graph_failures"]
    assert moody["a2"]["graph_contract_passed"]
    assert moody["a2"]["binary32_cuda_materially_useful"]
    assert moody["a3"]["derivative_gate_passed"]
    assert moody["a3"]["inverse_gate_passed"]
    assert all(
        device["compiled_admitted"]
        and device["compiled_unique_graphs"] == 1
        and device["compiled_graph_break_count"] == 0
        for device in moody["a3"]["devices"].values()
    )
    for cell in moody["a2"]["cells"]:
        cpu, cuda = EXPECTED[(cell["order"], cell["dtype"])]
        assert cell["fastest_cpu_lane"] == "compiled"
        assert cell["fastest_cuda_lane"] == "compiled"
        assert cell["fastest_cpu_median_ms"] == cpu
        assert cell["fastest_cuda_median_ms"] == cuda
        assert cell["fastest_cuda_over_fastest_cpu"] == cuda / cpu
        assert cell["fastest_cpu_over_fastest_cuda"] == cpu / cuda
    assert data["backend_identity"]["system"] == "gradflow"
    print(f"GradFlow academic export verified: {EXPORT_ID}")


if __name__ == "__main__":
    main()
