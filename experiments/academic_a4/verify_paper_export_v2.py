#!/usr/bin/env python3
"""Verify the reporting-complete downstream academic export."""

from __future__ import annotations

import json

from export_paper_data import INPUTS, ROOT, sha256


EXPORT_ID = "academic-v0.1.0-rc1-paper-v2"
EXPORT = ROOT / "experiments/academic_a4/exports" / EXPORT_ID


def main() -> None:
    manifest_path = EXPORT / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "gradflow-academic-export-v2"
    assert manifest["export_id"] == EXPORT_ID
    assert manifest["source_release_tag"] == "academic-v0.1.0-rc1"
    assert manifest["source_release_commit"] == (
        "99a2a806fdaedb6cc32cdad2d621144d014865de"
    )

    for path in INPUTS.values():
        relative = str(path.relative_to(ROOT))
        assert sha256(path) == manifest["input_sha256"][relative]

    dataset = EXPORT / "paper_data.json"
    record = manifest["outputs"]["paper_data.json"]
    assert dataset.stat().st_size == record["bytes"]
    assert sha256(dataset) == record["sha256"]

    data = json.loads(dataset.read_text())
    assert data["schema"] == "gradflow-academic-paper-data-v2"
    assert data["export_id"] == EXPORT_ID
    assert len(data["performance_64cube"]) == 12
    assert len(data["isolated_cache_deployment"]) == 6
    assert len(data["aot_packages"]) == 3
    assert len(data["differentiation_benchmarks"]) == 2
    assert all(row["count"] == 3 for row in data["isolated_cache_deployment"])
    assert data["inverse_problem"]["autograd"]["closure_evaluations"] == 11
    assert data["inverse_problem"]["golden_section"]["objective_evaluations"] == 67
    print(f"GradFlow academic export verified: {EXPORT_ID}")


if __name__ == "__main__":
    main()
