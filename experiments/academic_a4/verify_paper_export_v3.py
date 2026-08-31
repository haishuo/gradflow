#!/usr/bin/env python3
"""Verify the U4-E-complete downstream academic export."""

from __future__ import annotations

import json

from export_paper_data import ROOT, sha256
from export_paper_data_v3 import EXPORT, EXPORT_ID, INPUTS, SOURCE_REVISION


def main() -> None:
    manifest_path = EXPORT / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "gradflow-academic-export-v3"
    assert manifest["export_id"] == EXPORT_ID
    assert manifest["source_revision"] == SOURCE_REVISION
    assert manifest["source_release_tag"] is None

    for path in INPUTS.values():
        relative = str(path.relative_to(ROOT))
        assert sha256(path) == manifest["input_sha256"][relative]

    dataset = EXPORT / "paper_data.json"
    record = manifest["outputs"]["paper_data.json"]
    assert dataset.stat().st_size == record["bytes"]
    assert sha256(dataset) == record["sha256"]

    data = json.loads(dataset.read_text())
    assert data["schema"] == "gradflow-academic-paper-data-v3"
    assert data["export_id"] == EXPORT_ID
    assert data["source_revision"] == SOURCE_REVISION
    external = data["external_baseline"]
    assert sum(row["all_lanes_admitted"] for row in external["u4c_admission_surface"]) == 1
    assert external["u4e_qualification"]["decision"] == "all_six_lanes_qualified"
    assert all(lane["passed"] for lane in external["u4e_qualification"]["lanes"].values())
    assert external["u4e_resident"]["cpu"]["overall_winner"] == "dveb"
    assert external["u4e_resident"]["cuda"]["overall_winner"] == "dveb"
    assert external["u4e_schedule"]["cpu"]["synchronized"] == 0
    assert external["u4e_schedule"]["cuda"]["synchronized"] == 0
    assert data["backend_identity"]["legacy_evidence_alias"]["gradflow"] == (
        "pytorch_inductor"
    )
    print(f"GradFlow academic export verified: {EXPORT_ID}")


if __name__ == "__main__":
    main()
