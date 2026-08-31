#!/usr/bin/env python3
"""Verify the Unity-extended downstream academic export."""

from __future__ import annotations

import json

from export_paper_data import ROOT, sha256
from export_paper_data_v5 import (
    EXPORT,
    EXPORT_ID,
    INPUTS,
    SOURCE_RELEASE_TAG,
    SOURCE_REVISION,
)


def main() -> None:
    manifest_path = EXPORT / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "gradflow-academic-export-v5"
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
    assert data["schema"] == "gradflow-academic-paper-data-v5"
    assert data["export_id"] == EXPORT_ID
    assert data["source_revision"] == SOURCE_REVISION
    unity = data["second_machine"]["unity"]
    assert unity["status"] == "fail_needs_investigation"
    assert not unity["closes_second_machine_gate"]
    assert unity["machine"]["gpu"] == "Tesla M40 24GB"
    assert unity["machine"]["gpu_capability"] == [5, 2]
    assert unity["pytest"] == {"passed": 344, "skipped": 12, "failed": 11}
    assert unity["a1_completed"]
    assert unity["a3"]["derivative_gate_passed"]
    assert unity["a3"]["inverse_gate_passed"]
    assert unity["a3"]["cuda_eager_admitted"]
    assert not unity["a3"]["cuda_compiled_admitted"]
    assert "GPUTooOldForTriton" in unity["a3"]["cuda_compiled_error"]
    assert unity["a2"]["workers"] == 36
    assert unity["a2"]["compiled_cuda_admission_failures"] == 18
    assert not unity["a2"]["graph_contract_passed"]
    assert not unity["a2"]["binary32_cuda_materially_useful"]
    assert all(
        1.0 < cell["fastest_cuda_over_fastest_cpu"] < 2.0
        for cell in unity["a2"]["cells"]
    )
    assert data["backend_identity"]["system"] == "gradflow"
    print(f"GradFlow academic export verified: {EXPORT_ID}")


if __name__ == "__main__":
    main()
