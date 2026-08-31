#!/usr/bin/env python3
"""Verify the rc2 stable-toolchain downstream academic export."""

from __future__ import annotations

import json

from export_paper_data import ROOT, sha256
from export_paper_data_v4 import (
    EXPORT,
    EXPORT_ID,
    INPUTS,
    SOURCE_RELEASE_TAG,
    SOURCE_REVISION,
)


def main() -> None:
    manifest_path = EXPORT / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "gradflow-academic-export-v4"
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
    assert data["schema"] == "gradflow-academic-paper-data-v4"
    assert data["export_id"] == EXPORT_ID
    assert data["release_candidate"] == SOURCE_RELEASE_TAG
    assert data["source_revision"] == SOURCE_REVISION
    assert data["primary_toolchain"] == "stable_pytorch_2_13"
    stable = data["stable_toolchain"]
    assert stable["environment"]["torch"] == "2.13.0+cu130"
    assert len(stable["performance_64cube"]) == 12
    assert stable["speedup_ranges"]["float64"]["resident"][0] < 1.0
    assert stable["u4f"]["batched_cpu_compiler_failure_fixed"]
    assert [
        cell["devices"]["cuda"]["stable_decision"]
        for cell in stable["u4f"]["cells"]
    ] == [
        "dveb_native_win", "dveb_native_win", "dveb_native_win",
        "pytorch_inductor_win", "pytorch_inductor_win", "pytorch_inductor_win",
    ]
    assert data["artifact"]["cleanroom_all_passed"]
    assert data["artifact"]["cleanroom_command_count"] == 15
    assert data["artifact"]["cleanroom_tested_commit"] == SOURCE_REVISION
    assert data["artifact"]["cleanroom_network_used"] is False
    assert data["backend_identity"]["system"] == "gradflow"
    print(f"GradFlow academic export verified: {EXPORT_ID}")


if __name__ == "__main__":
    main()

