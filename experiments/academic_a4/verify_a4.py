#!/usr/bin/env python3
"""Offline integrity and release-policy verification for Academic A4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()

    index_path = evidence / "artifact_index.json"
    index = json.loads(index_path.read_text())
    assert index["schema"] == "gradflow-academic-a4-artifact-index-v1"
    assert index["source_tree_clean"]
    assert index["release_candidate_tag"] == "academic-v0.1.0-rc1"
    assert index["file_count"] == len(index["files"])
    assert index["total_bytes"] == sum(item["bytes"] for item in index["files"])
    paths = [item["path"] for item in index["files"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    for item in index["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), f"missing indexed file: {item['path']}"
        assert path.stat().st_size == item["bytes"], item["path"]
        assert sha256(path) == item["sha256"], item["path"]

    rights = index["redistribution"]
    assert rights["status"] == "unresolved_public_release_blockers"
    assert rights["top_level_project_license_present"] is False
    assert rights["reference_prefixes_requiring_permission_or_exclusion"] == [
        "references/gottlieb_matlab/",
        "references/jiang_shu_fortran/",
    ]
    indexed_references = [
        path
        for path in paths
        if any(
            path.startswith(prefix)
            for prefix in rights["reference_prefixes_requiring_permission_or_exclusion"]
        )
    ]
    assert len(indexed_references) == 9

    checksum_path = evidence / "SHA256SUMS"
    if checksum_path.exists():
        for line in checksum_path.read_text().splitlines():
            expected, relative = line.split("  ", maxsplit=1)
            assert sha256(evidence / relative) == expected, relative

    environment = json.loads((ROOT / "environments/academic-a4-forge.json").read_text())
    assert environment["schema"] == "gradflow-academic-a4-environment-v1"
    assert environment["gpu"]["model"] == "NVIDIA GeForce RTX 5070 Ti"
    assert environment["packages"]["torch"] == "2.9.0.dev20250705+cu128"

    protocol = (ROOT / "docs/ACADEMIC_A4_PROTOCOL.md").read_text()
    rights_document = (ROOT / "docs/ACADEMIC_A4_RIGHTS_STATUS.md").read_text()
    external = (ROOT / "docs/ACADEMIC_A4_EXTERNAL_REVIEW_PACKET.md").read_text()
    replication = (ROOT / "docs/ACADEMIC_A4_SECOND_MACHINE_PACKET.md").read_text()
    assert "Only state 3 closes A4" in protocol
    assert "No top-level `LICENSE`" in rights_document
    assert "review not yet performed" in external
    assert "physically distinct machine" in replication
    print("Academic A4 payload hashes and release-policy gates verify.")


if __name__ == "__main__":
    main()
