#!/usr/bin/env python3
"""Verify Academic A4 rc2 payload integrity and unresolved release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
TAG = "academic-v0.1.0-rc2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_content(reference: str, relative: str) -> bytes:
    return subprocess.run(
        ("git", "show", f"{reference}:{relative}"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--ref")
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()
    index = json.loads((evidence / "artifact_index.json").read_text())
    assert index["schema"] == "gradflow-academic-a4-artifact-index-v2"
    assert index["source_tree_clean"]
    assert index["release_candidate_tag"] == TAG
    assert index["previous_release_candidate_tag"] == "academic-v0.1.0-rc1"
    assert index["file_count"] == len(index["files"])
    assert index["total_bytes"] == sum(item["bytes"] for item in index["files"])
    paths = [item["path"] for item in index["files"]]
    assert paths == sorted(paths) and len(paths) == len(set(paths))
    for item in index["files"]:
        if arguments.ref:
            content = git_content(arguments.ref, item["path"])
            assert len(content) == item["bytes"], item["path"]
            assert hashlib.sha256(content).hexdigest() == item["sha256"], item["path"]
        else:
            path = ROOT / item["path"]
            assert path.stat().st_size == item["bytes"], item["path"]
            assert sha256(path) == item["sha256"], item["path"]

    rights = index["redistribution"]
    assert rights["status"] == "unresolved_public_release_blockers"
    assert rights["top_level_project_license_present"] is False
    assert rights["reference_prefixes_requiring_permission_or_exclusion"] == [
        "references/gottlieb_matlab/", "references/jiang_shu_fortran/"
    ]
    assert sum(
        any(path.startswith(prefix) for prefix in rights["reference_prefixes_requiring_permission_or_exclusion"])
        for path in paths
    ) == 9
    assert index["primary_toolchain_evidence"]["torch"] == "2.13.0+cu130"

    def text_at(relative: str) -> str:
        if arguments.ref:
            return git_content(arguments.ref, relative).decode()
        return (ROOT / relative).read_text()

    assert TAG in text_at("docs/ACADEMIC_A4_RC2_PROTOCOL.md")
    assert TAG in text_at("docs/ACADEMIC_A4_SECOND_MACHINE_PACKET.md")
    assert TAG in text_at("docs/ACADEMIC_A4_EXTERNAL_REVIEW_PACKET.md")
    assert "review not yet performed" in text_at("docs/ACADEMIC_A4_EXTERNAL_REVIEW_PACKET.md")
    assert "No top-level `LICENSE`" in text_at("docs/ACADEMIC_A4_RIGHTS_STATUS.md")
    comparison = json.loads(
        text_at("experiments/academic_u5/evidence/u5_20260831/comparison.json")
    )
    assert comparison["u4f"]["batched_cpu_compiler_failure_fixed"]
    assert comparison["claim_boundary"]["universal_backend_winner_claimed"] is False
    print("Academic A4 rc2 payload hashes and release-policy gates verify.")


if __name__ == "__main__":
    main()

