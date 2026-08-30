#!/usr/bin/env python3
"""Verify the frozen downstream academic export and its provenance hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "experiments/academic_a4/exports/academic-v0.1.0-rc1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((EXPORT / "export_manifest.json").read_text())
    assert manifest["schema"] == "gradflow-academic-export-v1"
    assert manifest["release_tag"] == "academic-v0.1.0-rc1"
    assert manifest["release_commit"] == (
        "99a2a806fdaedb6cc32cdad2d621144d014865de"
    )

    for relative, expected in manifest["input_sha256"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative

    for relative, record in manifest["outputs"].items():
        path = EXPORT / relative
        assert path.is_file(), relative
        assert path.stat().st_size == record["bytes"], relative
        assert sha256(path) == record["sha256"], relative

    dataset = json.loads((EXPORT / "paper_data.json").read_text())
    assert dataset["schema"] == "gradflow-academic-paper-data-v1"
    assert dataset["release_candidate"] == manifest["release_tag"]
    assert dataset["input_sha256"] == manifest["input_sha256"]
    print("GradFlow academic export verified")


if __name__ == "__main__":
    main()
