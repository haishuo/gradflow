#!/usr/bin/env python3
"""Verify the frozen ordinary-PyTorch face-ownership evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(evidence: Path) -> None:
    for line in (evidence / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        actual = sha256(evidence / relative)
        assert actual == expected, f"checksum mismatch for {relative}: {actual}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()
    verify_checksums(evidence)

    screen = json.loads((evidence / "screen.json").read_text())
    assert screen["schema"] == "gradflow-face-ownership-screen-v1"
    assert screen["complete"]
    assert screen["completed_configurations"] == 12
    assert screen["canonical_source_changed"] is False
    assert screen["candidate_backend_admitted"] is False
    configurations = screen["configurations"]
    assert len(configurations) == 12

    one_d = [record for record in configurations if record["dimensions"] == 1]
    three_d = [record for record in configurations if record["dimensions"] == 3]
    assert len(one_d) == 4
    assert len(three_d) == 8
    for record in one_d:
        assert record["status"] == "failed_compile_gate"
        assert record["passed_precompile_gate"]
        assert not record["passed_compile_gate"]
        assert "timing" not in record

    for record in three_d:
        assert record["status"] == "complete"
        assert record["passed_precompile_gate"]
        assert record["passed_compile_gate"]
        for mode in ("eager", "compiled"):
            timing = record["timing"][mode]
            analysis = timing["analysis"]
            assert timing["warmups_per_representation"] == 5
            assert timing["randomized_pair_blocks"] == 20
            assert len(timing["blocks"]) == 20
            assert analysis["decision"] == "face_once_win"
            ratio = analysis["paired_face_over_cell_ratio"]
            assert ratio["median"] < 0.95
            assert ratio["bootstrap_median_95_ci"][1] < 1.0
            assert len(ratio["values"]) == 20

    by_key = {
        (item["order"], item["dtype"], item["n"]): item for item in three_d
    }
    primary = by_key[(5, "float64", 96)]["timing"]["compiled"]["analysis"]
    assert primary["paired_face_over_cell_ratio"]["median"] == 0.3627117389512064
    largest = by_key[(5, "float32", 128)]["timing"]["compiled"]
    assert largest["analysis"]["paired_face_over_cell_ratio"]["median"] == (
        0.42121617259613375
    )
    assert largest["incremental_peak_allocated_bytes"] == {
        "cell_recompute": 679_477_760,
        "face_once": 327_156_224,
    }

    diagnostic = json.loads((evidence / "compile_1d_diagnostic.json").read_text())
    assert diagnostic["schema"] == (
        "gradflow.face_ownership_1d_compile_diagnostic.v1"
    )
    assert diagnostic["complete"]
    assert len(diagnostic["records"]) == 24
    assert all(item["alpha"]["absolute_error"] == 0.0 for item in diagnostic["records"])
    for item in diagnostic["records"]:
        for representation in ("face_once", "cell_recompute"):
            compilation = item["compilation"][representation]
            assert compilation["unique_graphs"] == 1
            assert compilation["graph_break_count"] == 0
    f32 = [item for item in diagnostic["records"] if item["dtype"] == "float32"]
    assert all(
        not item["compiled_versus_eager"]["face_once"]["passed"] for item in f32
    )
    f64_n65 = [
        item
        for item in diagnostic["records"]
        if item["dtype"] == "float64" and item["n"] == 65_536
    ]
    assert len(f64_n65) == 2
    assert all(
        item["compiled_versus_eager"]["face_once"]["passed"]
        and item["compiled_versus_eager"]["cell_recompute"]["passed"]
        for item in f64_n65
    )
    assert all(
        not item["compiled_versus_eager"]["face_once"]["passed"]
        for item in diagnostic["records"]
        if item["dtype"] == "float64" and item["n"] >= 262_144
    )
    print("Face-ownership screen evidence and bounded conclusion verify.")


if __name__ == "__main__":
    main()
