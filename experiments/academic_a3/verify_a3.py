#!/usr/bin/env python3
"""Offline checksum and semantic verification for Academic A3."""

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

    campaign = json.loads((evidence / "campaign.json").read_text())
    assert campaign["schema"] == "gradflow-academic-a3-campaign-v1"
    assert campaign["complete"]
    assert campaign["protocol_commit"] == "faa83d7"
    assert not campaign["canonical_source_changed"]
    assert campaign["environment"]["cuda_available"]
    for relative, expected in campaign["source_sha256"].items():
        assert sha256(ROOT / relative) == expected

    derivative = campaign["derivative_gate"]
    assert derivative["registered_window_passed"]
    assert len(derivative["records"]) == 8
    assert all(record["finite"] for record in derivative["records"])
    assert [record["step"] for record in derivative["records"]] == [
        1.0e-1,
        1.0e-2,
        1.0e-3,
        1.0e-4,
        1.0e-5,
        1.0e-6,
        1.0e-7,
        1.0e-8,
    ]
    assert min(record["relative_error"] for record in derivative["records"]) < 1.0e-10

    inverse = campaign["inverse_gate"]
    assert inverse["passed"]
    assert inverse["autograd_golden_speed_difference"] <= 2.0e-6
    assert inverse["autograd_truth_error"] <= 2.0e-3
    assert inverse["terminal_over_initial_objective"] <= 1.0e-4
    assert len(inverse["objective_scan"]) == 201
    local_minima = [
        index
        for index in range(1, 200)
        if inverse["objective_scan"][index]["objective"]
        < inverse["objective_scan"][index - 1]["objective"]
        and inverse["objective_scan"][index]["objective"]
        < inverse["objective_scan"][index + 1]["objective"]
    ]
    assert local_minima == [120]
    assert inverse["objective_scan"][120]["speed"] == 1.1

    resolution = campaign["resolution_study"]
    assert [record["n"] for record in resolution] == [64, 128, 256]
    assert [record["steps"] for record in resolution] == [8, 16, 32]
    errors = [record["truth_error"] for record in resolution]
    assert errors[0] > errors[1] > errors[2]

    for device in ("cpu", "cuda"):
        wrapper = campaign["benchmarks"][device]
        assert wrapper["returncode"] == 0
        record = wrapper["record"]
        assert record["status"] == "complete"
        assert record["eager"]["admitted"]
        assert record["compiled"]["admitted"]
        assert record["compiled"]["graph"]["unique_graphs"] == 1
        assert record["compiled"]["graph"]["graph_break_count"] == 0
        for lane in ("eager", "compiled"):
            timing = record["timings"][lane]
            assert timing["forward_ms"]["count"] == 20
            assert timing["objective_and_gradient_ms"]["count"] == 20
            assert timing["reverse_mode_over_forward_median"] > 1.0

    results = (ROOT / "docs/ACADEMIC_A3_RESULTS.md").read_text()
    assert "1.1000008465" in results
    assert "derivative-free" in results
    assert "504.493 s" in results and "339.093 s" in results
    assert "does not cover shocks" in results
    print("Academic A3 inverse, gradient, compiler, and timing evidence verify.")


if __name__ == "__main__":
    main()
