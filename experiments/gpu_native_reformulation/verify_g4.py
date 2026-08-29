#!/usr/bin/env python3
"""Verify the frozen G4 schedule-control performance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SIZES = (8, 16, 32, 64, 128, 192, 256)
STEPS = (1, 10)
UPSTREAM_HASHES = {
    "build.sh": "811d840500dbcd82ea48e0718d7d45d3bfbfb0225405c9a27768b2542c8b842f",
    "cpu.cpp": "d7282983ac5861b17a75daed3cc9457f9cefb75414a58309ac5112a535e7041c",
    "cuda.cu": "c3964d31399bb4d2b68bdd2c33a70aa5263ea3b370a3d94e2dde2f169dfcfb6d",
    "main.cpp": "c30918f2ec4bb80eb7961c1b86f8149277871ad2d47ddecb16593b417f9deda6",
    "runner.h": "56b07cad0b63ba8425d1e8b8b745c94e3a98e918c6fd32832e86cc9ea2252aaa",
    "shu_math.h": "125dd8ec0d60cc4c965e1a8f804b12ae471cf73850e3484520cc400ae0db9009",
}


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
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {relative}: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()
    verify_checksums(evidence)

    for name, expected in UPSTREAM_HASHES.items():
        assert sha256(evidence / "control_upstream" / name) == expected
    for unchanged in ("cpu.cpp", "runner.h", "shu_math.h"):
        assert (evidence / "control_upstream" / unchanged).read_bytes() == (
            evidence / "control_source" / unchanged
        ).read_bytes()

    result = json.loads((evidence / "campaign.json").read_text())
    assert result["schema"] == "gradflow-g4-face-once-performance-v1"
    assert result["candidate_backend_admitted"] is False
    assert result["validity"]["passed"]
    assert result["validity"]["maximum_absolute_difference"] <= 2.0e-5
    assert result["protocol"] == {
        "bootstrap_resamples": 20_000,
        "paired_repetitions": 30,
        "random_seed": 20260829,
        "sizes": list(SIZES),
        "steps": list(STEPS),
        "thermal_stop_c": 80,
        "warmup_processes_per_lane": 3,
    }
    assert len(result["configurations"]) == len(SIZES) * len(STEPS)

    observed = set()
    for configuration in result["configurations"]:
        key = (configuration["n"], configuration["steps"])
        observed.add(key)
        assert len(configuration["records"]) == 30
        for pair in configuration["records"]:
            assert sorted(pair["order"]) == ["cell_recompute", "face_once"]
            assert pair["telemetry_before"]["temperature_c"] < 80
            assert pair["telemetry_after"]["temperature_c"] < 80
            for lane in ("face_once", "cell_recompute"):
                assert pair["lanes"][lane]["native"]["finite"]
                assert pair["lanes"][lane]["resident_seconds"] > 0.0
                assert pair["lanes"][lane]["external_fresh_process_seconds"] > 0.0
    assert observed == {(n, steps) for n in SIZES for steps in STEPS}

    primary = {
        (item["n"], item["steps"]): item
        for item in result["configurations"]
        if item["primary_point"]
    }
    assert set(primary) == {(128, 1), (128, 10)}
    for item in primary.values():
        ratio = item["endpoints"]["resident_seconds"][
            "paired_cell_over_face_ratio"
        ]
        assert ratio["median"] > 1.10
        assert ratio["bootstrap_median_95_ci"][0] > 1.0
        assert item["schedule_hypothesis_supported_at_point"]
    assert result["primary_decision"]["schedule_hypothesis_supported"]
    assert result["primary_decision"]["backend_qualification_implication"] is False

    face_summary = (evidence / "nsys_face_once_kernel_summary.csv").read_text()
    cell_summary = (evidence / "nsys_cell_recompute_kernel_summary.csv").read_text()
    assert "face_kernel" in face_summary
    assert "rhs_kernel" in cell_summary
    assert "ERR_NVGPUCTRPERM" in (evidence / "PROFILER_RECORD.md").read_text()
    print("G4 evidence and non-admission schedule conclusion verify.")


if __name__ == "__main__":
    main()
