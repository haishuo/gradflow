#!/usr/bin/env python3
"""Independently verify the immutable Phase-6F process-entry bakeoff."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6f_performance_20260829"
AGGREGATE = RESULTS / "benchmark.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(ratios: list[float]) -> str:
    if len(ratios) != 3:
        return "unresolved"
    if all(value < 1.0 / 1.05 for value in ratios):
        return "confirmed_numerator_win"
    if all(value > 1.05 for value in ratios):
        return "confirmed_denominator_win"
    if all(1.0 / 1.05 <= value <= 1.05 for value in ratios):
        return "practical_equivalence_5_percent"
    return "unresolved"


def verify_checksums() -> None:
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = RESULTS / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative


def main() -> None:
    verify_checksums()
    payload = json.loads(AGGREGATE.read_text())
    assert payload["protocol_commit"] == "c3a19eb"
    assert all(item["passed"] for item in payload["prerequisites"])
    qualification = (
        ROOT / "experiments/fd_fv_euler/results/phase_6f_qualification_20260829/qualification.json"
    )
    phase6d = ROOT / "experiments/fd_fv_euler/results/phase_6d_20260829/benchmark.json"
    assert sha256(qualification) == payload["qualification_sha256"]
    assert sha256(phase6d) == payload["phase6d_sha256"]
    archive = Path(json.loads(qualification.read_text())["prepared_cache"]["archive"])
    assert sha256(archive) == payload["prepared_cache_archive_sha256"]
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected, relative

    raw = [json.loads(path.read_text()) for path in sorted((RESULTS / "raw").glob("*.json"))]
    assert len(raw) == payload["matrix"]["workers"] == 36
    assert all(item["eligible"] for item in raw)
    assert all(item["authority_parity"]["passed"] for item in raw)
    assert all(item["oracle"]["passed"] for item in raw)
    assert all(item["diagnostics"]["completed"] for item in raw)
    assert all(item["terminal_state_cuda_before_materialization"] for item in raw)
    assert all(not item["prepared_cache_restoration_timed"] for item in raw)
    for item in raw:
        assert math.isfinite(item["process_launch_to_exit_seconds"])
        assert item["process_launch_to_exit_seconds"] > 0.0
        stem = f"{item['endpoint']}_{item['problem']}_{item['method']}_r{item['replicate']}"
        array = RESULTS / "arrays" / f"{stem}.npy"
        assert sha256(array) == item["array_file_sha256"]

    phase6d_payload = json.loads(phase6d.read_text())
    for summary in payload["summaries"]:
        problem = summary["problem"]
        method = summary["method"]
        cpu_mode = "eager" if problem == "sod" else "compiled"
        cpu = sorted(
            (
                item
                for item in phase6d_payload["shock_records"]
                if item["problem"] == problem
                and item["method"] == method
                and item["cells"] == 800
                and item["device"] == "cpu"
                and item["mode"] == cpu_mode
                and item["eligible"]
            ),
            key=lambda item: item["replicate"],
        )
        assert len(cpu) == 3
        durations = {
            "cpu": [item["process_launch_to_exit_seconds"] for item in cpu]
        }
        for endpoint in ("cuda_jit", "aot_host", "aot_tensor"):
            selected = sorted(
                (
                    item
                    for item in raw
                    if item["problem"] == problem
                    and item["method"] == method
                    and item["endpoint"] == endpoint
                ),
                key=lambda item: item["replicate"],
            )
            assert len(selected) == 3
            durations[endpoint] = [
                item["process_launch_to_exit_seconds"] for item in selected
            ]
        assert durations == summary["durations_seconds"]
        for key, comparison in summary["comparisons"].items():
            numerator, denominator = key.split("_over_", 1)
            ratios = [
                left / right
                for left, right in zip(durations[numerator], durations[denominator])
            ]
            assert ratios == comparison["paired_ratios"]
            assert comparison["median_ratio"] == (
                statistics.median(durations[numerator])
                / statistics.median(durations[denominator])
            )
            assert decision(ratios) == comparison["decision"]

    assert payload["all_workers_eligible"]
    assert payload["performance_measurements_collected"]
    assert not payload["production_sources_modified"]
    assert not payload["dveb_modified"]
    assert not payload["publication_claim"]
    print("Phase 6F performance verification passed (36/36 workers).")


if __name__ == "__main__":
    main()
