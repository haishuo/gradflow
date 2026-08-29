#!/usr/bin/env python3
"""Independently verify the immutable Phase-6G process-entry bakeoff."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6g_performance_20260829"
QUALIFICATION = ROOT / "experiments/fd_fv_euler/results/phase_6g_qualification_20260829/qualification.json"
PHASE6D = ROOT / "experiments/fd_fv_euler/results/phase_6d_20260829/benchmark.json"
PHASE6F = ROOT / "experiments/fd_fv_euler/results/phase_6f_qualification_20260829"
ENDPOINTS = ("cuda_jit", "aot_host_internal", "aot_tensor_internal")


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


def break_even(preparation: float, jit: float, aot: float) -> int | None:
    saving = jit - aot
    return None if saving <= 0.0 else math.ceil(preparation / saving)


def verify_checksums() -> None:
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = RESULTS / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative


def main() -> None:
    verify_checksums()
    payload = json.loads((RESULTS / "benchmark.json").read_text())
    assert payload["protocol_commit"] == "0efdccb"
    assert all(item["passed"] for item in payload["prerequisites"])
    assert sha256(QUALIFICATION) == payload["qualification_sha256"]
    assert sha256(PHASE6D) == payload["phase6d_sha256"]
    assert sha256(PHASE6F / "prepared_cache_manifest.json") == payload[
        "prepared_cache_manifest_sha256"
    ]
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected, relative

    raw = [json.loads(path.read_text()) for path in sorted((RESULTS / "raw").glob("*.json"))]
    assert len(raw) == payload["matrix"]["workers"] == 36
    for item in raw:
        assert item["eligible"]
        assert item["authority_parity"]["passed"]
        assert item["oracle"]["passed"]
        assert item["diagnostics"]["completed"]
        assert item["terminal_state_cuda_before_materialization"]
        assert not item["prepared_cache_restoration_timed"]
        assert math.isfinite(item["process_launch_to_exit_seconds"])
        assert item["process_launch_to_exit_seconds"] > 0.0
        stem = f"{item['endpoint']}_{item['problem']}_{item['method']}_r{item['replicate']}"
        assert sha256(RESULTS / "arrays" / f"{stem}.npy") == item["array_file_sha256"]

    phase6d = json.loads(PHASE6D.read_text())
    phase6f = json.loads((PHASE6F / "qualification.json").read_text())
    cache_seconds = phase6f["preparation"]["process_launch_to_exit_seconds"]
    for summary in payload["summaries"]:
        problem = summary["problem"]
        method = summary["method"]
        cpu_mode = "eager" if problem == "sod" else "compiled"
        cpu = sorted(
            (
                item
                for item in phase6d["shock_records"]
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
        durations = {"cpu": [item["process_launch_to_exit_seconds"] for item in cpu]}
        for endpoint in ENDPOINTS:
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
            durations[endpoint] = [item["process_launch_to_exit_seconds"] for item in selected]
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
            assert comparison["decision"] == decision(ratios)
        jit = statistics.median(durations["cuda_jit"])
        for endpoint, record in summary["break_even_against_fresh_jit"].items():
            aot = statistics.median(durations[endpoint])
            build = record["aot_package_build_seconds"]
            assert record["shared_cache_preparation_seconds"] == cache_seconds
            assert record["jit_minus_aot_median_seconds"] == jit - aot
            assert record["package_only_invocations"] == break_even(build, jit, aot)
            assert record["package_plus_full_cache_prep_invocations"] == break_even(
                build + cache_seconds, jit, aot
            )

    assert payload["all_workers_eligible"]
    assert payload["performance_measurements_collected"]
    assert not payload["production_sources_modified"]
    assert not payload["dveb_modified"]
    assert not payload["publication_claim"]
    print("Phase 6G performance verification passed (36/36 workers).")


if __name__ == "__main__":
    main()
