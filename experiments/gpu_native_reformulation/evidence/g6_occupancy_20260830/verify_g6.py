#!/usr/bin/env python3
"""Verify the frozen G6 occupancy evidence and negative conclusion."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics


CANDIDATES = tuple(
    f"b{block}_{policy}"
    for block in (64, 128, 256)
    for policy in ("u", "r112", "r96")
)
LANES = ("frozen_r6q", *CANDIDATES)
CONFIGURATIONS = {(n, steps) for n in (64, 128, 256) for steps in (1, 10)}


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


def verify_raw_input(evidence: Path) -> None:
    digest = hashlib.sha256()
    with gzip.open(evidence / "profiler_input_n128.f32.gz", "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == (
        "05ff02323c83be003761e34809fc8168149b1434cb82d02dfd6fc7cd608ff70e"
    )


def paired_median(configuration: dict, candidate: str, control: str) -> float:
    return statistics.median(
        record["lanes"][candidate]["resident_seconds"]
        / record["lanes"][control]["resident_seconds"]
        for record in configuration["records"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()
    verify_checksums(evidence)
    verify_raw_input(evidence)

    gate_path = evidence / "gate/forward_gate.json"
    gate = json.loads(gate_path.read_text())
    assert gate["schema"] == "gradflow-g6-occupancy-forward-gate-v1"
    assert gate["passed"]
    assert gate["candidate_backend_admitted"] is False
    assert tuple(gate["candidate_order"]) == CANDIDATES
    assert tuple(gate["passing_candidates"]) == CANDIDATES
    assert len(gate["cases"]) == 5
    assert len(gate["cases"]) * len(CANDIDATES) == 45
    for case in gate["cases"]:
        for candidate in case["candidates"]:
            assert candidate["passed"]
            assert candidate["comparison"]["bitwise_identical"]
            assert candidate["comparison"]["maximum_absolute_difference"] == 0.0
            assert candidate["comparison"]["rms_difference"] == 0.0

    resources = gate["resource_metadata"]
    for candidate in CANDIDATES:
        metadata = resources[candidate]
        policy = candidate.rsplit("_", maxsplit=1)[1]
        expected_registers = {"u": 128, "r112": 112, "r96": 96}[policy]
        assert metadata["compiled_face_registers_per_thread"] == expected_registers
        assert metadata["compiled_face_local_bytes_per_thread"] == (
            40 if policy == "r96" else 0
        )
        expected_occupancy = (
            41.666666666666664
            if policy == "r96" and not candidate.startswith("b256")
            else 33.333333333333336
        )
        assert metadata["face_theoretical_occupancy_percent"] == expected_occupancy
        archived_binary = evidence / "binaries" / f"gradflow_g6_{candidate}"
        assert sha256(archived_binary) == gate["artifacts"][candidate]["executable"][
            "sha256"
        ]
    assert sha256(evidence / "binaries/gradflow_r6q") == gate["artifacts"][
        "frozen_r6q"
    ]["executable"]["sha256"]

    for candidate in ("b64_r96", "b128_r96", "b256_r96"):
        compiler = (evidence / "compiler_logs" / f"{candidate}.log").read_text()
        assert "40 bytes stack frame, 80 bytes spill stores, 88 bytes spill loads" in compiler
    assert "historical JSON is retained" in (
        evidence / "RESOURCE_RECORD_CORRECTION.md"
    ).read_text()

    campaign = json.loads((evidence / "campaign.json").read_text())
    assert campaign["schema"] == "gradflow-g6-occupancy-performance-v1"
    assert campaign["candidate_backend_admitted"] is False
    assert campaign["forward_gate"]["sha256"] == sha256(gate_path)
    assert campaign["protocol"] == {
        "bootstrap_resamples": 20_000,
        "lane_order": list(LANES),
        "random_seed": 20260830,
        "randomized_complete_lane_blocks": 30,
        "sizes": [64, 128, 256],
        "steps": [1, 10],
        "thermal_stop_c": 80,
        "warmup_processes_per_lane": 3,
    }
    observed = set()
    configurations = {}
    for configuration in campaign["configurations"]:
        key = (configuration["n"], configuration["steps"])
        observed.add(key)
        configurations[key] = configuration
        assert len(configuration["records"]) == 30
        for record in configuration["records"]:
            assert sorted(record["order"]) == sorted(LANES)
            assert record["telemetry_before"]["temperature_c"] < 80
            assert record["telemetry_after"]["temperature_c"] < 80
            for lane in LANES:
                assert record["lanes"][lane]["native"]["finite"]
                assert record["lanes"][lane]["resident_seconds"] > 0.0
    assert observed == CONFIGURATIONS

    decision = campaign["primary_decision"]
    assert decision["fastest_passing_candidate_by_frozen_rule"] == "b256_r112"
    assert decision["any_meaningful_occupancy_improvement"] is False
    assert decision["backend_qualification_implication"] is False
    assert all(
        not item["meaningful_improvement"]
        for item in decision["candidate_results"].values()
    )

    # The rebuilt uncapped 256-thread lane isolates the G6 changes from the
    # old executable's first-event setup asymmetry.
    for configuration in configurations.values():
        cap112 = paired_median(configuration, "b256_r112", "b256_u")
        assert 0.995 < cap112 < 1.005
        if configuration["n"] >= 128:
            assert paired_median(configuration, "b256_r96", "b256_u") > 1.015
    assert paired_median(configurations[(256, 10)], "b64_u", "b256_u") > 1.05

    frozen_nsys = (evidence / "nsys_frozen_kernel_summary.csv").read_text()
    candidate_nsys = (evidence / "nsys_b256_r112_kernel_summary.csv").read_text()
    assert "2526008" in frozen_nsys and "face_kernel" in frozen_nsys
    assert "2527223" in candidate_nsys and "face_kernel" in candidate_nsys
    for lane in ("frozen_r6q", "b256_r112"):
        log = (evidence / f"ncu_{lane}_n128_s1.log").read_text()
        assert "ERR_NVGPUCTRPERM" not in log
        assert log.count('Profiling "face_kernel"') == 3
        assert (evidence / f"ncu_{lane}_n128_s1.ncu-rep").stat().st_size > 0
    counters = json.loads((evidence / "ncu_summary.json").read_text())
    assert counters["schema"] == "gradflow-g6-ncu-basic-comparison-v1"
    frozen_counters = counters["lanes"]["frozen_r6q"]
    candidate_counters = counters["lanes"]["b256_r112"]
    assert frozen_counters["launch_invariants"]["registers_per_thread"] == 128.0
    assert candidate_counters["launch_invariants"]["registers_per_thread"] == 112.0
    for lane in (frozen_counters, candidate_counters):
        assert lane["launch_invariants"]["theoretical_occupancy_percent"] == 33.33
        assert 32.0 < lane["face_summary"]["median_achieved_occupancy_percent"] < 33.0
        assert 72.0 < lane["face_summary"]["median_compute_sm_throughput_percent"] < 74.0
    assert 0.99 < counters["candidate_over_frozen"]["face_total_duration_ratio"] < 1.01
    print("G6 evidence and no-occupancy-improvement conclusion verify.")


if __name__ == "__main__":
    main()
