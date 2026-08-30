#!/usr/bin/env python3
"""Verify the frozen G5 shared-pencil evidence and negative conclusion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SIZES = (32, 64, 128, 192, 256)
STEPS = (1, 10)
LANES = ("shared_pencil", "global_face_once", "cell_recompute")


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

    gate = json.loads((evidence / "gate/forward_gate.json").read_text())
    assert gate["schema"] == "gradflow-g5-shared-pencil-forward-gate-v1"
    assert gate["candidate_backend_admitted"] is False
    assert gate["passed"]
    assert gate["memory"]["passed"]
    assert gate["memory"]["p1_over_r6q"] <= 0.70
    assert len(gate["cases"]) == 5
    for case in gate["cases"]:
        assert case["passed"]
        assert case["comparison"]["bitwise_identical"]
        assert case["comparison"]["maximum_absolute_difference"] == 0.0
        assert case["comparison"]["rms_difference"] == 0.0

    result = json.loads((evidence / "campaign.json").read_text())
    assert result["schema"] == "gradflow-g5-shared-pencil-performance-v1"
    assert result["candidate_backend_admitted"] is False
    assert result["protocol"] == {
        "bootstrap_resamples": 20_000,
        "random_seed": 20260829,
        "randomized_triplets_per_configuration": 30,
        "sizes": list(SIZES),
        "steps": list(STEPS),
        "thermal_stop_c": 80,
        "warmup_processes_per_lane": 3,
    }
    assert len(result["configurations"]) == len(SIZES) * len(STEPS)
    assert sha256(evidence / "gradflow_p1") == result["artifacts"][
        "shared_pencil"
    ]["sha256"]
    assert sha256(evidence / "gradflow_r6q") == result["artifacts"][
        "global_face_once"
    ]["sha256"]

    observed = set()
    primary = {}
    for configuration in result["configurations"]:
        key = (configuration["n"], configuration["steps"])
        observed.add(key)
        assert len(configuration["records"]) == 30
        for triplet in configuration["records"]:
            assert sorted(triplet["order"]) == sorted(LANES)
            assert triplet["telemetry_before"]["temperature_c"] < 80
            assert triplet["telemetry_after"]["temperature_c"] < 80
            for lane in LANES:
                record = triplet["lanes"][lane]
                assert record["native"]["finite"]
                assert record["resident_seconds"] > 0.0
                assert record["external_fresh_process_seconds"] > 0.0
        ratios = configuration["endpoints"]["resident_seconds"]
        assert ratios["paired_shared_over_global_ratio"]["median"] > 1.0
        assert ratios["paired_cell_over_shared_ratio"]["median"] < 1.0
        if configuration["primary_point"]:
            primary[key] = configuration
    assert observed == {(n, steps) for n in SIZES for steps in STEPS}
    assert set(primary) == {(128, 1), (128, 10)}

    for configuration in primary.values():
        endpoints = configuration["endpoints"]["resident_seconds"]
        ratio = endpoints["paired_shared_over_global_ratio"]
        assert ratio["median"] > 1.10
        assert ratio["bootstrap_median_95_ci"][1] > 1.15
        assert endpoints["paired_cell_over_shared_ratio"]["median"] < 1.0
        assert not configuration["timing_criteria_passed_at_point"]
    memory = result["primary_decision"]["memory"]
    assert memory["passed"]
    first_primary = primary[(128, 1)]["records"][0]["lanes"]
    p1_peak = first_primary["shared_pencil"]["native"]["peak_allocated_bytes"]
    r6q_peak = first_primary["global_face_once"]["native"]["peak_allocated_bytes"]
    assert p1_peak / r6q_peak <= 0.70
    assert result["primary_decision"]["individual_timing_results"] == [False, False]
    assert not result["primary_decision"]["successful_memory_recovery_pareto_result"]
    assert result["primary_decision"]["backend_qualification_implication"] is False

    summary = (evidence / "nsys_p1_kernel_summary.csv").read_text()
    trace = (evidence / "nsys_p1_kernel_trace.csv").read_text()
    assert "pencil_kernel" in summary
    assert "alpha_kernel" in summary
    assert trace.count("pencil_kernel") == 9
    assert trace.count("alpha_kernel") == 9
    assert trace.count(',"<unnamed>::cfl_kernel(') == 1
    assert trace.count(",<unnamed>::finish_cfl_kernel(") == 1
    assert "ERR_NVGPUCTRPERM" in (
        evidence / "ncu_unprivileged_p1_n128_s1.log"
    ).read_text()
    privileged_log = (evidence / "ncu_privileged_p1_n128_s1.log").read_text()
    assert "ERR_NVGPUCTRPERM" not in privileged_log
    assert privileged_log.count('Profiling "pencil_kernel"') == 9
    assert (evidence / "ncu_privileged_p1_n128_s1.ncu-rep").stat().st_size > 0
    counters = json.loads(
        (evidence / "ncu_privileged_p1_summary.json").read_text()
    )
    assert counters["schema"] == "gradflow-g5-ncu-basic-summary-v1"
    assert len(counters["pencil_launches"]) == 9
    invariants = counters["launch_invariants"]
    assert invariants["registers_per_thread"] == 128.0
    assert invariants["theoretical_occupancy_percent"] == 33.33
    assert invariants["register_block_limit"] == 2.0
    axes = counters["axis_summary"]
    assert axes["x"]["median_l2_throughput_percent"] < 10.0
    assert axes["y"]["median_l2_throughput_percent"] > 60.0
    assert axes["z"]["median_l2_throughput_percent"] > 90.0
    assert axes["x"]["median_compute_sm_throughput_percent"] > 45.0
    assert axes["y"]["median_compute_sm_throughput_percent"] < 15.0
    compiler = (evidence / "compiler.log").read_text()
    assert "Used 128 registers" in compiler
    assert "0 bytes spill stores, 0 bytes spill loads" in compiler
    print("G5 evidence and failed speed-memory Pareto conclusion verify.")


if __name__ == "__main__":
    main()
