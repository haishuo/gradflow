from __future__ import annotations

import math

from experiments.fd_fv_nonlinear.performance_problem import (
    statistics_record,
    timestep,
)
from experiments.fd_fv_nonlinear.run_phase5c import (
    classification,
    target_selections,
)


def test_phase_5c_timestep_reaches_exact_final_time() -> None:
    for cells in (24, 36, 54, 81, 162):
        dt, steps = timestep(cells)
        assert steps > 0
        assert dt > 0.0
        assert math.isclose(dt * steps, 0.1, rel_tol=0.0, abs_tol=1.0e-15)


def test_phase_5c_statistics_retain_samples_and_quartiles() -> None:
    record = statistics_record([4.0, 1.0, 3.0, 2.0])
    assert record["samples_seconds"] == [4.0, 1.0, 3.0, 2.0]
    assert record["median_seconds"] == 2.5
    assert record["mean_seconds"] == 2.5
    assert record["minimum_seconds"] == 1.0
    assert record["maximum_seconds"] == 4.0
    assert record["q1_seconds"] == 1.75
    assert record["q3_seconds"] == 3.25


def test_phase_5c_classification_uses_five_percent_band() -> None:
    assert classification(1.06) == "fd_faster"
    assert classification(0.94) == "fv_faster"
    assert classification(1.0) == "unresolved_within_5_percent"


def test_target_selection_uses_fastest_qualifying_measured_point() -> None:
    records = []
    for method, seconds in (("fd", 2.0), ("fv", 1.0)):
        for device in ("cpu", "cuda"):
            records.append(
                {
                    "method": method,
                    "device": device,
                    "cells": 81,
                    "mode": "compiled",
                    "eligible": True,
                    "l2_error": 1.0e-8,
                    "aggregate_median_seconds": seconds,
                    "transfer_aggregate_median_seconds": seconds + 0.1,
                    "peak_process_rss_bytes": 100,
                    "peak_cuda_allocated_bytes": 50 if device == "cuda" else None,
                }
            )
    selections = target_selections(records)
    for boundary in selections.values():
        for target in boundary.values():
            assert target["fd"]["status"] == "reached"
            assert target["fv"]["status"] == "reached"
            assert target["classification"] == "fv_faster"
