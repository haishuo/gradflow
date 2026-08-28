from __future__ import annotations

import math

from experiments.fd_fv_nonlinear.performance_problem import METHOD_IDS
from experiments.fd_fv_nonlinear.resolve_phase5c import (
    accumulated_bound,
    original_cold_nonconservation_gates,
    original_complete_nonconservation_gates,
)


def test_accumulated_bound_applies_roundoff_per_step_and_slack_once() -> None:
    single = 9.0e-15
    assert accumulated_bound(single, 1) == single
    assert math.isclose(
        accumulated_bound(single, 531),
        531 * 7.0e-15 + 2.0e-15,
        rel_tol=0.0,
        abs_tol=1.0e-27,
    )


def complete_record() -> dict:
    accuracy = {
        "finite": True,
        "l1_error": 1.0e-6,
        "l2_error": 2.0e-6,
        "dtype": "float64",
        "shape": [81],
        "device": "cuda:0",
    }
    return {
        "status": "completed",
        "worker_returncode": 0,
        "kind": "complete",
        "method": "fd",
        "formulation_id": METHOD_IDS["fd"],
        "device": "cuda",
        "cells": 81,
        "accuracy": {
            "eager": dict(accuracy),
            "compiled": dict(accuracy),
            "compiled_eager_maximum_absolute_difference": 2.0e-12,
            "compiled_repeat_maximum_absolute_difference": 0.0,
        },
    }


def test_complete_nonconservation_gates_exclude_only_conservation() -> None:
    record = complete_record()
    assert original_complete_nonconservation_gates(record)
    record["accuracy"]["compiled"]["device"] = "cpu"
    assert not original_complete_nonconservation_gates(record)


def test_cold_nonconservation_gates_require_host_answer_and_identity() -> None:
    record = {
        "status": "completed",
        "worker_returncode": 0,
        "kind": "cold",
        "method": "fv",
        "formulation_id": METHOD_IDS["fv"],
        "finite": True,
        "l1_error": 1.0e-6,
        "l2_error": 2.0e-6,
        "host_visible_answer": True,
    }
    assert original_cold_nonconservation_gates(record)
    record["host_visible_answer"] = False
    assert not original_cold_nonconservation_gates(record)
