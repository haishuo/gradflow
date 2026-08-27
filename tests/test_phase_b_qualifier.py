from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.euler_boundary_shock.qualify_phase_b import (
    PILOT_ORDERS,
    QUALIFIED_EULER_WENO_ORDERS,
    REPRESENTATIVE_ORDERS,
    error_metrics,
    evaluate_local_decision,
    initial_state,
    physical_minima,
)
from experiments.euler_boundary_shock.verify_phase_b import (
    DEFAULT_RECORD as PHASE_B_RECORD,
    verify as verify_phase_b,
)


@pytest.mark.parametrize("problem", ("sod", "shu_osher"))
def test_phase_b_initial_states_are_explicit_float64(problem: str) -> None:
    x, conserved, dx = initial_state(problem, 200)
    density, pressure, finite = physical_minima(conserved)
    assert x.dtype is torch.float64
    assert conserved.dtype is torch.float64
    assert x.shape == (200,)
    assert conserved.shape == (3, 200)
    assert dx > 0.0
    assert density > 0.0
    assert pressure > 0.0
    assert finite


def test_phase_b_error_metrics_use_every_primitive_sample() -> None:
    expected = np.zeros((3, 4))
    actual = np.array(
        [[1.0, -1.0, 1.0, -1.0], [0.0, 2.0, 0.0, -2.0], [3.0] * 4]
    )
    metrics = error_metrics(actual, expected)
    assert metrics["l1"] == {"density": 1.0, "velocity": 1.0, "pressure": 3.0}
    assert metrics["linf"] == {
        "density": 1.0,
        "velocity": 2.0,
        "pressure": 3.0,
    }


def passing_local_evidence() -> tuple[dict, dict, dict, dict, dict]:
    local = {
        "uniform": {},
        "periodic_overlap": {},
        "smooth_convergence": {},
        "conservation": {},
    }
    for order in QUALIFIED_EULER_WENO_ORDERS:
        key = str(order)
        local["uniform"][key] = {
            "torch.float32_periodic": 0.0,
            "torch.float64_periodic": 0.0,
        }
        local["periodic_overlap"][key] = 0.0
        local["smooth_convergence"][key] = {
            "l2_errors": [1.0e-4, 1.0e-6, 1.0e-12, 2.0e-12],
            "observable_rates": [float(order)],
        }
        local["conservation"][key] = {"periodic": 0.0, "transmissive": 0.0}
    autograd = {
        str(order): {"finite": True, "relative_error": 0.0, "absolute_error": 0.0}
        for order in REPRESENTATIVE_ORDERS
    }
    compiler = {
        str(order): {
            boundary: {
                "graph_count": 1,
                "graph_break_count": 0,
                "maximum_absolute_difference": 0.0,
            }
            for boundary in ("periodic", "transmissive")
        }
        for order in REPRESENTATIVE_ORDERS
    }
    return local, autograd, compiler, {"available": False}, {"passed": True}


def test_phase_b_decision_enforces_monotonic_errors_above_floor() -> None:
    evidence = passing_local_evidence()
    assert evaluate_local_decision(*evidence)
    evidence[0]["smooth_convergence"][str(PILOT_ORDERS[0])] = {
        "l2_errors": [1.0e-4, 2.0e-4, 1.0e-12, 2.0e-12],
        "observable_rates": [float(PILOT_ORDERS[0])],
    }
    assert not evaluate_local_decision(*evidence)


def test_frozen_phase_b_record_verifies() -> None:
    result = verify_phase_b(PHASE_B_RECORD)
    assert result["passed"] is True
    assert len(result["arrays"]) == 6
