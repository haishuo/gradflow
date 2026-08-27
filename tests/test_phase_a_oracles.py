from __future__ import annotations

import numpy as np

from experiments.euler_boundary_shock.fv_reference import (
    conserved_to_primitive,
    finite_volume_rhs,
    primitive_to_conserved,
    sod_initial,
    solve,
)
from experiments.euler_boundary_shock.sod_exact import (
    sample_solution,
    sod_solution,
    validate_sod_oracle,
)


def test_sod_exact_relations() -> None:
    metrics = validate_sod_oracle()
    assert metrics["passed"]


def test_sod_initial_sampling() -> None:
    coordinates = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    sampled = sample_solution(
        sod_solution(), coordinates, time=0.0, interface=0.5
    )
    np.testing.assert_array_equal(sampled[:, 0], np.array([1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(sampled[:, 1], np.array([0.125, 0.0, 0.1]))
    np.testing.assert_array_equal(sampled[:, 2], np.array([0.125, 0.0, 0.1]))


def test_reference_state_round_trip() -> None:
    primitive = np.array(
        [[1.0, 0.7], [0.2, -0.1], [1.2, 0.4]], dtype=np.float64
    )
    recovered = conserved_to_primitive(primitive_to_conserved(primitive))
    np.testing.assert_allclose(recovered, primitive, rtol=2.0e-16, atol=2.0e-16)


def test_reference_uniform_rhs_and_conservation() -> None:
    primitive = np.repeat(
        np.array([[1.2], [0.3], [0.9]], dtype=np.float64), 32, axis=1
    )
    rhs, _, fallbacks, residual = finite_volume_rhs(
        primitive_to_conserved(primitive), 1.0 / 32.0
    )
    np.testing.assert_allclose(rhs, 0.0, rtol=0.0, atol=2.0e-14)
    assert fallbacks == 0
    assert residual <= 2.0


def test_reference_sod_refines_against_exact_solution() -> None:
    exact = sod_solution()
    errors = []
    for cells in (50, 100):
        x, primitive, statistics = solve(
            sod_initial,
            left=0.0,
            right=1.0,
            cells=cells,
            final_time=0.2,
        )
        expected = sample_solution(exact, x, time=0.2, interface=0.5)
        errors.append(float(np.mean(np.abs(primitive[0] - expected[0]))))
        assert statistics.minimum_density > 0.0
        assert statistics.minimum_pressure > 0.0
        assert statistics.maximum_conservation_residual < 64.0
    assert errors[1] < errors[0]
