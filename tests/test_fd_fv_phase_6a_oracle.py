from __future__ import annotations

import numpy as np

from experiments.fd_fv_euler.phase6a_oracle import (
    SHOCK_SIZES,
    SMOOTH_SIZES,
    build_projections,
    entropy_average,
    entropy_point,
    shu_reference_projections,
    sod_average,
    sod_integral_expected,
)


def test_entropy_point_and_average_are_distinct_conservative_projections() -> None:
    for cells in SMOOTH_SIZES:
        point, point_rhs = entropy_point(cells, 0.1)
        average, average_rhs = entropy_average(cells, 0.1)
        assert point.shape == average.shape == (3, cells)
        assert point_rhs.shape == average_rhs.shape == (3, cells)
        assert np.max(np.abs(point - average)) > 0.0
        assert np.max(np.abs(np.sum(point_rhs, axis=-1))) < 5.0e-14
        assert np.max(np.abs(np.sum(average_rhs, axis=-1))) < 5.0e-14


def test_exact_sod_cell_averages_converge_and_balance_boundary_flux() -> None:
    expected = sod_integral_expected()
    for cells in SHOCK_SIZES:
        primitive_32, conserved_32 = sod_average(cells, order=32)
        primitive_64, conserved_64 = sod_average(cells, order=64)
        assert np.max(np.abs(conserved_32 - conserved_64)) <= 5.0e-13
        assert np.max(np.abs(np.mean(conserved_64, axis=-1) - expected)) <= 5.0e-13
        assert np.min(primitive_32[[0, 2]]) > 0.0
        assert np.min(primitive_64[[0, 2]]) > 0.0


def test_shu_osher_restriction_is_conservative() -> None:
    for cells in SHOCK_SIZES:
        projection = shu_reference_projections(cells)
        assert np.max(
            np.abs(
                projection["fine_conserved_integral"]
                - projection["restricted_conserved_integral"]
            )
        ) <= 5.0e-15


def test_complete_projection_builder_passes_frozen_diagnostic_bounds() -> None:
    arrays, diagnostics = build_projections()
    assert arrays
    assert all(
        item["analytic_quadrature_maximum_absolute_difference"] <= 5.0e-15
        for item in diagnostics["smooth"].values()
    )
    assert all(
        item["quadrature_32_64_maximum_absolute_difference"] <= 5.0e-13
        for item in diagnostics["sod"].values()
    )
