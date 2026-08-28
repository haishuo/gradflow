from __future__ import annotations

import torch

from experiments.fd_fv_euler.phase6b_problem import (
    conserved_to_primitive,
    error_metrics,
    rates,
    shock_expected,
    shock_initial,
    shu_structure,
    sod_wave_metrics,
)


def test_phase6b_sod_initial_is_conservative_cell_average() -> None:
    state = shock_initial("sod", 200)
    assert state.shape == (3, 200)
    assert torch.equal(
        state[:, 99], torch.tensor([1.0, 0.0, 2.5], dtype=torch.float64)
    )
    assert torch.equal(
        state[:, 100], torch.tensor([0.125, 0.0, 0.25], dtype=torch.float64)
    )


def test_phase6b_error_and_rate_helpers() -> None:
    expected = torch.ones((3, 4), dtype=torch.float64)
    actual = expected + 0.25
    metrics = error_metrics(actual, expected)
    assert metrics["l1"] == {
        "density": 0.25,
        "velocity": 0.25,
        "pressure": 0.25,
    }
    assert rates([4.0, 1.0, 0.25], (8, 16, 32)) == [2.0, 2.0]


def test_phase6b_stored_shock_targets_are_physical() -> None:
    for problem in ("sod", "shu_osher"):
        conserved, primitive = shock_expected(problem, 200)
        torch.testing.assert_close(conserved_to_primitive(conserved), primitive)
        assert float(torch.min(primitive[0])) > 0.0
        assert float(torch.min(primitive[2])) > 0.0


def test_phase6b_shock_metrics_recover_self_comparison() -> None:
    _, sod = shock_expected("sod", 800)
    locations = sod_wave_metrics(sod, 800)
    assert locations["contact"]["error_cells"] <= 1.0
    assert locations["shock"]["error_cells"] <= 1.0
    _, shu = shock_expected("shu_osher", 800)
    structure = shu_structure(shu, shu, 800)
    assert abs(structure["density_correlation"] - 1.0) < 1.0e-15
    assert abs(structure["density_total_variation_ratio"] - 1.0) < 1.0e-15
