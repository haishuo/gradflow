from __future__ import annotations

import torch

from experiments.fd_fv_euler.phase6b_problem import evolve, rk_stages
from experiments.fd_fv_euler.phase6c_problem import (
    FINAL_SMOOTH_TIME,
    adaptive_solve,
    fixed_step_function,
    smooth_initial,
    stage_function,
    statistics_record,
)
from experiments.fd_fv_euler.verify_phase6c import main as verify_phase6c


def test_phase6c_stage_helper_matches_qualified_rk_algebra() -> None:
    for method in ("fd", "fv"):
        state = smooth_initial(method, 24)
        dt = state.new_tensor(2.0e-4)
        actual = stage_function(method, 24, "periodic")(state, dt)
        expected = rk_stages(method, state, 1.0 / 24.0, dt, "periodic")
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_phase6c_adaptive_solve_matches_phase6b_qualification_solve() -> None:
    for method in ("fd", "fv"):
        state = smooth_initial(method, 24)
        expected, expected_diagnostics = evolve(
            method, state, 1.0 / 24.0, FINAL_SMOOTH_TIME, "periodic"
        )
        actual, diagnostics = adaptive_solve(
            method,
            state,
            FINAL_SMOOTH_TIME,
            "periodic",
            stage_function(method, 24, "periodic"),
            check_stages=False,
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert diagnostics["steps"] == expected_diagnostics["steps"]
        assert diagnostics["simulated_time"] == FINAL_SMOOTH_TIME


def test_phase6c_fixed_step_is_declared_rk_step() -> None:
    state = smooth_initial("fv", 32)
    expected = stage_function("fv", 32, "periodic")(
        state, state.new_tensor(0.05 / 32.0)
    )[-1]
    actual = fixed_step_function("fv", 32)(state)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_phase6c_statistics_preserve_samples() -> None:
    result = statistics_record([4.0, 1.0, 3.0, 2.0])
    assert result["samples_seconds"] == [4.0, 1.0, 3.0, 2.0]
    assert result["median_seconds"] == 2.5
    assert result["q1_seconds"] == 1.75
    assert result["q3_seconds"] == 3.25


def test_phase6c_committed_record_verifies() -> None:
    verify_phase6c()
