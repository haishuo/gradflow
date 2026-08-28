from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.fd_fv_nonlinear.burgers_oracle import (
    BASE,
    FINAL_TIME,
    MINIMUM_CHARACTERISTIC_JACOBIAN,
    SHOCK_TIME,
    characteristic_foot,
    characteristic_map,
    exact_cell_average,
    exact_cell_average_by_quadrature,
    exact_point,
    projected_state,
)
ROOT = Path(__file__).resolve().parents[1]


def test_characteristic_inverse_is_unique_and_accurate_before_shock() -> None:
    assert FINAL_TIME < SHOCK_TIME
    assert MINIMUM_CHARACTERISTIC_JACOBIAN > 0.0
    for index in range(41):
        x = index / 40
        foot = characteristic_foot(x, FINAL_TIME)
        assert characteristic_map(foot, FINAL_TIME) == pytest.approx(x, abs=3e-16)


def test_primitive_cell_average_agrees_with_independent_quadrature() -> None:
    for index in range(8):
        left = index / 8
        right = (index + 1) / 8
        primitive = exact_cell_average(left, right, FINAL_TIME)
        quadrature = exact_cell_average_by_quadrature(
            left, right, FINAL_TIME, panels=4096
        )
        assert primitive == pytest.approx(quadrature, abs=2e-12)


def test_exact_fv_projection_conserves_periodic_mean() -> None:
    for cells in (8, 17, 24):
        for time in (0.0, FINAL_TIME):
            averages = projected_state("fv", cells, time)
            assert math.fsum(averages) / cells == pytest.approx(BASE, abs=2e-14)


def test_fv_average_is_not_silently_replaced_by_center_sample() -> None:
    cells = 8
    averages = projected_state("fv", cells, FINAL_TIME)
    center_samples = tuple(
        exact_point((index + 0.5) / cells, FINAL_TIME)
        for index in range(cells)
    )
    assert max(abs(a - c) for a, c in zip(averages, center_samples)) > 1e-4


def test_phase_5a_generator_refuses_to_overwrite(tmp_path: Path) -> None:
    script = ROOT / "experiments/fd_fv_nonlinear/freeze_phase_5a.py"
    output = tmp_path / "record"
    command = [sys.executable, str(script), "--output-dir", str(output)]
    first = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert first.returncode == 0, first.stderr
    second = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr


def test_frozen_phase_5a_record_verifies() -> None:
    script = ROOT / "experiments/fd_fv_nonlinear/verify_phase_5a.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "FD/FV nonlinear Phase 5A verified" in result.stdout
