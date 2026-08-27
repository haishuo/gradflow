from __future__ import annotations

from fractions import Fraction as F
from pathlib import Path
import subprocess
import sys

from experiments.fd_fv_contract.fv_js5_oracle import (
    LEFT_OFFSETS,
    LITERAL_CANDIDATES,
    LITERAL_FULL,
    LITERAL_OPTIMAL_WEIGHTS,
    LITERAL_SMOOTHNESS,
    derive_all,
    periodic_rusanov_rhs,
    polynomial_cell_averages,
    polynomial_value,
)


ROOT = Path(__file__).resolve().parents[1]


def test_independent_exact_derivation_matches_literal_js5_tables() -> None:
    assert derive_all() == {
        "candidate_coefficients": LITERAL_CANDIDATES,
        "optimal_weights": LITERAL_OPTIMAL_WEIGHTS,
        "full_coefficients": LITERAL_FULL,
        "smoothness_matrices": LITERAL_SMOOTHNESS,
    }


def test_cell_average_reconstruction_reproduces_polynomials() -> None:
    for degree in range(3):
        polynomial = tuple(F(int(index == degree)) for index in range(degree + 1))
        expected = polynomial_value(polynomial, F(1, 2))
        for offsets, coefficients in zip(LEFT_OFFSETS, LITERAL_CANDIDATES):
            averages = polynomial_cell_averages(polynomial, offsets)
            actual = sum(c * value for c, value in zip(coefficients, averages))
            assert actual == expected


def test_fraction_oracle_preserves_periodic_conservation_for_both_signs() -> None:
    values = (F(2), F(-1), F(3), F(0), F(4), F(1), F(-2), F(5))
    spacing = F(1, len(values))
    for speed in (F(2), F(-3)):
        rhs, fluxes, left, right = periodic_rusanov_rhs(
            values,
            spacing,
            lambda value, speed=speed: speed * value,
            abs(speed),
        )
        assert spacing * sum(rhs, F(0)) == 0
        upwind = left if speed > 0 else right
        assert fluxes == tuple(speed * value for value in upwind)


def test_phase_2_generator_refuses_to_overwrite(tmp_path: Path) -> None:
    script = ROOT / "experiments/fd_fv_contract/derive_phase_2.py"
    command = [sys.executable, str(script), "--output-dir", str(tmp_path)]
    first = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert first.returncode == 0, first.stderr
    second = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert second.returncode != 0
    assert "refusing to overwrite" in second.stderr


def test_frozen_phase_2_record_verifies() -> None:
    script = ROOT / "experiments/fd_fv_contract/verify_phase_2.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "FD/FV Phase 2 verified" in result.stdout
