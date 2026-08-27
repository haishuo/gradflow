#!/usr/bin/env python3
"""Derive and freeze the independent FD/FV Phase-2 contract records."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from fv_js5_oracle import (
    F,
    FULL_OFFSETS,
    LEFT_OFFSETS,
    LITERAL_CANDIDATES,
    LITERAL_FULL,
    LITERAL_OPTIMAL_WEIGHTS,
    LITERAL_SMOOTHNESS,
    MATCHED_EPSILON,
    NONLINEAR_POWER,
    SMOOTHNESS_SCALE,
    aligned_stencil,
    composite_simpson_average,
    derive_all,
    fourier_cell_average,
    js5_reconstruct,
    periodic_rusanov_rhs,
    polynomial_cell_averages,
    polynomial_value,
    principal_minors,
    quadratic_form,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT / "experiments/fd_fv_contract/results/phase_2_20260827"
)
PROTOCOL = ROOT / "docs/FD_FV_PHASE_2_PROTOCOL.md"
PROTOCOL_COMMIT = "4638a4ab592338ce24e268ef549c7d960e03605d"


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def encode(value: Any) -> Any:
    if isinstance(value, Fraction):
        return fraction_text(value)
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def polynomial_cases() -> dict[str, object]:
    candidate_checks = []
    for degree in range(3):
        coefficients = tuple(F(int(index == degree)) for index in range(degree + 1))
        expected = polynomial_value(coefficients, F(1, 2))
        for candidate, offsets in enumerate(LEFT_OFFSETS):
            averages = polynomial_cell_averages(coefficients, offsets)
            actual = sum(
                coefficient * average
                for coefficient, average in zip(
                    LITERAL_CANDIDATES[candidate], averages
                )
            )
            candidate_checks.append(
                {
                    "degree": degree,
                    "candidate": candidate,
                    "cell_averages": averages,
                    "expected_face_value": expected,
                    "actual_face_value": actual,
                    "passed": actual == expected,
                }
            )

    full_checks = []
    for degree in range(5):
        coefficients = tuple(F(int(index == degree)) for index in range(degree + 1))
        averages = polynomial_cell_averages(coefficients, FULL_OFFSETS)
        expected = polynomial_value(coefficients, F(1, 2))
        actual = sum(
            coefficient * average
            for coefficient, average in zip(LITERAL_FULL, averages)
        )
        full_checks.append(
            {
                "degree": degree,
                "cell_averages": averages,
                "expected_face_value": expected,
                "actual_face_value": actual,
                "passed": actual == expected,
            }
        )
    return {
        "candidate_degree_0_through_2": candidate_checks,
        "full_degree_0_through_4": full_checks,
    }


def smoothness_cases() -> list[dict[str, object]]:
    constant = (F(1), F(1), F(1))
    return [
        {
            "candidate": candidate,
            "symmetric": all(
                matrix[row][column] == matrix[column][row]
                for row in range(3)
                for column in range(3)
            ),
            "principal_minors": principal_minors(matrix),
            "positive_semidefinite": all(
                value >= 0 for value in principal_minors(matrix)
            ),
            "constant_quadratic_form": quadratic_form(matrix, constant),
            "constant_nullspace": quadratic_form(matrix, constant) == 0,
        }
        for candidate, matrix in enumerate(LITERAL_SMOOTHNESS)
    ]


def projection_case() -> dict[str, object]:
    cells = 8
    mode = 2
    wave = 2.0 * math.pi * mode
    sine_amplitude = 1.0
    cosine_amplitude = 0.25

    def function(x: float) -> float:
        return sine_amplitude * math.sin(wave * x) + cosine_amplitude * math.cos(
            wave * x
        )

    analytic = []
    integrated = []
    centers = []
    for index in range(cells):
        left = index / cells
        right = (index + 1) / cells
        analytic.append(
            fourier_cell_average(
                left,
                right,
                sine_amplitude=sine_amplitude,
                cosine_amplitude=cosine_amplitude,
                wavenumber=wave,
            )
        )
        integrated.append(composite_simpson_average(function, left, right))
        centers.append(function((index + 0.5) / cells))
    return {
        "domain": [0.0, 1.0],
        "cells": cells,
        "mode": mode,
        "sine_amplitude": sine_amplitude,
        "cosine_amplitude": cosine_amplitude,
        "analytic_cell_average_hex": [value.hex() for value in analytic],
        "simpson_cell_average_hex": [value.hex() for value in integrated],
        "center_sample_hex": [value.hex() for value in centers],
        "max_analytic_simpson_error_hex": max(
            abs(a - b) for a, b in zip(analytic, integrated)
        ).hex(),
        "max_average_center_difference_hex": max(
            abs(a - b) for a, b in zip(analytic, centers)
        ).hex(),
        "integration_tolerance_hex": float(2.0e-15).hex(),
        "integration_passed": max(
            abs(a - b) for a, b in zip(analytic, integrated)
        )
        <= 2.0e-15,
        "center_sampling_is_distinct": any(a != b for a, b in zip(analytic, centers)),
    }


def semidiscrete_cases() -> dict[str, object]:
    values = (F(2), F(-1), F(3), F(0), F(4), F(1), F(-2), F(5))
    spacing = F(1, len(values))
    constant = tuple(F(7, 3) for _ in values)
    constant_rhs, _, constant_left, constant_right = periodic_rusanov_rhs(
        constant, spacing, lambda value: 2 * value, F(2)
    )

    cases = {}
    for name, speed in (("positive", F(2)), ("negative", F(-3))):
        rhs, fluxes, left, right = periodic_rusanov_rhs(
            values,
            spacing,
            lambda value, speed=speed: speed * value,
            abs(speed),
        )
        expected_flux = tuple(
            speed * value for value in (left if speed > 0 else right)
        )
        cases[name] = {
            "speed": speed,
            "left_face_states": left,
            "right_face_states": right,
            "face_fluxes": fluxes,
            "expected_upwind_fluxes": expected_flux,
            "rhs": rhs,
            "upwind_selection_passed": fluxes == expected_flux,
            "periodic_telescoping_sum": sum(rhs, F(0)) * spacing,
            "periodic_conservation_passed": sum(rhs, F(0)) * spacing == 0,
        }

    reflection_stencils = [
        {
            "candidate": candidate,
            "left_offsets": list(LEFT_OFFSETS[candidate]),
            "right_offsets": [1 - offset for offset in LEFT_OFFSETS[candidate]],
            "left_values_at_face_3": aligned_stencil(values, 3, candidate, "left"),
            "right_values_at_face_3": aligned_stencil(values, 3, candidate, "right"),
        }
        for candidate in range(3)
    ]
    return {
        "deterministic_cell_averages": values,
        "spacing": spacing,
        "constant": {
            "value": F(7, 3),
            "left_face_states": constant_left,
            "right_face_states": constant_right,
            "rhs": constant_rhs,
            "passed": all(value == 0 for value in constant_rhs)
            and all(value == F(7, 3) for value in constant_left + constant_right),
        },
        "linear_advection": cases,
        "reflection_stencils": reflection_stencils,
        "representative_faces": {
            "left_face_3": js5_reconstruct(values, 3, bias="left"),
            "right_face_3": js5_reconstruct(values, 3, bias="right"),
        },
    }


def contract_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "fd_fv_phase_2",
        "freeze_date": "2026-08-27",
        "protocol_commit": PROTOCOL_COMMIT,
        "formulation": {
            "id": "fv_dimensional_js5_global_lf_periodic_v1",
            "taxonomy": "one-dimensional dimension-by-dimension finite-volume WENO-JS",
            "equation_class": "scalar conservation law u_t + f(u)_x = 0",
            "persistent_state": "physical cell average",
            "order": 5,
            "boundary": "unique periodic cells without duplicated endpoint",
        },
        "grid": {
            "domain": "[a,b)",
            "spacing": "(b-a)/N",
            "face_i": "a+i*dx",
            "cell_i": "[a+i*dx,a+(i+1)*dx]",
            "center_i": "a+(i+1/2)*dx",
            "state_i": "(1/dx)*integral(cell_i,u(x) dx)",
            "face_storage": "index i is the right face of cell i",
        },
        "reconstruction": {
            "left_candidate_offsets": LEFT_OFFSETS,
            "right_offset_map": "j -> 1-j",
            "candidate_coefficients": LITERAL_CANDIDATES,
            "optimal_weights": LITERAL_OPTIMAL_WEIGHTS,
            "full_offsets": FULL_OFFSETS,
            "full_coefficients": LITERAL_FULL,
            "standard_smoothness_matrices": LITERAL_SMOOTHNESS,
            "runtime_smoothness_scale": SMOOTHNESS_SCALE,
            "epsilon": MATCHED_EPSILON,
            "nonlinear_power": NONLINEAR_POWER,
            "weight_policy": "d_k/(epsilon+12*beta_k)^2, normalized",
        },
        "numerical_flux": {
            "id": "global_lax_friedrichs_rusanov",
            "formula": "0.5*(f(uL)+f(uR)-alpha*(uR-uL))",
            "alpha_contract": "global alpha >= max abs(f'(u)) for one RHS",
            "linear_advection": "f(u)=c*u and alpha=abs(c)",
        },
        "semidiscrete_operator": {
            "formula": "rhs[i]=-(face_flux[i]-face_flux[i-1])/dx",
            "conservation": "dx*sum_i(rhs[i])=0 for periodic faces",
        },
        "future_time_integrator": {
            "id": "SSP-RK3",
            "stage_1": "u1=u+dt*L(u)",
            "stage_2": "u2=3/4*u+1/4*(u1+dt*L(u1))",
            "stage_3": "u_next=1/3*u+2/3*(u2+dt*L(u2))",
            "final_step": "shorten only to reach exact final physical time",
            "executed_in_phase_2": False,
        },
        "projection": {
            "initial_and_exact_state": "cell average of the continuous field",
            "analytic_preference": True,
            "fallback": "independently converged quadrature with negligible error",
            "center_sampling_permitted_as_cell_average": False,
        },
        "precision_gate": {
            "reference": "binary64 for future implementation qualification",
            "oracle": "exact Fraction arithmetic plus hexadecimal binary64 cases",
            "performance_precision_in_phase_2": "none",
        },
        "explicit_exclusions": [
            "canonical FV implementation",
            "Euler or characteristic reconstruction",
            "nonperiodic boundaries",
            "multidimensional face quadrature",
            "genuinely multidimensional FV",
            "alternative finite difference",
            "WENO-Z or positivity limiting",
            "adaptive or conventional best-practical epsilon",
            "arbitrary order",
            "GPU, compilation, optimization, or timing",
        ],
    }


def oracle_record() -> dict[str, object]:
    derived = derive_all()
    literals = {
        "candidate_coefficients": LITERAL_CANDIDATES,
        "optimal_weights": LITERAL_OPTIMAL_WEIGHTS,
        "full_coefficients": LITERAL_FULL,
        "smoothness_matrices": LITERAL_SMOOTHNESS,
    }
    return {
        "schema_version": 1,
        "phase": "fd_fv_phase_2",
        "freeze_date": "2026-08-27",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_hashes": {
            "docs/FD_FV_PHASE_2_PROTOCOL.md": sha256(PROTOCOL),
            "experiments/fd_fv_contract/fv_js5_oracle.py": sha256(
                Path(__file__).with_name("fv_js5_oracle.py")
            ),
            "experiments/fd_fv_contract/derive_phase_2.py": sha256(Path(__file__)),
        },
        "exact_derivation": {
            "derived": derived,
            "literal": literals,
            "matches_literal": derived == literals,
            "optimal_weights_positive": all(
                weight > 0 for weight in LITERAL_OPTIMAL_WEIGHTS
            ),
            "optimal_weights_sum": sum(LITERAL_OPTIMAL_WEIGHTS, F(0)),
        },
        "polynomial_reproduction": polynomial_cases(),
        "smoothness": smoothness_cases(),
        "fourier_projection": projection_case(),
        "semidiscrete": semidiscrete_cases(),
    }


def write_json(path: Path, record: dict[str, object]) -> None:
    path.write_text(json.dumps(encode(record), indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    contract_path = output / "contract.json"
    oracle_path = output / "oracle_cases.json"
    sums_path = output / "SHA256SUMS"
    existing = [
        path for path in (contract_path, oracle_path, sums_path) if path.exists()
    ]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite frozen Phase-2 files: {names}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(contract_path, contract_record())
    write_json(oracle_path, oracle_record())
    sums_path.write_text(
        f"{sha256(contract_path)}  contract.json\n"
        f"{sha256(oracle_path)}  oracle_cases.json\n"
    )
    print(f"wrote independent Phase-2 records to {output}")


if __name__ == "__main__":
    main()
