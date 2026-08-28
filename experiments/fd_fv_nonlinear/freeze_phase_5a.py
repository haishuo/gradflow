#!/usr/bin/env python3
"""Create deterministic immutable records for FD/FV nonlinear Phase 5A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fd_fv_nonlinear.burgers_oracle import (
    AMPLITUDE,
    BASE,
    FINAL_TIME,
    LF_ALPHA,
    MINIMUM_CHARACTERISTIC_JACOBIAN,
    PHASE,
    SHOCK_TIME,
    characteristic_foot,
    characteristic_map,
    exact_cell_average_by_quadrature,
    exact_point,
    projected_state,
)


PROTOCOL = ROOT / "docs/FD_FV_PHASE_5A_PROTOCOL.md"
INFRASTRUCTURE = ROOT / "docs/EXECUTION_INFRASTRUCTURE_ADMISSION.md"
CONSTITUTION = ROOT / "docs/FD_FV_EXPERIMENTAL_CONSTITUTION.md"
ORACLE = ROOT / "experiments/fd_fv_nonlinear/burgers_oracle.py"
GENERATOR = Path(__file__).resolve()
DEFAULT_OUTPUT = (
    ROOT / "experiments/fd_fv_nonlinear/results/phase_5a_20260828"
)
ORACLE_TOLERANCE = 2.0e-12
CASE_CELLS = (8, 17)
QUALIFICATION_CELLS = (24, 36, 54, 81)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hexes(values: tuple[float, ...]) -> list[str]:
    return [value.hex() for value in values]


def build_contract() -> dict[str, Any]:
    predecessor_paths = {
        "linear_cuda_replication": (
            ROOT
            / "experiments/fd_fv_bakeoff/results/phase_4r_cuda_20260828"
            / "replication_cuda.json"
        ),
        "deferred_cuda_gates": (
            ROOT
            / "experiments/deferred_cuda_gates/results/qualification_20260828"
            / "qualification.json"
        ),
    }
    return {
        "schema": "gradflow.fd_fv.nonlinear.phase_5a.contract.v1",
        "phase": "fd_fv_nonlinear_5a",
        "freeze_date_utc": "2026-08-28",
        "governing_order": ["correctness", "performance", "convenience"],
        "continuous_problem": {
            "equation": "u_t + (u^2/2)_x = 0",
            "domain": "[0,1)",
            "boundary": "periodic_unique",
            "initial_base_hex": BASE.hex(),
            "initial_amplitude_hex": AMPLITUDE.hex(),
            "initial_phase_hex": PHASE.hex(),
            "final_time_hex": FINAL_TIME.hex(),
            "shock_time_hex": SHOCK_TIME.hex(),
            "minimum_characteristic_jacobian_hex": (
                MINIMUM_CHARACTERISTIC_JACOBIAN.hex()
            ),
        },
        "formulations": {
            "fd": {
                "id": "fd_classical_js5_burgers_global_lf_periodic_v1",
                "persistent_state": "point_values_at_x_i=i/N",
                "operation": "split_physical_flux_reconstruction",
            },
            "fv": {
                "id": "fv_dimensional_js5_burgers_global_lf_periodic_v1",
                "persistent_state": "physical_cell_averages_[i/N,(i+1)/N]",
                "operation": "state_reconstruction_then_rusanov_flux",
            },
        },
        "shared_numerics": {
            "weno": "JS5",
            "epsilon": "1e-29",
            "smoothness_scale": 12,
            "nonlinear_power": 2,
            "flux": "u^2/2",
            "global_lf_alpha_hex": LF_ALPHA.hex(),
            "time_integrator": "SSP-RK3",
            "nominal_dt": "0.2*dx^(5/3)/0.7",
            "qualification_cells": list(QUALIFICATION_CELLS),
        },
        "oracle": {
            "point": "characteristic_bisection",
            "cell_average": "exact_characteristic_conservation_primitive",
            "quadrature_cross_check": "composite_simpson_2048_and_4096_panels",
            "tolerance_hex": ORACLE_TOLERANCE.hex(),
            "forbidden_dependencies": ["torch", "numpy", "gradflow"],
        },
        "infrastructure_contract": (
            "docs/EXECUTION_INFRASTRUCTURE_ADMISSION.md"
        ),
        "performance_measurements_collected": False,
        "production_burgers_implementation_added": False,
        "explicit_exclusions": [
            "timing",
            "optimization",
            "shock_solution",
            "multidimensional_burgers",
            "euler_extension",
            "arbitrary_order_fv",
            "publication_claim",
        ],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PROTOCOL, INFRASTRUCTURE, CONSTITUTION, ORACLE, GENERATOR)
        },
        "predecessor_sha256": {
            name: sha256(path) for name, path in predecessor_paths.items()
        },
    }


def build_cases() -> dict[str, Any]:
    projections: dict[str, Any] = {}
    maximum_residual = 0.0
    maximum_quadrature_difference = 0.0
    maximum_quadrature_refinement = 0.0
    maximum_center_difference = 0.0
    mass_errors: dict[str, str] = {}

    for cells in CASE_CELLS:
        spacing = 1.0 / cells
        cell_record: dict[str, Any] = {}
        for time_name, time in (("initial", 0.0), ("terminal", FINAL_TIME)):
            fd = projected_state("fd", cells, time)
            fv = projected_state("fv", cells, time)
            centers = tuple(
                exact_point((index + 0.5) * spacing, time)
                for index in range(cells)
            )
            residuals = []
            quadrature_differences = []
            refinement_differences = []
            for index in range(cells):
                x = index * spacing
                foot = characteristic_foot(x, time)
                residuals.append(abs(characteristic_map(foot, time) - x))
                left = index * spacing
                right = (index + 1) * spacing
                coarse = exact_cell_average_by_quadrature(
                    left, right, time, panels=2048
                )
                fine = exact_cell_average_by_quadrature(
                    left, right, time, panels=4096
                )
                quadrature_differences.append(abs(fv[index] - fine))
                refinement_differences.append(abs(fine - coarse))
            maximum_residual = max(maximum_residual, *residuals)
            maximum_quadrature_difference = max(
                maximum_quadrature_difference, *quadrature_differences
            )
            maximum_quadrature_refinement = max(
                maximum_quadrature_refinement, *refinement_differences
            )
            maximum_center_difference = max(
                maximum_center_difference,
                *(abs(average - center) for average, center in zip(fv, centers)),
            )
            mass_error = abs(sum(fv) / cells - BASE)
            mass_errors[f"n{cells}_{time_name}"] = mass_error.hex()
            cell_record[time_name] = {
                "time_hex": time.hex(),
                "fd_point_values_hex": _hexes(fd),
                "fv_cell_averages_hex": _hexes(fv),
                "fv_center_samples_hex": _hexes(centers),
                "maximum_characteristic_residual_hex": max(residuals).hex(),
                "maximum_primitive_vs_simpson4096_hex": max(
                    quadrature_differences
                ).hex(),
                "maximum_simpson_refinement_hex": max(
                    refinement_differences
                ).hex(),
                "fv_mass_error_hex": mass_error.hex(),
            }
        projections[str(cells)] = cell_record

    return {
        "schema": "gradflow.fd_fv.nonlinear.phase_5a.oracle_cases.v1",
        "case_cells": list(CASE_CELLS),
        "oracle_tolerance_hex": ORACLE_TOLERANCE.hex(),
        "analytic": {
            "shock_time_hex": SHOCK_TIME.hex(),
            "final_over_shock_time_hex": (FINAL_TIME / SHOCK_TIME).hex(),
            "minimum_characteristic_jacobian_hex": (
                MINIMUM_CHARACTERISTIC_JACOBIAN.hex()
            ),
        },
        "projections": projections,
        "summary": {
            "maximum_characteristic_residual_hex": maximum_residual.hex(),
            "maximum_primitive_vs_simpson4096_hex": (
                maximum_quadrature_difference.hex()
            ),
            "maximum_simpson_refinement_hex": maximum_quadrature_refinement.hex(),
            "maximum_fv_average_vs_center_difference_hex": (
                maximum_center_difference.hex()
            ),
            "mass_errors_hex": mass_errors,
            "all_oracle_checks_passed": (
                maximum_residual <= ORACLE_TOLERANCE
                and maximum_quadrature_difference <= ORACLE_TOLERANCE
                and maximum_quadrature_refinement <= ORACLE_TOLERANCE
                and all(
                    float.fromhex(value) <= ORACLE_TOLERANCE
                    for value in mass_errors.values()
                )
                and maximum_center_difference > 1.0e-5
                and MINIMUM_CHARACTERISTIC_JACOBIAN > 0.0
                and FINAL_TIME < SHOCK_TIME
            ),
        },
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {output}")
    output.mkdir(parents=True)
    contract_path = output / "contract.json"
    cases_path = output / "oracle_cases.json"
    _write_json(contract_path, build_contract())
    _write_json(cases_path, build_cases())
    manifest = "".join(
        f"{sha256(path)}  {path.name}\n" for path in (contract_path, cases_path)
    )
    (output / "SHA256SUMS").write_text(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    freeze(arguments.output_dir.resolve())
    print(f"froze FD/FV nonlinear Phase 5A at {arguments.output_dir}")


if __name__ == "__main__":
    main()
