#!/usr/bin/env python3
"""Freeze the independent FD/FV Euler Phase-6A contracts and projections."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any

import numpy as np

from experiments.fd_fv_euler.phase6a_oracle import (
    GAMMA,
    SHOCK_SIZES,
    SMOOTH_SIZES,
    SMOOTH_TIMES,
    build_projections,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "docs/FD_FV_PHASE_6A_PROTOCOL.md"
PHASE_A = ROOT / "experiments/euler_boundary_shock/results/phase_a_20260827"
PHASE_B = ROOT / "experiments/euler_boundary_shock/results/phase_b_20260827"
INHERITED = (
    ROOT / "experiments/euler_boundary_shock/sod_exact.py",
    ROOT / "experiments/euler_boundary_shock/fv_reference.py",
    PHASE_A / "manifest.json",
    PHASE_A / "thresholds.json",
    PHASE_A / "sod_exact_t0p2_n8192.npz",
    PHASE_A / "shu_osher_fv_wenoz_hllc_t1p8_n12800.npz",
    PHASE_B / "qualification.json",
)
EXPECTED_INHERITED = {
    "experiments/euler_boundary_shock/sod_exact.py": (
        "0eb78a61f391eb0640564decfde1f9242ea1831516f948a06bd9b2720fa0e758"
    ),
    "experiments/euler_boundary_shock/fv_reference.py": (
        "0d1a7aba657e953d169f72988a967df2497aaffb6f003c678a9c763d1a7ae220"
    ),
    "experiments/euler_boundary_shock/results/phase_a_20260827/manifest.json": (
        "c99f4b9687f818af486f8fb5905e5363ca503cb17747626332fe4952e2e056fe"
    ),
    "experiments/euler_boundary_shock/results/phase_a_20260827/thresholds.json": (
        "7c3d3c057d9b291a197a8d0c14b1cdeee79b272a39522ed351ff84909513486e"
    ),
    "experiments/euler_boundary_shock/results/phase_a_20260827/"
    "sod_exact_t0p2_n8192.npz": (
        "d7aa679fb05021edad4b494ac1ff3f33bfda07a9fd09b9c71c44f554d16b6858"
    ),
    "experiments/euler_boundary_shock/results/phase_a_20260827/"
    "shu_osher_fv_wenoz_hllc_t1p8_n12800.npz": (
        "67d551dd2560c7ddead29b9c805082ef896bc2ff8bcd5a5e2cfc9856f8b02f65"
    ),
    "experiments/euler_boundary_shock/results/phase_b_20260827/"
    "qualification.json": (
        "95d3da968fc063d204e13effc8d6190e027e4550a4fe2063462ccbbf170c6b5d"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, text=True
    ).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def freeze(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6A requires a clean committed source tree")
    inherited_hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in INHERITED
    }
    if inherited_hashes != EXPECTED_INHERITED:
        raise RuntimeError("an inherited Euler authority changed identity")
    arrays, diagnostics = build_projections()
    gates = {
        "smooth_analytic_projection": all(
            item["analytic_quadrature_maximum_absolute_difference"] <= 5.0e-15
            for item in diagnostics["smooth"].values()
        ),
        "smooth_periodic_conservation": all(
            max(
                item["fd_periodic_rhs_sum_maximum_absolute"],
                item["fv_periodic_rhs_sum_maximum_absolute"],
            )
            <= 5.0e-14
            for item in diagnostics["smooth"].values()
        ),
        "sod_quadrature_convergence": all(
            item["quadrature_32_64_maximum_absolute_difference"] <= 5.0e-13
            for item in diagnostics["sod"].values()
        ),
        "sod_integral_balance": all(
            item["integral_maximum_absolute_difference"] <= 5.0e-13
            for item in diagnostics["sod"].values()
        ),
        "sod_exact_average_admissibility": all(
            item["minimum_exact_average_density"] > 0.0
            and item["minimum_exact_average_pressure"] > 0.0
            for item in diagnostics["sod"].values()
        ),
        "shu_conservative_restriction": all(
            item["fine_restricted_integral_maximum_absolute_difference"]
            <= 5.0e-15
            for item in diagnostics["shu_osher"].values()
        ),
    }
    contract = {
        "schema_version": 1,
        "phase": "fd_fv_euler_phase_6a",
        "record_date": date.today().isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": False,
        "protocol_commit": "9d1b567",
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                PROTOCOL,
                Path(__file__).resolve(),
                Path(__file__).with_name("phase6a_oracle.py"),
            )
        },
        "inherited_hashes": inherited_hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "mathematics": {
            "equations": "one_dimensional_ideal_gas_euler",
            "gamma": GAMMA,
            "order": 5,
            "weno_family": "Jiang-Shu",
            "epsilon": 1.0e-6,
            "smoothness_scaling": 12.0,
            "nonlinear_power": 2,
            "roe_projection": "face_frozen",
            "lf_policy": "line_global_characteristic_family_1p1_enlargement",
            "time_integrator": "SSP-RK3",
            "cfl": 0.1,
            "qualification_dtype": "float64",
        },
        "formulations": {
            "fd": {
                "id": "fd_classical_characteristic_js5_global_lf_euler1d_v1",
                "class": "classical_conservative_fd_flux_reconstruction",
                "persistent_state": "point_values_at_cell_centers",
                "reconstruction": "positive_negative_characteristic_split_flux",
            },
            "fv": {
                "id": "fv_dimensional_characteristic_js5_global_matrix_lf_euler1d_v1",
                "class": "dimension_by_dimension_fv",
                "persistent_state": "physical_conservative_cell_averages",
                "reconstruction": "left_right_characteristic_state",
                "numerical_flux": "characteristic_matrix_global_lf",
            },
        },
        "problems": {
            "smooth_entropy_wave": {
                "domain": [0.0, 1.0],
                "boundary": "periodic",
                "sizes": list(SMOOTH_SIZES),
                "times": list(SMOOTH_TIMES),
            },
            "sod": {
                "domain": [0.0, 1.0],
                "boundary": "transmissive",
                "sizes": list(SHOCK_SIZES),
                "final_time": 0.2,
            },
            "shu_osher": {
                "domain": [-5.0, 5.0],
                "boundary": "transmissive",
                "sizes": list(SHOCK_SIZES),
                "final_time": 1.8,
                "reference_cells": 12800,
            },
        },
        "evaluation": {
            "fd": "point oracle at physical point coordinates",
            "fv": "exact or conservatively restricted cell-average oracle",
            "primitive_fv": "primitive conversion of conservative cell average",
            "fd_fv_arrays_directly_compared": False,
        },
        "diagnostics": diagnostics,
        "gate_decisions": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "passed": all(gates.values()),
        "production_fv_euler_implemented": False,
        "performance_measurements_collected": False,
        "dveb_modified": False,
        "publication_claim": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="phase6a_", dir=output.parent
    ) as temporary:
        temporary_path = Path(temporary)
        np.savez_compressed(temporary_path / "projections.npz", **arrays)
        write_json(temporary_path / "contract.json", contract)
        files = sorted(temporary_path.iterdir())
        (temporary_path / "SHA256SUMS").write_text(
            "".join(f"{sha256(path)}  {path.name}\n" for path in files)
        )
        temporary_path.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    freeze(arguments.output.resolve())
    print(f"froze FD/FV Euler Phase 6A at {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
