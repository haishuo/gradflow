#!/usr/bin/env python3
"""Create the immutable Phase-A Euler shock oracle record."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import Any

import numpy as np

from experiments.euler_boundary_shock.fv_reference import (
    primitive_to_conserved,
    shu_osher_initial,
    sod_initial,
    solve,
)
from experiments.euler_boundary_shock.sod_exact import (
    sample_solution,
    sod_solution,
    validate_sod_oracle,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXPERIMENT / "results" / "phase_a_20260827"
SOD_CELLS = (100, 200, 400, 800, 1600)
SHU_OSHER_CELLS = (800, 1600, 3200, 6400, 12800)
GAMMA = 1.4
REFERENCE_CFL = 0.4
THRESHOLD_MULTIPLIER = 3.0


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual - expected)
    names = ("density", "velocity", "pressure")
    return {
        "l1": {
            name: float(np.mean(difference[index]))
            for index, name in enumerate(names)
        },
        "linf": {
            name: float(np.max(difference[index]))
            for index, name in enumerate(names)
        },
    }


def _total_variation(values: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(values))))


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = math.sqrt(
        float(np.dot(first_centered, first_centered))
        * float(np.dot(second_centered, second_centered))
    )
    return float(np.dot(first_centered, second_centered) / denominator)


def _run_sod_study() -> tuple[dict[str, Any], dict[int, tuple[np.ndarray, np.ndarray]]]:
    exact_solution = sod_solution()
    cases: dict[str, Any] = {}
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cells in SOD_CELLS:
        x, primitive, statistics = solve(
            sod_initial,
            left=0.0,
            right=1.0,
            cells=cells,
            final_time=0.2,
            cfl=REFERENCE_CFL,
            gamma=GAMMA,
        )
        exact = sample_solution(exact_solution, x, time=0.2, interface=0.5)
        arrays[cells] = (x, primitive)
        cases[str(cells)] = {
            "errors": _error_metrics(primitive, exact),
            "statistics": asdict(statistics),
        }
    for name in ("density", "velocity", "pressure"):
        errors = [cases[str(cells)]["errors"]["l1"][name] for cells in SOD_CELLS]
        if any(later >= earlier for earlier, later in zip(errors, errors[1:])):
            raise RuntimeError(f"independent Sod reference did not refine for {name}")
    return cases, arrays


def _run_shu_osher_study() -> tuple[
    dict[str, Any], dict[int, tuple[np.ndarray, np.ndarray]]
]:
    arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    statistics: dict[int, Any] = {}
    for cells in SHU_OSHER_CELLS:
        x, primitive, run_statistics = solve(
            shu_osher_initial,
            left=-5.0,
            right=5.0,
            cells=cells,
            final_time=1.8,
            cfl=REFERENCE_CFL,
            gamma=GAMMA,
        )
        arrays[cells] = (x, primitive)
        statistics[cells] = asdict(run_statistics)

    finest_x, finest = arrays[SHU_OSHER_CELLS[-1]]
    cases: dict[str, Any] = {}
    for cells in SHU_OSHER_CELLS[:-1]:
        x, primitive = arrays[cells]
        expected = np.stack(
            [np.interp(x, finest_x, finest[index]) for index in range(3)]
        )
        window = (x >= -3.0) & (x <= 3.0)
        expected_density = expected[0, window]
        actual_density = primitive[0, window]
        cases[str(cells)] = {
            "errors_to_n12800": _error_metrics(primitive, expected),
            "structure": {
                "window": [-3.0, 3.0],
                "density_correlation": _correlation(
                    actual_density, expected_density
                ),
                "density_total_variation_ratio": (
                    _total_variation(actual_density)
                    / _total_variation(expected_density)
                ),
            },
            "statistics": statistics[cells],
        }
    cases[str(SHU_OSHER_CELLS[-1])] = {
        "reference": True,
        "statistics": statistics[SHU_OSHER_CELLS[-1]],
        "structure": {
            "window": [-3.0, 3.0],
            "density_total_variation": _total_variation(
                finest[0, (finest_x >= -3.0) & (finest_x <= 3.0)]
            ),
        },
    }
    resolution_error = cases["6400"]["errors_to_n12800"]["l1"]["density"]
    if resolution_error > 2.5e-3:
        raise RuntimeError(
            "N=6400/N=12800 Shu--Osher density disagreement exceeds 2.5e-3"
        )
    for cell_count in SHU_OSHER_CELLS:
        if statistics[cell_count]["reconstruction_fallbacks"] != 0:
            raise RuntimeError("Shu--Osher reference required a positivity fallback")
    return cases, arrays


def _derive_thresholds(
    sod_cases: dict[str, Any], shu_cases: dict[str, Any]
) -> dict[str, Any]:
    names = ("density", "velocity", "pressure")
    sod_baseline = sod_cases["800"]["errors"]
    sod_l1 = {
        name: max(1.0e-12, THRESHOLD_MULTIPLIER * sod_baseline["l1"][name])
        for name in names
    }
    shu_baseline = shu_cases["800"]
    shu_l1 = {
        name: max(
            1.0e-12,
            THRESHOLD_MULTIPLIER
            * shu_baseline["errors_to_n12800"]["l1"][name],
        )
        for name in names
    }
    baseline_correlation = shu_baseline["structure"]["density_correlation"]
    baseline_tv_ratio = shu_baseline["structure"][
        "density_total_variation_ratio"
    ]
    return {
        "derivation": {
            "independent_method": "finite-volume WENO-Z/HLLC float64",
            "multiplier": THRESHOLD_MULTIPLIER,
            "target_grids": [200, 400, 800],
            "reference_grid": 12800,
            "selected_without_gradflow_boundary_implementation": True,
        },
        "sod": {
            "final_time": 0.2,
            "finest_grid": 800,
            "l1_max": sod_l1,
            "each_variable_must_decrease_on_200_400_800": True,
            "finest_to_coarsest_error_ratio_max": 0.75,
            "wave_location_error_cells_max": 3.0,
            "minimum_density_strictly_positive": True,
            "minimum_pressure_strictly_positive": True,
        },
        "shu_osher": {
            "final_time": 1.8,
            "finest_grid": 800,
            "l1_max_to_n12800": shu_l1,
            "finest_to_coarsest_density_error_ratio_max": 0.8,
            "structure_window": [-3.0, 3.0],
            "density_correlation_min": max(0.0, baseline_correlation - 0.05),
            "density_total_variation_ratio_min": max(
                0.5, baseline_tv_ratio - 0.2
            ),
            "density_total_variation_ratio_max": min(
                1.5, baseline_tv_ratio + 0.2
            ),
            "minimum_density_strictly_positive": True,
            "minimum_pressure_strictly_positive": True,
        },
        "conservation": {
            "roundoff_scaled_rhs_boundary_flux_residual_max": 64.0
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def prepare(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if _git("status", "--porcelain"):
        raise RuntimeError("Phase-A records require a clean source worktree")

    source_commit = _git("rev-parse", "HEAD")
    source_files = (
        EXPERIMENT / "sod_exact.py",
        EXPERIMENT / "fv_reference.py",
        EXPERIMENT / "prepare_phase_a.py",
        ROOT / "docs" / "EULER_BOUNDARY_SHOCK_PROTOCOL.md",
    )
    source_hashes = {
        str(path.relative_to(ROOT)): _sha256(path) for path in source_files
    }

    sod_validation = validate_sod_oracle()
    if not sod_validation["passed"]:
        raise RuntimeError("exact Sod oracle validation failed")
    sod_cases, sod_arrays = _run_sod_study()
    shu_cases, shu_arrays = _run_shu_osher_study()
    thresholds = _derive_thresholds(sod_cases, shu_cases)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phase_a_", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        exact_x = (np.arange(8192, dtype=np.float64) + 0.5) / 8192.0
        exact_primitive = sample_solution(
            sod_solution(), exact_x, time=0.2, interface=0.5
        )
        np.savez_compressed(
            temporary_path / "sod_exact_t0p2_n8192.npz",
            x=exact_x,
            primitive=exact_primitive,
            conserved=primitive_to_conserved(exact_primitive, GAMMA),
        )
        sod_x, sod_primitive = sod_arrays[1600]
        np.savez_compressed(
            temporary_path / "sod_fv_wenoz_hllc_t0p2_n1600.npz",
            x=sod_x,
            primitive=sod_primitive,
            conserved=primitive_to_conserved(sod_primitive, GAMMA),
        )
        shu_x, shu_primitive = shu_arrays[12800]
        np.savez_compressed(
            temporary_path / "shu_osher_fv_wenoz_hllc_t1p8_n12800.npz",
            x=shu_x,
            primitive=shu_primitive,
            conserved=primitive_to_conserved(shu_primitive, GAMMA),
        )
        _write_json(temporary_path / "thresholds.json", thresholds)
        manifest = {
            "schema": "gradflow.euler_boundary_shock.phase_a.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": source_commit,
            "source_worktree_clean": True,
            "source_hashes": source_hashes,
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "mathematics": {
                "gamma": GAMMA,
                "reference_cfl": REFERENCE_CFL,
                "reference_method": (
                    "componentwise primitive finite-volume WENO-Z, HLLC, SSP-RK3"
                ),
                "reference_boundary": "transmissive constant extrapolation",
                "sod": {
                    "domain": [0.0, 1.0],
                    "interface": 0.5,
                    "final_time": 0.2,
                    "left_primitive": [1.0, 0.0, 1.0],
                    "right_primitive": [0.125, 0.0, 0.1],
                },
                "shu_osher": {
                    "domain": [-5.0, 5.0],
                    "interface": -4.0,
                    "final_time": 1.8,
                    "left_primitive": [3.857143, 2.629369, 10.33333],
                    "right_primitive": ["1 + 0.2*sin(5*x)", 0.0, 1.0],
                },
            },
            "sod_exact_validation": sod_validation,
            "sod_resolution_study": sod_cases,
            "shu_osher_resolution_study": shu_cases,
            "threshold_file": "thresholds.json",
            "claim_boundary": {
                "gradflow_boundary_implementation_run": False,
                "performance_measured": False,
                "dveb_modified": False,
                "publication_claim": False,
            },
        }
        _write_json(temporary_path / "manifest.json", manifest)
        artifact_files = sorted(
            path for path in temporary_path.iterdir() if path.name != "SHA256SUMS"
        )
        checksum_lines = [
            f"{_sha256(path)}  {path.name}" for path in artifact_files
        ]
        (temporary_path / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
        temporary_path.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    prepare(arguments.output.resolve())
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
