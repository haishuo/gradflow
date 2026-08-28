#!/usr/bin/env python3
"""Independently verify the preserved FD/FV Euler Phase-6B record."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from experiments.fd_fv_euler.phase6a_oracle import build_projections
from experiments.euler_boundary_shock.sod_exact import sod_solution


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6b_20260828"
RECORD = RESULTS / "qualification.json"
RAW = RESULTS / "raw_arrays.npz"
PROJECTIONS = (
    ROOT / "experiments/fd_fv_euler/results/phase_6a_20260828/projections.npz"
)
EXPECTED_SOURCE_COMMIT = "c237716"
EXPECTED_PROTOCOL_COMMIT = "6662943"
SIZES = (24, 36, 54, 81)
SHOCK_SIZES = (200, 400, 800)
METHODS = ("fd", "fv")
BOUNDARIES = ("periodic", "transmissive")
EPSILON = np.finfo(np.float64).eps


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(actual: Any, expected: Any, tolerance: float = 2.0e-14) -> None:
    assert math.isclose(
        float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance
    ), (actual, expected)


def norms(difference: np.ndarray) -> dict[str, float]:
    absolute = np.abs(difference)
    return {
        "l1": float(np.mean(absolute)),
        "l2": float(np.sqrt(np.mean(difference**2))),
        "linf": float(np.max(absolute)),
    }


def rates(errors: list[float]) -> list[float]:
    return [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors, errors[1:], SIZES, SIZES[1:]
        )
    ]


def primitive(conserved: np.ndarray) -> np.ndarray:
    density = conserved[0]
    velocity = conserved[1] / density
    pressure = 0.4 * (conserved[2] - 0.5 * conserved[1] ** 2 / density)
    return np.stack((density, velocity, pressure))


def primitive_errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual - expected)
    names = ("density", "velocity", "pressure")
    return {
        "l1": {name: float(np.mean(difference[i])) for i, name in enumerate(names)},
        "l2": {
            name: float(np.sqrt(np.mean(difference[i] ** 2)))
            for i, name in enumerate(names)
        },
        "linf": {name: float(np.max(difference[i])) for i, name in enumerate(names)},
    }


def conserved_errors(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    difference = np.abs(actual - expected)
    names = ("density", "momentum", "energy")
    return {
        "l1": {name: float(np.mean(difference[i])) for i, name in enumerate(names)},
        "l2": {
            name: float(np.sqrt(np.mean(difference[i] ** 2)))
            for i, name in enumerate(names)
        },
        "linf": {name: float(np.max(difference[i])) for i, name in enumerate(names)},
    }


def compare_metric_tree(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            compare_metric_tree(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            compare_metric_tree(left, right)
    elif isinstance(expected, bool) or expected is None:
        assert actual == expected
    elif isinstance(expected, (int, float)):
        assert_close(actual, expected)
    else:
        assert actual == expected


def verify_manifest() -> tuple[dict[str, Any], np.lib.npyio.NpzFile]:
    lines = (RESULTS / "SHA256SUMS").read_text().splitlines()
    expected = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        expected[name] = digest
    assert expected == {
        "qualification.json": sha256(RECORD),
        "raw_arrays.npz": sha256(RAW),
    }
    payload = json.loads(RECORD.read_text())
    assert payload["phase"] == "fd_fv_euler_phase_6b"
    assert payload["source_commit"].startswith(EXPECTED_SOURCE_COMMIT)
    assert payload["protocol_commit"] == EXPECTED_PROTOCOL_COMMIT
    assert payload["source_dirty"] is False
    assert payload["performance_measurements_collected"] is False
    assert payload["phase_6c_begun"] is False
    assert payload["dveb_modified"] is False
    assert payload["publication_claim"] is False
    for name, digest in payload["source_hashes"].items():
        assert sha256(ROOT / name) == digest
    for name, digest in payload["authority_hashes"].items():
        assert sha256(ROOT / name) == digest
    return payload, np.load(RAW)


def verify_predecessors(payload: dict[str, Any]) -> bool:
    cases = {
        "phase_6a": ROOT / "experiments/fd_fv_euler/verify_phase6a.py",
        "fd_phase_b": ROOT / "experiments/euler_boundary_shock/verify_phase_b.py",
        "deferred_cuda": ROOT / "experiments/deferred_cuda_gates/verify.py",
    }
    for name, script in cases.items():
        result = subprocess.run(
            (sys.executable, str(script)), cwd=ROOT, check=False
        )
        assert result.returncode == 0
        assert payload["predecessors"][name]["passed"] is True
    return True


def verify_projection_identity(payload: dict[str, Any]) -> bool:
    generated, _ = build_projections()
    with np.load(PROJECTIONS) as frozen:
        assert set(generated) == set(frozen.files)
        assert all(np.array_equal(value, frozen[key]) for key, value in generated.items())
    assert payload["projection_identity"]["same_keys"] is True
    assert all(
        case["array_equal"] and case["passed"]
        for case in payload["projection_identity"]["cases"].values()
    )
    return True


def verify_uniform(payload: dict[str, Any], raw: np.lib.npyio.NpzFile) -> bool:
    for method in METHODS:
        for boundary in BOUNDARIES:
            key = f"{method}_{boundary}"
            maximum = float(np.max(np.abs(raw[f"uniform_{key}_rhs"])))
            record = payload["uniform_states"]["cases"][key]
            assert_close(maximum, record["maximum_absolute_rhs"])
            assert record["passed"] == (maximum <= 2.0e-12)
    return all(x["passed"] for x in payload["uniform_states"]["cases"].values())


def verify_spatial(payload: dict[str, Any], raw: np.lib.npyio.NpzFile) -> bool:
    decisions = []
    for method in METHODS:
        record = payload["smooth_spatial_convergence"]["methods"][method]
        errors = {name: [] for name in ("l1", "l2", "linf")}
        for cells, stored in zip(SIZES, record["records"]):
            computed = norms(
                raw[f"spatial_{method}_n{cells}_actual"]
                - raw[f"spatial_{method}_n{cells}_expected"]
            )
            for name, value in computed.items():
                assert_close(value, stored[name])
                errors[name].append(value)
        computed_rates = {name: rates(values) for name, values in errors.items()}
        compare_metric_tree(computed_rates, record["rates"])
        observable = [
            rate
            for rate, coarse, fine in zip(
                computed_rates["l2"], errors["l2"], errors["l2"][1:]
            )
            if coarse > 1.0e-11 and fine > 1.0e-11
        ]
        compare_metric_tree(observable, record["observable_l2_rates"])
        decreasing = all(
            fine < coarse
            for values in errors.values()
            for coarse, fine in zip(values, values[1:])
        )
        passed = decreasing and bool(observable) and max(observable) >= 4.0
        assert record["passed"] == passed
        decisions.append(passed)
    return all(decisions)


def verify_solves(payload: dict[str, Any], raw: np.lib.npyio.NpzFile) -> bool:
    decisions = []
    for method in METHODS:
        record = payload["smooth_complete_solve_convergence"]["methods"][method]
        errors = {name: [] for name in ("l1", "l2", "linf")}
        conserved = []
        for cells, stored in zip(SIZES, record["records"]):
            initial = raw[f"solve_{method}_n{cells}_initial"]
            actual = raw[f"solve_{method}_n{cells}_actual"]
            expected = raw[f"solve_{method}_n{cells}_expected"]
            computed = norms(actual - expected)
            for name, value in computed.items():
                assert_close(value, stored[name])
                errors[name].append(value)
            dx = 1.0 / cells
            drift = np.abs(dx * np.sum(actual - initial, axis=-1))
            single = 64.0 * EPSILON * dx * np.sum(np.abs(initial), axis=-1) + 2.0e-15
            accumulated = stored["steps"] * (single - 2.0e-15) + 2.0e-15
            np.testing.assert_allclose(drift, stored["conservation_drift"], atol=2e-14)
            np.testing.assert_allclose(single, stored["single_step_roundoff_bound"])
            np.testing.assert_allclose(
                accumulated, stored["accumulated_roundoff_bound"]
            )
            conservation_passed = bool(np.all(drift <= accumulated))
            assert stored["conservation_passed"] == conservation_passed
            conserved.append(conservation_passed)
        computed_rates = {name: rates(values) for name, values in errors.items()}
        compare_metric_tree(computed_rates, record["rates"])
        observable = [
            rate
            for rate, coarse, fine in zip(
                computed_rates["l2"], errors["l2"], errors["l2"][1:]
            )
            if coarse > 1.0e-11 and fine > 1.0e-11
        ]
        decreasing = all(
            fine < coarse
            for name in ("l1", "l2")
            for coarse, fine in zip(errors[name], errors[name][1:])
        )
        passed = (
            decreasing
            and bool(observable)
            and max(observable) >= 2.5
            and all(x["completed"] for x in record["records"])
            and all(conserved)
        )
        assert record["passed"] == passed
        decisions.append(passed)
    return all(decisions)


def verify_conservation(payload: dict[str, Any], raw: np.lib.npyio.NpzFile) -> bool:
    dx = 1.0 / 43.0
    for method in METHODS:
        for boundary in BOUNDARIES:
            key = f"{method}_{boundary}"
            rhs = raw[f"conservation_{key}_rhs"]
            fluxes = raw[f"conservation_{key}_fluxes"]
            residual = np.abs(dx * np.sum(rhs, axis=-1) + fluxes[:, 1] - fluxes[:, 0])
            scale = EPSILON * np.maximum(
                dx * np.sum(np.abs(rhs), axis=-1)
                + np.abs(fluxes[:, 0])
                + np.abs(fluxes[:, 1]),
                1.0,
            )
            ratio = residual / scale
            record = payload["conservation"]["cases"][key]
            np.testing.assert_allclose(residual, record["residual"], atol=2e-14)
            np.testing.assert_allclose(scale, record["roundoff_scale"])
            np.testing.assert_allclose(ratio, record["roundoff_scaled_ratio"])
            assert record["passed"] == (float(np.max(ratio)) <= 64.0)
    return all(x["passed"] for x in payload["conservation"]["cases"].values())


def sod_locations(values: np.ndarray, cells: int) -> dict[str, float]:
    solution = sod_solution()
    exact = {
        "contact": 0.5 + 0.2 * solution.star_velocity,
        "shock": 0.5 + 0.2 * solution.right_head_speed,
    }
    jumps = np.abs(np.diff(values[0]))
    interfaces = np.arange(1, cells, dtype=np.float64) / cells
    result = {}
    for name, location in exact.items():
        candidates = np.flatnonzero(np.abs(interfaces - location) <= 0.05)
        selected = candidates[np.argmax(jumps[candidates])]
        result[name] = float(abs(interfaces[selected] - location) * cells)
    return result


def shu_metrics(actual: np.ndarray, expected: np.ndarray, cells: int) -> dict[str, float]:
    x = -5.0 + (np.arange(cells, dtype=np.float64) + 0.5) * (10.0 / cells)
    window = (x >= -3.0) & (x <= 3.0)
    left = actual[0, window]
    right = expected[0, window]
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    correlation = np.dot(left_centered, right_centered) / (
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    tv_ratio = np.sum(np.abs(np.diff(left))) / np.sum(np.abs(np.diff(right)))
    return {"density_correlation": float(correlation), "density_total_variation_ratio": float(tv_ratio)}


def verify_shocks(payload: dict[str, Any], raw: np.lib.npyio.NpzFile) -> bool:
    thresholds = payload["shock_thresholds"]
    with np.load(PROJECTIONS) as expected_arrays:
        for problem in ("sod", "shu_osher"):
            problem_record = payload["shock_study"]["problems"][problem]
            for cells, record in zip(SHOCK_SIZES, problem_record["records"]):
                conserved = raw[f"shock_{problem}_n{cells}_conserved"]
                actual_primitive = raw[f"shock_{problem}_n{cells}_primitive"]
                np.testing.assert_allclose(primitive(conserved), actual_primitive, atol=2e-14)
                prefix = "sod" if problem == "sod" else "shu"
                expected_conserved = expected_arrays[
                    f"{prefix}_n{cells}_fv_"
                    + ("conserved" if problem == "sod" else "reference_conserved")
                ]
                expected_primitive = expected_arrays[
                    f"{prefix}_n{cells}_fv_"
                    + ("primitive" if problem == "sod" else "reference_primitive")
                ]
                compare_metric_tree(
                    primitive_errors(actual_primitive, expected_primitive),
                    record["primitive_errors"],
                )
                compare_metric_tree(
                    conserved_errors(conserved, expected_conserved),
                    record["conserved_errors"],
                )
            records = problem_record["records"]
            positive = all(
                item["minimum_density"] > 0.0 and item["minimum_pressure"] > 0.0
                for item in records
            )
            if problem == "sod":
                errors = {
                    name: [item["primitive_errors"]["l1"][name] for item in records]
                    for name in ("density", "velocity", "pressure")
                }
                limits = thresholds["sod"]
                locations = sod_locations(raw["shock_sod_n800_primitive"], 800)
                for name, value in locations.items():
                    assert_close(value, records[-1]["wave_locations"][name]["error_cells"])
                gates = {
                    "completed": all(item["completed"] for item in records),
                    "positive_stages": positive,
                    "monotonic_refinement": all(
                        fine < coarse
                        for values in errors.values()
                        for coarse, fine in zip(values, values[1:])
                    ),
                    "finest_l1_thresholds": all(
                        errors[name][-1] <= limits["l1_max"][name] for name in errors
                    ),
                    "finest_to_coarsest_ratio": all(
                        values[-1] / values[0]
                        <= limits["finest_to_coarsest_error_ratio_max"]
                        for values in errors.values()
                    ),
                    "wave_locations": all(
                        value <= limits["wave_location_error_cells_max"]
                        for value in locations.values()
                    ),
                }
            else:
                limits = thresholds["shu_osher"]
                errors = [
                    item["primitive_errors"]["l1"]["density"] for item in records
                ]
                expected = expected_arrays["shu_n800_fv_reference_primitive"]
                structure = shu_metrics(raw["shock_shu_osher_n800_primitive"], expected, 800)
                compare_metric_tree(
                    structure,
                    {
                        name: records[-1]["structure"][name]
                        for name in structure
                    },
                )
                gates = {
                    "completed": all(item["completed"] for item in records),
                    "positive_stages": positive,
                    "finest_l1_thresholds": all(
                        records[-1]["primitive_errors"]["l1"][name]
                        <= limits["l1_max_to_n12800"][name]
                        for name in ("density", "velocity", "pressure")
                    ),
                    "finest_to_coarsest_density_ratio": errors[-1] / errors[0]
                    <= limits["finest_to_coarsest_density_error_ratio_max"],
                    "density_correlation": structure["density_correlation"]
                    >= limits["density_correlation_min"],
                    "density_total_variation_ratio": limits[
                        "density_total_variation_ratio_min"
                    ]
                    <= structure["density_total_variation_ratio"]
                    <= limits["density_total_variation_ratio_max"],
                }
            assert gates == problem_record["gate_decisions"]
            assert problem_record["passed"] == all(gates.values())
    return all(x["passed"] for x in payload["shock_study"]["problems"].values())


def verify_record_decisions(payload: dict[str, Any]) -> dict[str, bool]:
    gradients = all(
        item["finite"]
        and (item["relative_error"] <= 2.0e-5 or item["absolute_error"] <= 2.0e-7)
        for item in payload["differentiation"]["cases"].values()
    )
    compiler = all(
        item["passed"]
        for name in ("cpu_cases", "cuda_cases", "cpu_cuda_agreement")
        for item in payload["compiler_and_device"][name].values()
    )
    transfers = (
        not any(payload["transfer_evidence"]["static_forbidden_hits"].values())
        and all(
            item["passed"]
            for item in payload["transfer_evidence"]["profiles"].values()
        )
    )
    assert payload["differentiation"]["passed"] == gradients
    assert payload["compiler_and_device"]["passed"] == compiler
    assert payload["transfer_evidence"]["passed"] == transfers
    return {
        "differentiation": gradients,
        "compiler_and_device": compiler,
        "no_hidden_transfer": transfers,
    }


def main() -> None:
    payload, raw = verify_manifest()
    try:
        gates = {
            "predecessors": verify_predecessors(payload),
            "projection_identity": verify_projection_identity(payload),
            "uniform_states": verify_uniform(payload, raw),
            "smooth_spatial_convergence": verify_spatial(payload, raw),
            "smooth_complete_solve_convergence": verify_solves(payload, raw),
            "conservation": verify_conservation(payload, raw),
            **verify_record_decisions(payload),
            "fv_shocks": verify_shocks(payload, raw),
            "inherited_fd_shock_decision": json.loads(
                (ROOT / payload["predecessors"]["fd_phase_b"]["record"]).read_text()
            )["decision"]
            == "PASS",
        }
        assert gates == payload["gate_decisions"]
        assert payload["failed_gates"] == sorted(
            name for name, passed in gates.items() if not passed
        )
        assert payload["passed"] == all(gates.values())
        assert payload["passed"] is True
    finally:
        raw.close()
    print("FD/FV Euler Phase 6B verification passed")


if __name__ == "__main__":
    main()
