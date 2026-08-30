#!/usr/bin/env python3
"""Run the frozen Academic A1 numerical-limit characterization."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterator

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gradflow import QUALIFIED_ORDERS, WENOJS, generate_weno_js_coefficients  # noqa: E402


ROUND_OFF_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
EPSILONS = (1.0e-40, 1.0e-29, 1.0e-20, 1.0e-12, 1.0e-6)
AMPLITUDES = (1.0, 1.0e-3, 1.0e-6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction_values(value: Any) -> Iterator[Fraction]:
    if isinstance(value, Fraction):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from fraction_values(getattr(value, field.name))
    elif isinstance(value, dict):
        for item in value.values():
            yield from fraction_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from fraction_values(item)


def cell_average_moment(offset: int, degree: int) -> Fraction:
    left = Fraction(2 * offset - 1, 2)
    right = Fraction(2 * offset + 1, 2)
    exponent = degree + 1
    return (right**exponent - left**exponent) / exponent


def moment_matrix(offsets: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        [
            [float(cell_average_moment(offset, degree)) for degree in range(len(offsets))]
            for offset in offsets
        ],
        dtype=np.float64,
    )


def smoothness_restricted_condition(matrix: tuple[tuple[Fraction, ...], ...]) -> float:
    width = len(matrix)
    difference_basis = np.zeros((width, width - 1), dtype=np.float64)
    for column in range(width - 1):
        difference_basis[column, column] = 1.0
        difference_basis[-1, column] = -1.0
    orthonormal, _ = np.linalg.qr(difference_basis, mode="reduced")
    numeric = np.asarray(
        [[float(value) for value in row] for row in matrix], dtype=np.float64
    )
    restricted = orthonormal.T @ numeric @ orthonormal
    eigenvalues = np.linalg.eigvalsh(restricted)
    if eigenvalues[0] <= 0.0:
        raise RuntimeError(f"nonpositive restricted smoothness eigenvalue: {eigenvalues}")
    return float(eigenvalues[-1] / eigenvalues[0])


def coefficient_diagnostics(order: int) -> dict[str, Any]:
    exact = generate_weno_js_coefficients(order)
    fractions = tuple(fraction_values(exact))
    candidate_l1 = [sum(abs(float(value)) for value in row) for row in exact.candidate_coefficients]
    full_l1 = sum(abs(float(value)) for value in exact.full_coefficients)
    weights = [float(value) for value in exact.optimal_weights]
    candidate_conditions = [
        float(np.linalg.cond(moment_matrix(offsets), p=2))
        for offsets in exact.candidate_offsets
    ]
    full_condition = float(np.linalg.cond(moment_matrix(exact.full_offsets), p=2))
    smoothness_conditions = [
        smoothness_restricted_condition(matrix) for matrix in exact.smoothness_matrices
    ]
    return {
        "order": order,
        "substencil_width": exact.substencil_width,
        "minimum_optimal_weight": min(weights),
        "maximum_optimal_weight": max(weights),
        "optimal_weight_dynamic_range": max(weights) / min(weights),
        "candidate_coefficient_l1": candidate_l1,
        "maximum_candidate_coefficient_l1": max(candidate_l1),
        "full_stencil_coefficient_l1": full_l1,
        "candidate_moment_condition_2": candidate_conditions,
        "maximum_candidate_moment_condition_2": max(candidate_conditions),
        "full_moment_condition_2": full_condition,
        "smoothness_restricted_condition_2": smoothness_conditions,
        "maximum_smoothness_restricted_condition_2": max(smoothness_conditions),
        "maximum_numerator_bits": max(abs(value.numerator).bit_length() for value in fractions),
        "maximum_denominator_bits": max(value.denominator.bit_length() for value in fractions),
        "exact_fraction_count": len(fractions),
    }


def smooth_state_and_rhs(n: int, amplitude: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(n, dtype=torch.float64) / n
    state = amplitude * (torch.sin(2.0 * math.pi * x) + 0.15 * torch.cos(6.0 * math.pi * x))
    exact = -amplitude * (
        2.0 * math.pi * torch.cos(2.0 * math.pi * x)
        - 0.9 * math.pi * torch.sin(6.0 * math.pi * x)
    )
    return state, exact


def critical_state_and_rhs(n: int, amplitude: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.arange(n, dtype=torch.float64) / n
    sine = torch.sin(2.0 * math.pi * x)
    cosine = torch.cos(2.0 * math.pi * x)
    state = amplitude * sine.pow(3)
    exact = -amplitude * 6.0 * math.pi * sine.square() * cosine
    return state, exact


def error_norms(actual: torch.Tensor, exact: torch.Tensor, scale: float = 1.0) -> dict[str, float]:
    error = actual.to(torch.float64) - exact
    absolute = torch.abs(error)
    return {
        "l1": float(torch.mean(absolute)) / scale,
        "l2": float(torch.sqrt(torch.mean(error.square()))) / scale,
        "linf": float(torch.amax(absolute)) / scale,
    }


def health(rhs: torch.Tensor) -> dict[str, Any]:
    absolute_sum = torch.sum(torch.abs(rhs), dtype=torch.float64)
    conservation = torch.abs(torch.sum(rhs, dtype=torch.float64))
    bound = 32.0 * torch.finfo(rhs.dtype).eps * absolute_sum
    return {
        "finite": bool(torch.isfinite(rhs).all()),
        "conservation_absolute": float(conservation),
        "conservation_bound": float(bound),
        "conservation_passed": bool(conservation <= bound),
    }


def roundoff_sweep(order: int, dtype: torch.dtype) -> dict[str, Any]:
    scheme = WENOJS(order)
    samples = []
    for n in ROUND_OFF_SIZES:
        source, exact = smooth_state_and_rhs(n)
        state = source.to(dtype)
        rhs = scheme.rhs(state, 1.0 / n, lambda value: value, alpha=1.0)
        samples.append({"n": n, "errors": error_norms(rhs, exact), "health": health(rhs)})
    l2_values = [sample["errors"]["l2"] for sample in samples]
    minimum_index = min(range(len(samples)), key=l2_values.__getitem__)
    onset = None
    for index in range(minimum_index + 1, len(samples)):
        if l2_values[index] > 1.05 * l2_values[index - 1]:
            onset = samples[index]["n"]
            break
    rates = [
        math.log(coarse / fine, 2.0)
        for coarse, fine in zip(l2_values, l2_values[1:])
    ]
    return {
        "order": order,
        "dtype": str(dtype).removeprefix("torch."),
        "samples": samples,
        "successive_l2_rates": rates,
        "sampled_minimum_l2": l2_values[minimum_index],
        "sampled_minimum_n": samples[minimum_index]["n"],
        "first_sampled_roundoff_onset_n": onset,
        "all_finite": all(sample["health"]["finite"] for sample in samples),
        "all_conservative": all(sample["health"]["conservation_passed"] for sample in samples),
    }


def epsilon_sweep(order: int) -> dict[str, Any]:
    records = []
    n = 128
    for family, constructor in (("smooth", smooth_state_and_rhs), ("critical", critical_state_and_rhs)):
        for amplitude in AMPLITUDES:
            source, exact = constructor(n, amplitude)
            baseline_scheme = WENOJS(order, epsilon=EPSILONS[0])
            baseline_rhs = baseline_scheme.rhs(source, 1.0 / n, lambda value: value, alpha=1.0)
            baseline_errors = error_norms(baseline_rhs, exact, scale=amplitude)
            denominator = max(float(torch.amax(torch.abs(baseline_rhs))), amplitude, 1.0e-300)
            for epsilon in EPSILONS:
                scheme = WENOJS(order, epsilon=epsilon)
                rhs = scheme.rhs(source, 1.0 / n, lambda value: value, alpha=1.0)
                errors = error_norms(rhs, exact, scale=amplitude)
                baseline_l2 = baseline_errors["l2"]
                if baseline_l2 == 0.0:
                    error_ratio = 1.0 if errors["l2"] == 0.0 else math.inf
                else:
                    error_ratio = errors["l2"] / baseline_l2
                normalized_rhs_difference = (
                    float(torch.amax(torch.abs(rhs - baseline_rhs))) / denominator
                )
                material = (
                    error_ratio < 0.5
                    or error_ratio > 2.0
                    or normalized_rhs_difference > 1.0e-8
                )
                records.append(
                    {
                        "family": family,
                        "amplitude": amplitude,
                        "epsilon": epsilon,
                        "errors_normalized_by_amplitude": errors,
                        "baseline_error_ratio": error_ratio,
                        "normalized_rhs_difference_from_1e-40": normalized_rhs_difference,
                        "material_change": material,
                        "point_zero_absolute_error_normalized_by_amplitude": (
                            abs(float(rhs[0] - exact[0])) / amplitude
                        ),
                        "health": health(rhs),
                    }
                )
    return {
        "order": order,
        "n": n,
        "baseline_epsilon": EPSILONS[0],
        "records": records,
        "material_change_count": sum(record["material_change"] for record in records),
        "all_finite": all(record["health"]["finite"] for record in records),
        "all_conservative": all(record["health"]["conservation_passed"] for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing existing output: {arguments.output}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    source_paths = (
        ROOT / "docs/ACADEMIC_A1_PROTOCOL.md",
        ROOT / "docs/ACADEMIC_A1_PROTOCOL_CLARIFICATION.md",
        ROOT / "src/gradflow/weno_js.py",
        ROOT / "src/gradflow/weno_js_coefficients.py",
        Path(__file__),
    )
    document = {
        "schema": "gradflow-academic-a1-numerical-limits-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "protocol_commit": "418e2d4",
        "qualified_orders": list(QUALIFIED_ORDERS),
        "coefficient_diagnostics": [
            coefficient_diagnostics(order) for order in QUALIFIED_ORDERS
        ],
        "roundoff_sweeps": [
            roundoff_sweep(order, dtype)
            for order in QUALIFIED_ORDERS
            for dtype in (torch.float32, torch.float64)
        ],
        "epsilon_sweeps": [epsilon_sweep(order) for order in QUALIFIED_ORDERS],
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "claim_boundary": {
            "performance_measured": False,
            "default_epsilon_changed": False,
            "canonical_source_changed": False,
            "condition_numbers_are_intrinsic_stability_proofs": False,
            "sampled_roundoff_floor_is_universal": False,
            "scalar_epsilon_results_transfer_to_characteristic_euler": False,
        },
    }
    document["complete"] = True
    document["completed_utc"] = datetime.now(timezone.utc).isoformat()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "sha256": sha256(arguments.output)}))


if __name__ == "__main__":
    main()
