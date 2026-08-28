#!/usr/bin/env python3
"""Verify the immutable FD/FV Euler Phase-6A record independently."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT,):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from experiments.fd_fv_euler.phase6a_oracle import build_projections


RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6a_20260828"
RECORD = RESULTS / "contract.json"
PROJECTIONS = RESULTS / "projections.npz"
SOURCE_COMMIT = "93b8749fef8611e3a5450329d2509f9c6ef26fb2"
RECORD_SHA256 = "00ffe129fff3bb5e1f1ccea817ba6a5164adc46d489e20709854e93de9121c9d"
PROJECTIONS_SHA256 = (
    "56670eb847c8fe643f96f55275edf20a1c9a01957d248c24fd81ff3ebe6f27a4"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-27)


def verify_manifest() -> None:
    entries = {}
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert relative not in entries
        entries[relative] = expected
        assert sha256(RESULTS / relative) == expected
    assert entries == {
        "contract.json": RECORD_SHA256,
        "projections.npz": PROJECTIONS_SHA256,
    }


def verify_diagnostics(diagnostics: dict) -> dict[str, bool]:
    for item in diagnostics["smooth"].values():
        assert item["analytic_quadrature_maximum_absolute_difference"] >= 0.0
        assert item["fd_periodic_rhs_sum_maximum_absolute"] >= 0.0
        assert item["fv_periodic_rhs_sum_maximum_absolute"] >= 0.0
        assert item["point_cell_average_maximum_absolute_difference"] > 0.0
    for item in diagnostics["sod"].values():
        assert item["quadrature_32_64_maximum_absolute_difference"] >= 0.0
        actual = np.asarray(item["conserved_integral"])
        expected = np.asarray(item["expected_conserved_integral"])
        close(
            item["integral_maximum_absolute_difference"],
            float(np.max(np.abs(actual - expected))),
        )
        assert item["minimum_exact_average_density"] > 0.0
        assert item["minimum_exact_average_pressure"] > 0.0
    for item in diagnostics["shu_osher"].values():
        assert 12800 % item["restriction_factor"] == 0
        assert item["fine_restricted_integral_maximum_absolute_difference"] >= 0.0
        assert item["fd_fv_initial_maximum_absolute_difference"] > 0.0
    return {
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


def main() -> None:
    verify_manifest()
    payload = json.loads(RECORD.read_text())
    assert payload["schema_version"] == 1
    assert payload["phase"] == "fd_fv_euler_phase_6a"
    assert payload["source_commit"] == SOURCE_COMMIT
    assert payload["source_dirty"] is False
    assert payload["protocol_commit"] == "9d1b567"
    assert payload["mathematics"]["gamma"] == 1.4
    assert payload["mathematics"]["order"] == 5
    assert payload["mathematics"]["qualification_dtype"] == "float64"
    assert payload["evaluation"]["fd_fv_arrays_directly_compared"] is False
    assert payload["production_fv_euler_implemented"] is False
    assert payload["performance_measurements_collected"] is False
    assert payload["dveb_modified"] is False
    assert payload["publication_claim"] is False
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected
    for relative, expected in payload["inherited_hashes"].items():
        assert sha256(ROOT / relative) == expected

    rebuilt_arrays, rebuilt_diagnostics = build_projections()
    with np.load(PROJECTIONS) as archive:
        assert set(archive.files) == set(rebuilt_arrays)
        for name, expected in rebuilt_arrays.items():
            assert np.array_equal(archive[name], expected), name
    assert rebuilt_diagnostics == payload["diagnostics"]
    gates = verify_diagnostics(payload["diagnostics"])
    assert gates == payload["gate_decisions"]
    assert payload["failed_gates"] == sorted(
        name for name, passed in gates.items() if not passed
    )
    assert payload["passed"] is all(gates.values())
    assert payload["passed"] is True

    oracle_source = (
        ROOT / "experiments/fd_fv_euler/phase6a_oracle.py"
    ).read_text()
    assert "import torch" not in oracle_source
    assert "from gradflow" not in oracle_source
    print(
        "FD/FV Euler Phase 6A verified: inherited authorities, exact point "
        "and cell-average projections, Sod quadrature/balance, Shu--Osher "
        "restriction, hashes, and no-timing claim all pass."
    )


if __name__ == "__main__":
    main()
