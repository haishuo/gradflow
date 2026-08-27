#!/usr/bin/env python3
"""Verify the frozen independent FD/FV Phase-2 contract and oracle records."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from derive_phase_2 import contract_record, encode, oracle_record


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "experiments/fd_fv_contract/results/phase_2_20260827"
DATA_FILES = ("contract.json", "oracle_cases.json")
ORACLE_SOURCES = (
    ROOT / "experiments/fd_fv_contract/fv_js5_oracle.py",
    ROOT / "experiments/fd_fv_contract/derive_phase_2.py",
)
FORBIDDEN_IMPORTS = {"torch", "numpy", "gradflow"}


def load(name: str) -> dict:
    with (RESULT_DIR / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", maxsplit=1)[0])
    return roots


def main() -> None:
    contract = load("contract.json")
    oracle = load("oracle_cases.json")
    assert contract == encode(contract_record())
    assert oracle == encode(oracle_record())

    assert contract["schema_version"] == 1
    assert contract["phase"] == "fd_fv_phase_2"
    assert contract["formulation"]["persistent_state"] == "physical cell average"
    assert contract["formulation"]["order"] == 5
    assert contract["future_time_integrator"]["executed_in_phase_2"] is False
    assert contract["projection"]["center_sampling_permitted_as_cell_average"] is False
    assert contract["precision_gate"]["performance_precision_in_phase_2"] == "none"

    exact = oracle["exact_derivation"]
    assert exact["matches_literal"] is True
    assert exact["optimal_weights_positive"] is True
    assert exact["optimal_weights_sum"] == "1/1"
    candidate_checks = oracle["polynomial_reproduction"][
        "candidate_degree_0_through_2"
    ]
    full_checks = oracle["polynomial_reproduction"]["full_degree_0_through_4"]
    assert len(candidate_checks) == 9
    assert all(case["passed"] for case in candidate_checks)
    assert len(full_checks) == 5 and all(case["passed"] for case in full_checks)
    assert all(case["symmetric"] for case in oracle["smoothness"])
    assert all(case["positive_semidefinite"] for case in oracle["smoothness"])
    assert all(case["constant_nullspace"] for case in oracle["smoothness"])

    projection = oracle["fourier_projection"]
    assert projection["integration_passed"] is True
    assert projection["center_sampling_is_distinct"] is True
    semidiscrete = oracle["semidiscrete"]
    assert semidiscrete["constant"]["passed"] is True
    for direction in ("positive", "negative"):
        case = semidiscrete["linear_advection"][direction]
        assert case["upwind_selection_passed"] is True
        assert case["periodic_conservation_passed"] is True
        assert case["periodic_telescoping_sum"] == "0/1"

    for path in ORACLE_SOURCES:
        forbidden = imported_roots(path) & FORBIDDEN_IMPORTS
        assert not forbidden, f"forbidden oracle imports in {path}: {sorted(forbidden)}"

    expected = {}
    for line in (RESULT_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", maxsplit=1)
        expected[name] = value
    assert set(expected) == set(DATA_FILES)
    for name in DATA_FILES:
        assert digest(RESULT_DIR / name) == expected[name]

    print(
        "FD/FV Phase 2 verified: exact FV-JS5 coefficients, 14 polynomial "
        "reproduction checks, 3 smoothness matrices, analytic cell-average "
        "projection, and positive/negative periodic flux invariants."
    )


if __name__ == "__main__":
    main()
