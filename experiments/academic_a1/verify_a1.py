#!/usr/bin/env python3
"""Offline checksum and semantic verification for Academic A1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORDERS = (5, 7, 9, 11, 13, 15)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(evidence: Path) -> None:
    for line in (evidence / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        actual = sha256(evidence / relative)
        assert actual == expected, f"checksum mismatch for {relative}: {actual}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()
    verify_checksums(evidence)

    limits = json.loads((evidence / "numerical_limits.json").read_text())
    assert limits["schema"] == "gradflow-academic-a1-numerical-limits-v1"
    assert limits["complete"]
    assert tuple(limits["qualified_orders"]) == ORDERS
    assert limits["protocol_commit"] == "418e2d4"
    assert limits["claim_boundary"] == {
        "performance_measured": False,
        "default_epsilon_changed": False,
        "canonical_source_changed": False,
        "condition_numbers_are_intrinsic_stability_proofs": False,
        "sampled_roundoff_floor_is_universal": False,
        "scalar_epsilon_results_transfer_to_characteristic_euler": False,
    }
    for relative, expected in limits["source_sha256"].items():
        assert sha256(ROOT / relative) == expected
    assert len(limits["coefficient_diagnostics"]) == 6
    assert len(limits["roundoff_sweeps"]) == 12
    assert len(limits["epsilon_sweeps"]) == 6
    coefficients = {item["order"]: item for item in limits["coefficient_diagnostics"]}
    assert tuple(coefficients) == ORDERS
    assert coefficients[5]["full_moment_condition_2"] == 53.16502466843912
    assert coefficients[15]["full_moment_condition_2"] == 3_249_885_053_166.1367
    assert coefficients[15]["optimal_weight_dynamic_range"] == 2450.0
    assert coefficients[15]["maximum_numerator_bits"] == 145
    assert coefficients[15]["maximum_denominator_bits"] == 148

    roundoff = {
        (item["order"], item["dtype"]): item for item in limits["roundoff_sweeps"]
    }
    assert len(roundoff) == 12
    assert all(item["all_finite"] and item["all_conservative"] for item in roundoff.values())
    assert roundoff[(5, "float32")]["sampled_minimum_n"] == 256
    assert roundoff[(5, "float64")]["first_sampled_roundoff_onset_n"] is None
    assert roundoff[(15, "float32")]["first_sampled_roundoff_onset_n"] == 64
    assert roundoff[(15, "float64")]["first_sampled_roundoff_onset_n"] == 256
    assert all(len(item["samples"]) == 9 for item in roundoff.values())

    epsilon = {item["order"]: item for item in limits["epsilon_sweeps"]}
    assert tuple(epsilon) == ORDERS
    assert [epsilon[order]["material_change_count"] for order in ORDERS] == [
        11,
        8,
        6,
        6,
        3,
        0,
    ]
    assert all(item["all_finite"] and item["all_conservative"] for item in epsilon.values())
    canonical = [
        record
        for item in epsilon.values()
        for record in item["records"]
        if record["epsilon"] == 1.0e-29
    ]
    assert len(canonical) == 36
    assert all(not record["material_change"] for record in canonical)

    consolidation = json.loads((evidence / "consolidation.json").read_text())
    assert consolidation["schema"] == "gradflow-academic-a1-consolidation-v1"
    assert consolidation["complete"]
    for record in consolidation["source_records"].values():
        assert sha256(ROOT / record["path"]) == record["sha256"]
    statuses = {item["id"]: item["status"] for item in consolidation["claims"]}
    assert statuses == {
        "M1": "established",
        "M2": "established",
        "O1": "observed",
        "O2": "observed",
        "O3": "observed",
        "O4": "observed",
        "O5": "observed",
        "O6": "observed",
        "O7": "observed",
        "O8": "observed",
        "I1": "inferred",
        "U1": "untested",
        "U2": "untested",
        "U3": "untested",
        "P1": "prohibited",
        "P2": "prohibited",
        "P3": "prohibited",
    }
    known_sources = set(consolidation["source_records"])
    assert all(
        set(claim["sources"]).issubset(known_sources)
        for claim in consolidation["claims"]
    )
    assert [item["id"] for item in consolidation["prior_art"]] == [
        "opensbli",
        "pyweno",
        "pyclaw_2012",
        "hope",
        "jax_fluids",
        "jax_shock",
    ]
    assert consolidation["first_paper_scope"]["remaining_gates"] == ["A2", "A3", "A4"]
    assert len(consolidation["numerical_summary"]) == 6

    results = (ROOT / "docs/ACADEMIC_A1_RESULTS.md").read_text()
    claims = (ROOT / "docs/ACADEMIC_A1_CLAIM_MATRIX.md").read_text()
    numerical = (ROOT / "docs/ACADEMIC_A1_NUMERICAL_LIMITS.md").read_text()
    prior_art = (ROOT / "docs/ACADEMIC_A1_PRIOR_ART_COMPARISON.md").read_text()
    assert "A1 complete; A2 is now the active academic gate" in results
    assert "P1" in claims and "Prohibited statements" in claims
    assert "3.250e12" in numerical and "No epsilon changed" in numerical
    for name in ("OpenSBLI", "PyWENO", "HOPE", "JAX-Fluids", "JAX-Shock"):
        assert name in prior_art
    print("Academic A1 evidence, source index, and frozen conclusions verify.")


if __name__ == "__main__":
    main()
