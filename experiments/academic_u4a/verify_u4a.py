#!/usr/bin/env python3
"""Offline checksum and semantic verification for Academic U4-A."""

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

    audit = json.loads((evidence / "compatibility_audit.json").read_text())
    assert audit["schema"] == "gradflow-academic-u4a-compatibility-audit-v1"
    assert audit["complete"]
    assert not audit["comparative_timing_performed"]
    protocol = audit["protocol"]
    assert protocol["commit"] == "c5cc2ad"
    assert sha256(ROOT / protocol["path"]) == protocol["sha256"]
    for record in audit["local_sources"]:
        assert sha256(ROOT / record["path"]) == record["sha256"]

    opensbli = audit["external_sources"]["opensbli"]
    pyweno = audit["external_sources"]["pyweno"]
    assert opensbli["commit"] == "e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d"
    assert pyweno["commit"] == "cfc12766556d8989b03c1051e2dd32510dc33f6e"
    for source in (opensbli, pyweno):
        assert len(source["tree"]) == 40
        assert source["official_url"].startswith("https://github.com/")
        assert source["files"]
        assert all(len(item["sha256"]) == 64 for item in source["files"])

    cross_check = audit["symbolic_cross_check"]
    script = cross_check["script"]
    record = cross_check["record"]
    assert sha256(ROOT / script["path"]) == script["sha256"]
    assert sha256(evidence / record["path"]) == record["sha256"]
    frozen_cross_check = json.loads((evidence / record["path"]).read_text())
    assert frozen_cross_check["schema"] == (
        "gradflow-academic-u4a-symbolic-crosscheck-v1"
    )
    assert not frozen_cross_check["performance_result"]
    assert tuple(
        item["order"] for item in frozen_cross_check["candidate_checks"]
    ) == ORDERS
    assert all(
        item["exact_match"] for item in frozen_cross_check["candidate_checks"]
    )
    assert frozen_cross_check["smoothness_checks"] == [
        {"order": 5, "exact_match": True},
        {"order": 7, "exact_match": True},
        {"order": 9, "exact_match": True},
    ]
    candidate_checks = cross_check[
        "candidate_offsets_coefficients_and_optimal_weights"
    ]
    assert tuple(item["order"] for item in candidate_checks) == ORDERS
    assert all(item["exact_match"] for item in candidate_checks)
    smoothness = cross_check["smoothness_matrices"]
    assert tuple(item["order"] for item in smoothness) == ORDERS
    assert [item["exact_match"] for item in smoothness] == [
        True,
        True,
        True,
        None,
        None,
        None,
    ]

    classifications = {item["id"]: item for item in audit["classifications"]}
    assert set(classifications) == {"opensbli", "pyweno", "jax_fluids", "hope"}
    assert classifications["opensbli"]["class"] == "matched_operator_candidate"
    assert classifications["opensbli"]["selected_for_u4b_qualification"]
    assert not classifications["opensbli"]["stock_lane_admitted"]
    assert classifications["pyweno"]["class"] == "building_block_only"
    assert classifications["jax_fluids"]["class"] == "application_context_only"
    assert classifications["hope"]["class"] == "application_context_only"
    assert not any(
        item["stock_lane_admitted"] for item in classifications.values()
    )

    decision = audit["decision"]
    assert decision["u4a_closed"]
    assert not decision["u4b_authorized_by_this_record"]
    assert not decision["jax_fluids_timing_now"]
    assert decision["paper_claim"] == (
        "No independent external performance result exists yet."
    )

    results = (ROOT / "docs/ACADEMIC_U4A_RESULTS.md").read_text()
    assert "complete; no external benchmark run" in results
    assert "OpenSBLI" in results and "PyWENO" in results
    assert "JAX-Fluids" in results and "HOPE" in results
    assert "no independent external performance result" in results
    print("Academic U4-A evidence and compatibility decisions verify.")


if __name__ == "__main__":
    main()
