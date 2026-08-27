#!/usr/bin/env python3
"""Verify the frozen GradFlow FD/FV Phase-1 review records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "experiments/fd_fv_review/results/phase_1_20260827"
DATA_FILES = ("search_log.json", "studies.json", "claim_matrix.json")
VALID_SETS = {"direct", "close", "lineage"}
VALID_STATUSES = {
    "established_non_novel",
    "supported_candidate_contribution",
    "narrowed_candidate_contribution",
    "rejected_candidate_contribution",
    "insufficient_evidence",
}
REQUIRED_STUDY_FIELDS = {
    "id",
    "set",
    "title",
    "authors",
    "year",
    "identity",
    "taxonomy_endpoints",
    "weno_family_orders",
    "equations_dimensions_grid",
    "state_semantics",
    "reconstruction_flux",
    "time_integration",
    "implementation",
    "hardware_precision",
    "comparison_basis",
    "execution_boundary",
    "correctness",
    "method_specific_optimizations",
    "reported_result",
    "limitations",
    "autodiff",
    "memory",
    "claim_relevance",
    "evidence",
}


def load(name: str) -> dict:
    with (RESULT_DIR / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    search_log = load("search_log.json")
    studies_record = load("studies.json")
    claims_record = load("claim_matrix.json")

    for record in (search_log, studies_record, claims_record):
        assert record["schema_version"] == 1
        assert record["review_date"] == "2026-08-27"

    searches = search_log["searches"]
    for provider in ("OpenAlex", "Crossref"):
        found = {item["family"] for item in searches if item["provider"] == provider}
        assert found == {f"S{number}" for number in range(1, 11)}
    for item in searches:
        assert item["status"] in {"completed", "partial", "blocked"}
        assert item["candidates_screened"] >= 0
        assert item["returned"] >= item["candidates_screened"]
        assert item["query"]
        assert item["endpoint"].startswith("https://")
    assert search_log["supplemental_searches"]
    snowballs = [
        item
        for item in search_log["supplemental_searches"]
        if item["provider"] == "Backward/forward citation snowball"
    ]
    assert len(snowballs) == 1
    assert snowballs[0]["passes"] >= 1 and snowballs[0]["stable"] is True

    studies = studies_record["studies"]
    study_ids = [study["id"] for study in studies]
    assert len(study_ids) == len(set(study_ids))
    assert {study["set"] for study in studies} <= VALID_SETS
    assert {
        "shu_2016_survey",
        "zhang_zhang_shu_2011",
        "luo_xuan_xu_2013",
        "balsara_bhoriya_shu_2025",
        "jax_fluids_2",
        "hope_2025",
    } <= set(study_ids)
    for study in studies:
        assert set(study) == REQUIRED_STUDY_FIELDS
        assert study["authors"]
        assert study["taxonomy_endpoints"]
        assert study["evidence"]
        assert study["autodiff"] in {"yes", "no", "unknown", "partial"}
        assert set(study["claim_relevance"]) <= {
            f"F{number}" for number in range(6)
        }
    assert studies_record["excluded_plausible"]

    claims = claims_record["claims"]
    assert {claim["id"] for claim in claims} == {
        f"F{number}" for number in range(6)
    }
    for claim in claims:
        assert claim["status"] in VALID_STATUSES
        assert set(claim["evidence_ids"]) <= set(study_ids)
        assert claim["defensible_statement"]
        assert claim["prohibited_statement"]
        assert claim["required_work"]

    expected: dict[str, str] = {}
    for line in (RESULT_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", maxsplit=1)
        expected[name] = value
    assert set(expected) == set(DATA_FILES)
    for name in DATA_FILES:
        actual = digest(RESULT_DIR / name)
        assert actual == expected[name], f"hash mismatch for {name}"

    print(
        "FD/FV Phase 1 verified: "
        f"{len(searches)} database searches, {len(studies)} included records, "
        f"{len(studies_record['excluded_plausible'])} plausible exclusions, "
        f"{len(claims)} claim decisions."
    )


if __name__ == "__main__":
    main()
