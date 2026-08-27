#!/usr/bin/env python3
"""Verify the frozen GradFlow Phase-C literature-review records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "experiments/literature_review/results/phase_c_20260827"
DATA_FILES = ("search_log.json", "studies.json", "claim_matrix.json")
VALID_CLASSES = {"direct", "close", "lineage"}
VALID_STATUSES = {
    "established_non_novel",
    "supported_candidate_contribution",
    "narrowed_candidate_contribution",
    "rejected_candidate_contribution",
    "insufficient_evidence",
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

    assert search_log["schema_version"] == 1
    assert studies_record["schema_version"] == 1
    assert claims_record["schema_version"] == 1
    assert search_log["review_date"] == "2026-08-27"

    openalex = [
        item for item in search_log["searches"] if item["provider"] == "OpenAlex"
    ]
    assert {item["family"] for item in openalex} == {
        f"S{number}" for number in range(1, 11)
    }
    for item in search_log["searches"]:
        assert item["candidates_screened"] >= 0
        assert item["status"] in {"completed", "partial", "blocked"}

    studies = studies_record["studies"]
    study_ids = [study["id"] for study in studies]
    assert len(study_ids) == len(set(study_ids))
    assert {study["set"] for study in studies} <= VALID_CLASSES
    assert {"opensbli", "pyweno", "jax_fluids", "hope"} <= set(study_ids)
    for study in studies:
        assert study["evidence"]
        assert study["spatial_formulation"]
        assert study["coefficient_policy"]
        assert study["autodiff"] in {"yes", "no", "unknown", "partial"}

    claims = claims_record["claims"]
    assert {claim["id"] for claim in claims} == {
        f"C{number}" for number in range(6)
    }
    for claim in claims:
        assert claim["status"] in VALID_STATUSES
        assert set(claim["evidence_ids"]) <= set(study_ids)
        assert claim["defensible_statement"]
        assert claim["prohibited_statement"]

    expected: dict[str, str] = {}
    for line in (RESULT_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", maxsplit=1)
        expected[name] = value
    assert set(expected) == set(DATA_FILES)
    for name in DATA_FILES:
        actual = digest(RESULT_DIR / name)
        assert actual == expected[name], f"hash mismatch for {name}"

    print(
        "Phase C verified: "
        f"{len(search_log['searches'])} searches, {len(studies)} included records, "
        f"{len(studies_record['excluded_plausible'])} plausible exclusions, "
        f"{len(claims)} claim decisions."
    )


if __name__ == "__main__":
    main()
