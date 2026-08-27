#!/usr/bin/env python3
"""Verify the completeness, hashes, and classifications of Phase D Tier 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_ORDERS = (5, 7, 9, 11, 13, 15)
EXPECTED_MASKS = tuple(range(64))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_classification(payload: dict[str, Any], record: dict[str, Any]) -> str:
    safety = record["safety"]
    if not safety["passed"]:
        return "failed"
    for name in payload["class_order"]:
        threshold = payload["class_thresholds"][name]
        if (
            record["maximum_linf_normalized"] <= threshold["linf"]
            and record["maximum_rms_normalized"] <= threshold["rms"]
        ):
            return name
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result = args.result_dir / "search.json"
    sums = args.result_dir / "SHA256SUMS"
    expected_hash, expected_name = sums.read_text().strip().split()
    if expected_name != result.name or sha256(result) != expected_hash:
        raise SystemExit("result checksum mismatch")

    payload = json.loads(result.read_text())
    if payload["schema_version"] != 1 or not payload["complete_frozen_matrix"]:
        raise SystemExit("result is not a complete schema-v1 frozen search")
    records = payload["records"]
    if len(records) != len(EXPECTED_ORDERS) * len(EXPECTED_MASKS):
        raise SystemExit("wrong number of order/policy records")
    observed = {(record["order"], record["mask"]) for record in records}
    expected = set(itertools_product(EXPECTED_ORDERS, EXPECTED_MASKS))
    if observed != expected:
        raise SystemExit("order/policy matrix is incomplete or duplicated")

    for record in records:
        numeric_values = (
            record["maximum_linf_normalized"],
            record["maximum_rms_normalized"],
            record["safety"]["constant_rhs_linf"],
            record["safety"]["conservation_residual"],
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise SystemExit(f"non-finite metric in {record['policy_id']}")
        if record["classification"] != expected_classification(payload, record):
            raise SystemExit(f"classification mismatch in {record['policy_id']}")
        assignment = record["assignment"]
        if set(assignment) != set(payload["precision_blocks"]):
            raise SystemExit(f"assignment mismatch in {record['policy_id']}")
        if any(value not in {"float32", "float64"} for value in assignment.values()):
            raise SystemExit(f"invalid dtype in {record['policy_id']}")

    print(
        f"verified {len(records)} records; sha256={expected_hash}; "
        f"source={payload['source_commit']}"
    )


def itertools_product(left: tuple[int, ...], right: tuple[int, ...]):
    for first in left:
        for second in right:
            yield first, second


if __name__ == "__main__":
    main()
