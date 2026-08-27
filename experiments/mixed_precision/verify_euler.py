#!/usr/bin/env python3
"""Verify the Phase-D Tier-2 Euler mixed-precision qualification record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ORDERS = (5, 7, 9, 11, 13, 15)
REPRESENTATIVE_ORDERS = (5, 11, 15)
POLICIES = (
    "all_f64",
    "indicators_f32",
    "weight_formation_f32",
    "indicators_and_weight_formation_f32",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(directory: Path) -> None:
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"checksum mismatch: {name}")


def expected_keys(orders: tuple[int, ...]) -> set[str]:
    return {f"order{order}_{policy}" for order in orders for policy in POLICIES}


def verify(directory: Path) -> dict:
    verify_checksums(directory)
    payload = json.loads((directory / "qualification.json").read_text())
    if payload["schema_version"] != 1:
        raise ValueError("unknown Tier-2 schema")
    if set(payload["local"]) != expected_keys(ORDERS):
        raise ValueError("local matrix is incomplete")
    for section in ("repeated_step", "gradients", "compiler_device"):
        if set(payload[section]) != expected_keys(REPRESENTATIVE_ORDERS):
            raise ValueError(f"{section} matrix is incomplete")
    shock_keys = {
        f"{problem}_order{order}_{policy}"
        for problem in ("sod", "shu_osher")
        for order in REPRESENTATIVE_ORDERS
        for policy in POLICIES
    }
    if set(payload["shocks"]) != shock_keys:
        raise ValueError("shock matrix is incomplete")

    all_records = []
    for section in ("local", "repeated_step", "gradients", "compiler_device"):
        all_records.extend(payload[section].values())
    all_records.extend(payload["shocks"].values())
    for record in all_records:
        if not isinstance(record["passed"], bool):
            raise ValueError("non-boolean gate decision")
    numeric = []
    for record in payload["local"].values():
        numeric.extend(
            (
                case["maximum_linf_normalized"],
                case["maximum_rms_normalized"],
            )
            for case in record["parity_cases"].values()
        )
    if not all(math.isfinite(value) for pair in numeric for value in pair):
        raise ValueError("non-finite local metric")

    expected_decision = "PASS" if (
        all(record["passed"] for record in all_records)
        and payload["static"]["passed"]
    ) else "FAIL"
    if payload["decision"] != expected_decision:
        raise ValueError("top-level decision mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    payload = verify(args.result_dir)
    print(
        f"verified Tier 2 decision={payload['decision']} "
        f"source={payload['source_commit']}"
    )


if __name__ == "__main__":
    main()
