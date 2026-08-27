#!/usr/bin/env python3
"""Verify the frozen Phase-D scalar CUDA benchmark record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED = {
    (order, policy)
    for order in (5, 11, 15)
    for policy in (
        "all_f64",
        "indicators_f32",
        "weight_formation_f32",
        "indicators_and_weight_formation_f32",
        "all_internal_f32",
    )
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    for line in (args.result_dir / "SHA256SUMS").read_text().splitlines():
        expected_hash, name = line.split()
        path = args.result_dir / name
        if not path.is_file() or sha256(path) != expected_hash:
            raise SystemExit(f"checksum mismatch: {name}")

    payload = json.loads((args.result_dir / "benchmark.json").read_text())
    records = payload["records"]
    observed = {(record["order"], record["policy"]) for record in records}
    if len(records) != len(EXPECTED) or observed != EXPECTED:
        raise SystemExit("benchmark matrix is incomplete or duplicated")
    for record in records:
        if record["status"] == "completed":
            for mode in ("eager", "compiled"):
                values = record[mode]
                if len(values["samples_ms"]) != 30:
                    raise SystemExit("timing record does not have 30 samples")
                checked = (
                    values["median_ms"],
                    values["q1_ms"],
                    values["q3_ms"],
                    values["mean_ms"],
                    values["speedup_vs_all_f64"],
                )
                if not all(math.isfinite(value) and value > 0.0 for value in checked):
                    raise SystemExit("invalid timing value")
            parity = record["compiled_parity"]
            parity_values = (
                parity["maximum_absolute_difference"],
                parity["rms_absolute_difference"],
                parity["maximum_normalized_difference"],
                parity["rms_normalized_difference"],
            )
            if not all(math.isfinite(value) and value >= 0.0 for value in parity_values):
                raise SystemExit("invalid compiled-parity value")
            if record["policy"] != "all_internal_f32" and not parity["passed"]:
                raise SystemExit("eligible policy failed compiled parity")
    print(
        f"verified {len(records)} benchmark records; "
        f"completed={sum(r['status'] == 'completed' for r in records)}; "
        f"source={payload['source_commit']}"
    )


if __name__ == "__main__":
    main()
