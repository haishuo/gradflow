#!/usr/bin/env python3
"""Verify the immutable files and recorded conclusion of G3 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(directory: Path) -> None:
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        path = directory / relative
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {relative}: {actual}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()
    verify_checksums(evidence)

    record = json.loads((evidence / "qualification.json").read_text())
    assert record["schema"] == "gradflow-g3-r6q-qualification-v1"
    assert record["identity"]["passed"]
    assert all(
        row["fp32_parity"]["linf"] <= record["tolerances"]["step_atol"]
        for row in record["full_step_parity"]
    )
    assert all(
        row["fp32_parity"]["linf"] <= record["tolerances"]["step_atol"]
        for row in record["periodic_discontinuity_stress"]
    )
    assert all(
        row["admissibility"]["finite"] and row["admissibility"]["positive"]
        for row in (
            record["full_step_parity"]
            + record["periodic_discontinuity_stress"]
        )
    )
    assert record["smooth_spatial_convergence"]["convergence_passed"]
    assert record["directional_sensitivity"]["passed"]

    rhs_rows = (
        record["smooth_spatial_convergence"]["rows"]
        + record["critical_point_characterization"]["rows"]
    )
    assert all(
        row["fp32_parity"]["linf"] <= record["tolerances"]["rhs_atol"]
        for row in rhs_rows
    )
    assert any(not row["parity_passed"] for row in rhs_rows)
    assert any(
        not row["conservation"]["passed"]
        for row in (
            record["full_step_parity"]
            + record["periodic_discontinuity_stress"]
        )
    )
    assert not record["passed"]
    assert record["differentiability"]["backend_admission_blocker"]
    print("G3 qualification evidence and strict non-admission conclusion verify.")


if __name__ == "__main__":
    main()
