#!/usr/bin/env python3
"""Offline checksum and semantic verifier for the frozen U4-C C1 record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence" / "u4c_c1_20260830"
ATOL = 2.0e-12


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = EVIDENCE / "SHA256SUMS"
    entries = [line.split("  ", 1) for line in manifest.read_text().splitlines()]
    assert entries
    for expected, relative in entries:
        path = EVIDENCE / relative
        assert path.is_file(), relative
        assert digest(path) == expected, relative

    record = json.loads((EVIDENCE / "qualification.json").read_text())
    assert record["schema"] == "gradflow.academic_u4c.cuda_qualification.v1"
    assert record["decision"] == "cuda_correctness_qualified"
    assert record["timing_interpretation_prohibited"] is True
    assert record["atol"] == ATOL
    assert record["environment"]["cuda_target"] == "sm_120"
    assert [case["case"] for case in record["cases"]] == [
        "state_a",
        "state_b",
        "constant",
    ]

    arrays = np.load(EVIDENCE / "qualification_arrays.npz")
    for case in record["cases"]:
        name = case["case"]
        seq = arrays[f"{name}_seq_rhs"]
        cuda = arrays[f"{name}_cuda_rhs"]
        canonical = arrays[f"{name}_canonical_rhs"]
        assert seq.shape == cuda.shape == canonical.shape == (64,)
        assert np.all(np.isfinite(seq)) and np.all(np.isfinite(cuda))
        assert np.max(np.abs(seq - cuda)) <= ATOL
        assert np.max(np.abs(cuda - canonical)) <= ATOL
        assert case["passed"] is True
        assert case["seq_conservation"]["passed"] is True
        assert case["cuda_conservation"]["passed"] is True
        if name == "constant":
            assert np.max(np.abs(cuda)) <= ATOL

    print("U4-C C1 CUDA qualification evidence verified")


if __name__ == "__main__":
    main()
