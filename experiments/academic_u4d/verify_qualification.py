#!/usr/bin/env python3
"""Offline checksum and semantic verifier for the frozen U4-D D1 record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence" / "u4d_d1_20260830"
SIZE = 8192
LANES = (
    "dveb_cpu",
    "dveb_cuda",
    "opensbli_cpu",
    "opensbli_cuda",
    "gradflow_cpu",
    "gradflow_cuda",
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    manifest = EVIDENCE / "SHA256SUMS"
    entries = [line.split("  ", 1) for line in manifest.read_text().splitlines()]
    assert entries
    for expected, relative in entries:
        path = EVIDENCE / relative
        assert path.is_file(), relative
        assert digest(path) == expected, relative

    record = json.loads((EVIDENCE / "qualification.json").read_text())
    assert record["schema"] == "gradflow.academic_u4d.qualification.v1"
    assert record["decision"] == "all_six_lanes_qualified"
    assert record["timing_interpretation_prohibited"] is True
    assert record["size"] == SIZE
    assert record["bounds"] == {
        "maximum_normalized": 5.0e-11,
        "rms_normalized": 5.0e-12,
    }
    assert record["environment"]["dveb_contract"] == "fma"
    assert record["environment"]["cpu_threads"] == 1

    for lane in LANES:
        path = EVIDENCE / "qualification_arrays" / f"{lane}.bin"
        values = np.fromfile(path, dtype=np.float64)
        result = record["qualification"][lane]
        assert values.shape == (SIZE,)
        assert np.all(np.isfinite(values))
        assert digest(path) == result["sha256"]
        assert result["finite"] is True
        assert result["conservation"]["passed"] is True
        assert result["maximum_normalized"] <= record["bounds"]["maximum_normalized"]
        assert result["rms_normalized"] <= record["bounds"]["rms_normalized"]
        assert result["passed"] is True

    for device in ("cpu", "cuda"):
        worker = record["qualification"][f"gradflow_{device}"]["metadata"]["worker"]
        assert worker["graph"] == {"unique_graphs": 1, "graph_break_count": 0}
        comparison = record["qualification"][f"dveb_{device}"][
            "versus_gradflow_same_device"
        ]
        assert comparison["passed"] is True
    assert record["qualification"]["dveb_cpu_cuda"]["passed"] is True

    print("U4-D D1 six-lane qualification evidence verified")


if __name__ == "__main__":
    main()
