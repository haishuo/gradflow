#!/usr/bin/env python3
"""Offline integrity and semantic verifier for frozen U4-E E1 evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from array import array


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence" / "u4e_e1_20260831"
SIZE = 8192
PROTOCOL_COMMIT = "4788585cecd3a765e452406e08e4de5788ae7f0b"
BUNDLE_SHA256 = "2342f66416b1b120efd42e0e4ca8838f32cef4c62a13bf43042fb12ef7354ae0"
LIBRARY_SHA256 = "9ff9172b1ac712b8bc97ca9523fd114b2637e5d7825259371ba9850459168443"
LANES = (
    "dveb_cpu", "dveb_cuda", "opensbli_cpu", "opensbli_cuda",
    "gradflow_cpu", "gradflow_cuda",
)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    entries = [
        line.split("  ", 1)
        for line in (EVIDENCE / "SHA256SUMS").read_text().splitlines()
    ]
    assert entries
    for expected, relative in entries:
        path = EVIDENCE / relative
        assert path.is_file(), relative
        assert digest(path) == expected, relative

    record = json.loads((EVIDENCE / "qualification.json").read_text())
    assert record["schema"] == "gradflow.academic_u4e.qualification.v1"
    assert record["decision"] == "all_six_lanes_qualified"
    assert record["timing_interpretation_prohibited"] is True
    assert record["protocol_commit"] == PROTOCOL_COMMIT
    assert record["size"] == SIZE
    assert record["bounds"] == {
        "maximum_normalized": 5.0e-11,
        "rms_normalized": 5.0e-12,
    }
    assert record["canonical"]["state_sha256"] == (
        "7def0f1a410959390af68416a01f92d0ec917a23aaf022f5b90d52c366bb5530"
    )
    assert record["canonical"]["rhs_sha256"] == (
        "d92a1dd5f20cba9533dd25682fd19ca2d39f584b883b9fee3c994f1dd46b3621"
    )
    assert record["canonical"]["finite"] is True
    assert record["canonical"]["conservation"]["passed"] is True

    assert record["handoff"]["abi_version"] == 1
    assert record["handoff"]["bundle_sha256"] == BUNDLE_SHA256
    assert record["handoff"]["members"]["weno5_schedule_abi_v1.so"] == LIBRARY_SHA256
    assert record["artifacts"]["dveb_library_sha256"] == LIBRARY_SHA256
    assert record["sources"]["dveb_closure"] == {
        "commit": "39bd1c323daa3dbce6421a09dc34dc0cd2109d88",
        "tree": "3711d334ee48f24717900456f17c6518a1f0bada",
    }
    driver = HERE / "adapter" / "dveb_u4e_abi_driver.cpp"
    assert record["sources"]["driver_sha256"] == digest(driver)

    for lane in LANES:
        path = EVIDENCE / "qualification_arrays" / f"{lane}.bin"
        values = array("d")
        values.frombytes(path.read_bytes())
        result = record["qualification"][lane]
        assert len(values) == SIZE
        assert all(math.isfinite(value) for value in values)
        assert digest(path) == result["sha256"]
        assert result["finite"] is True
        assert result["conservation"]["passed"] is True
        assert result["maximum_normalized"] <= record["bounds"]["maximum_normalized"]
        assert result["rms_normalized"] <= record["bounds"]["rms_normalized"]
        assert result["passed"] is True
        if lane.startswith("gradflow_"):
            assert result["metadata"]["worker"]["graph"] == {
                "unique_graphs": 1,
                "graph_break_count": 0,
            }

    expected_policy = {
        "dveb_cpu": {
            "target": "cpu", "cpu_loop": 2, "cuda_block": 0, "reuse": 2,
            "launches": 2, "scratch_bytes": 65584, "elements": SIZE,
        },
        "dveb_cuda": {
            "target": "cuda", "cpu_loop": 0, "cuda_block": 32, "reuse": 2,
            "launches": 2, "scratch_bytes": 65584, "elements": SIZE,
        },
    }
    for lane, expected in expected_policy.items():
        metadata = record["qualification"][lane]["metadata"]
        for kind in ("query", "run"):
            for key, value in expected.items():
                assert metadata[kind][key] == value
        assert metadata["query"]["synchronized"] == 0
    assert record["qualification"]["dveb_cpu"]["metadata"]["run"]["synchronized"] == 1
    assert record["qualification"]["dveb_cuda"]["metadata"]["run"]["synchronized"] == 0
    assert record["qualification"]["dveb_cpu_cuda"]["passed"] is True
    print("U4-E E1 six-lane qualification evidence verified")


if __name__ == "__main__":
    main()
