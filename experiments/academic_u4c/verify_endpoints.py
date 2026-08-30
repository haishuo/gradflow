#!/usr/bin/env python3
"""Offline checksum and semantic verifier for frozen U4-C C3 evidence."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence" / "u4c_c3_20260830"


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

    record = json.loads((EVIDENCE / "endpoints.json").read_text())
    assert record["schema"] == "gradflow.academic_u4c.endpoints.v1"
    assert record["complete"] is True
    assert record["size"] == 8192
    assert record["transfer"]["warmups"] == 5
    assert record["transfer"]["samples"] == 20
    assert set(record["transfer"]["records"]) == {"opensbli", "gradflow"}
    for lane, lane_record in record["transfer"]["records"].items():
        assert len(lane_record["samples_milliseconds"]) == 20
        assert lane_record["correctness"]["passed"] is True
        assert lane_record["correctness"]["finite"] is True
        rhs = EVIDENCE / f"transfer_{lane}_rhs.bin"
        assert rhs.stat().st_size == 8192 * 8
        assert lane_record["rhs_sha256"] == digest(rhs)
        expected_median = statistics.median(lane_record["samples_milliseconds"])
        assert record["transfer"]["analysis"][lane]["median"] == expected_median
    expected_transfer_ratio = (
        record["transfer"]["analysis"]["opensbli"]["median"]
        / record["transfer"]["analysis"]["gradflow"]["median"]
    )
    assert (
        record["transfer"]["analysis"][
            "median_ratio_opensbli_over_gradflow"
        ]
        == expected_transfer_ratio
    )

    assert record["aot_build"]["status"] == "complete"
    assert record["aot_build"]["package_bytes"] > 0
    assert len(record["aot_build"]["package_sha256"]) == 64
    assert record["aot_admission"]["status"] == "qualified"
    assert record["aot_admission"]["comparison"]["passed"] is True

    launch = record["prepared_launch"]
    assert launch["repetitions"] == 3
    for lane in ("opensbli", "gradflow_aot"):
        assert len(launch["records"][lane]) == 3
        values = [
            item["parent_launch_to_answer_seconds"]
            for item in launch["records"][lane]
        ]
        assert launch["analysis"][lane]["median"] == statistics.median(values)
    assert all(item["finite_checksum"] for item in launch["records"]["opensbli"])
    assert all(
        item["worker"]["finite"]
        for item in launch["records"]["gradflow_aot"]
    )
    expected_launch_ratio = (
        launch["analysis"]["opensbli"]["median"]
        / launch["analysis"]["gradflow_aot"]["median"]
    )
    assert (
        launch["analysis"]["median_ratio_opensbli_over_gradflow_aot"]
        == expected_launch_ratio
    )
    print("U4-C C3 endpoint evidence verified")


if __name__ == "__main__":
    main()
