#!/usr/bin/env python3
"""Offline checksum and semantic verifier for frozen U4-C C2 evidence."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence" / "u4c_c2_20260830"
SIZES = (8192, 131072, 1048576, 8388608)
LANES = ("opensbli_cpu", "opensbli_cuda", "gradflow_cpu", "gradflow_cuda")


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

    document = json.loads((EVIDENCE / "campaign.json").read_text())
    assert document["schema"] == "gradflow.academic_u4c.performance.v1"
    assert document["complete"] is True
    assert tuple(map(int, document["sizes"])) == SIZES
    assert document["protocol_commit"] == "2106a02"
    assert document["c1_commit"] == "7a1f696"

    for size in SIZES:
        record = document["sizes"][str(size)]
        qualification = record["qualification"]
        for lane in LANES:
            array = EVIDENCE / "qualification_arrays" / f"n{size}_{lane}.bin"
            assert array.stat().st_size == size * 8
            assert qualification[lane]["sha256"] == digest(array)
            assert qualification[lane]["finite"] is True
            assert qualification[lane]["conservation"]["passed"] is True
        state = EVIDENCE / "qualification_arrays" / f"n{size}_state.bin"
        canonical = EVIDENCE / "qualification_arrays" / f"n{size}_canonical.bin"
        assert state.stat().st_size == canonical.stat().st_size == size * 8
        assert qualification["canonical"]["state_sha256"] == digest(state)
        assert qualification["canonical"]["sha256"] == digest(canonical)

        if size == 8192:
            assert record["status"] == "complete"
            assert qualification["all_lanes_admitted"] is True
            for device in ("cpu", "cuda"):
                timing = record["timing"][device]
                assert timing["workers_per_lane"] == 6
                assert timing["warmups_per_worker"] == 5
                assert timing["samples_per_worker"] == 20
                assert len(timing["randomized_blocks"]) == 6
                assert timing["analysis"]["decision"] == "opensbli_win"
                worker_records = timing["worker_records"]
                assert set(worker_records) == {"opensbli", "gradflow"}
                for lane in worker_records:
                    assert len(worker_records[lane]) == 6
                    assert all(
                        len(worker["samples_milliseconds"]) == 20
                        for worker in worker_records[lane]
                    )
                expected_ratios = [
                    statistics.median(left["samples_milliseconds"])
                    / statistics.median(right["samples_milliseconds"])
                    for left, right in zip(
                        worker_records["opensbli"], worker_records["gradflow"]
                    )
                ]
                observed = timing["analysis"][
                    "paired_worker_median_ratio_opensbli_over_gradflow"
                ]
                assert expected_ratios == observed["values"]
                assert observed["median"] < 0.95
                assert observed["bootstrap_median_95_ci"][1] < 1.0
        else:
            assert record["status"] == "correctness_excluded"
            assert qualification["all_lanes_admitted"] is False
            assert qualification["opensbli_cpu"]["passed"] is False
            assert qualification["opensbli_cuda"]["passed"] is False
            assert "timing" not in record

    print("U4-C C2 performance evidence verified")


if __name__ == "__main__":
    main()
