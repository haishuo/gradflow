#!/usr/bin/env python3
"""Offline checksum and semantic verifier for frozen U4-D D2/D3 evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVIDENCE = HERE / "evidence" / "u4d_campaign_20260830"
QUALIFICATION = HERE / "evidence" / "u4d_d1_20260830" / "qualification.json"
CAMPAIGN = HERE / "run_campaign.py"
SIZE = 8192
LANES = ("dveb", "opensbli", "gradflow")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_temperature(record: dict) -> None:
    assert record["telemetry_before"]["temperature_c"] < 80.0
    assert record["telemetry_after"]["temperature_c"] < 80.0


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

    record = json.loads((EVIDENCE / "campaign.json").read_text())
    assert record["schema"] == "gradflow.academic_u4d.campaign.v1"
    assert record["complete"] is True
    assert record["size"] == SIZE
    assert record["protocol_commit"] == "ce21751"
    assert record["d1_commit"] == "ea7359c"
    assert record["qualification_record_sha256"] == digest(QUALIFICATION)
    assert record["artifacts"]["campaign_harness_sha256"] == digest(CAMPAIGN)
    assert record["environment"]["cpu_threads"] == 1

    for device in ("cpu", "cuda"):
        resident = record["resident"][device]
        assert resident["workers_per_lane"] == 6
        assert resident["warmups_per_worker"] == 5
        assert resident["samples_per_worker"] == 20
        assert len(resident["randomized_blocks"]) == 6
        assert all(set(block["order"]) == set(LANES) for block in resident["randomized_blocks"])
        assert resident["analysis"]["overall_winner"] == "opensbli"
        workers = resident["worker_records"]
        assert set(workers) == set(LANES)
        medians = {}
        for lane in LANES:
            assert len(workers[lane]) == 6
            assert all(len(worker["samples_milliseconds"]) == 20 for worker in workers[lane])
            medians[lane] = [
                statistics.median(worker["samples_milliseconds"])
                for worker in workers[lane]
            ]
            assert resident["analysis"]["lanes"][lane]["worker_medians"]["values"] == medians[lane]
            if device == "cuda":
                for worker in workers[lane]:
                    check_temperature(worker)
            if lane == "gradflow":
                assert all(
                    worker["graph"] == {"unique_graphs": 1, "graph_break_count": 0}
                    for worker in workers[lane]
                )
        for left, right in itertools.combinations(LANES, 2):
            pair = resident["analysis"]["paired_worker_median_ratios"][f"{left}_over_{right}"]
            expected = [a / b for a, b in zip(medians[left], medians[right])]
            assert pair["values"] == expected
            assert pair["decision"] != "unresolved"

    transfer = record["transfer"]
    assert transfer["statistical_winner_prohibited"] is True
    assert transfer["warmups"] == 5 and transfer["samples"] == 20
    assert set(transfer["order"]) == set(LANES)
    for lane in LANES:
        lane_record = transfer["records"][lane]
        assert len(lane_record["samples_milliseconds"]) == 20
        assert lane_record["correctness"]["passed"] is True
        array = EVIDENCE / "endpoint_arrays" / f"transfer_{lane}_rhs.bin"
        assert array.stat().st_size == SIZE * 8
        assert lane_record["rhs_sha256"] == digest(array)
        assert transfer["analysis"][lane]["median"] == statistics.median(
            lane_record["samples_milliseconds"]
        )
        check_temperature(lane_record)
    for left, right in itertools.combinations(LANES, 2):
        expected = transfer["analysis"][left]["median"] / transfer["analysis"][right]["median"]
        assert transfer["analysis"][f"median_ratio_{left}_over_{right}"] == expected

    assert record["preparation"]["gradflow_aot_build"]["status"] == "complete"
    assert record["aot_admission"]["status"] == "qualified"
    assert record["aot_admission"]["comparison"]["passed"] is True
    aot = EVIDENCE / "endpoint_arrays" / "aot_qualification_rhs.bin"
    assert aot.stat().st_size == SIZE * 8
    assert record["aot_admission"]["rhs_sha256"] == digest(aot)

    launch = record["prepared_launch"]
    assert launch["statistical_winner_prohibited"] is True
    assert launch["repetitions_per_lane"] == 3
    assert len(launch["randomized_blocks"]) == 3
    launch_lanes = ("dveb", "opensbli", "gradflow_aot")
    for lane in launch_lanes:
        records = launch["records"][lane]
        assert len(records) == 3
        assert all(item["finite_checksum"] for item in records)
        assert all(item["repetition"] == index for index, item in enumerate(records))
        values = [item["parent_launch_to_answer_seconds"] for item in records]
        assert launch["analysis"][lane]["median"] == statistics.median(values)
        for item in records:
            check_temperature(item)
    for left, right in itertools.combinations(launch_lanes, 2):
        expected = launch["analysis"][left]["median"] / launch["analysis"][right]["median"]
        assert launch["analysis"][f"median_ratio_{left}_over_{right}"] == expected

    state = ROOT / "experiments" / "academic_u4c" / "evidence" / "u4c_c2_20260830" / "qualification_arrays" / f"n{SIZE}_state.bin"
    canonical = state.with_name(f"n{SIZE}_canonical.bin")
    assert record["artifacts"]["frozen_state_sha256"] == digest(state)
    assert record["artifacts"]["canonical_rhs_sha256"] == digest(canonical)
    print("U4-D D2/D3 three-way campaign evidence verified")


if __name__ == "__main__":
    main()
