#!/usr/bin/env python3
"""Offline integrity and semantic verifier for frozen U4-E E2/E3 evidence."""

from __future__ import annotations

import hashlib
import itertools
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVIDENCE = HERE / "evidence" / "u4e_campaign_20260831"
QUALIFICATION = HERE / "evidence" / "u4e_e1_20260831" / "qualification.json"
CAMPAIGN = HERE / "run_campaign.py"
SIZE = 8192
LANES = ("dveb", "opensbli", "gradflow")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def check_telemetry(record: dict) -> None:
    assert record["temperature_c"] < 80.0
    assert record["active_throttle_reasons"] == "0x0000000000000000"


def check_dveb_policy(record: dict, device: str) -> None:
    expected = {
        "cpu": {
            "target": "cpu", "cpu_loop": 2, "cuda_block": 0, "reuse": 2,
            "launches": 2, "scratch_bytes": 65584, "elements": SIZE,
        },
        "cuda": {
            "target": "cuda", "cpu_loop": 0, "cuda_block": 32, "reuse": 2,
            "launches": 2, "scratch_bytes": 65584, "elements": SIZE,
        },
    }[device]
    for kind in ("query", "run"):
        for key, value in expected.items():
            assert record[kind][key] == value
    assert record["query"]["synchronized"] == 0
    assert record["run"]["synchronized"] == (1 if device == "cpu" else 0)


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
    assert record["schema"] == "gradflow.academic_u4e.campaign.v1"
    assert record["complete"] is True
    assert record["size"] == SIZE
    assert record["protocol_commit"] == "4788585cecd3a765e452406e08e4de5788ae7f0b"
    assert record["e1_commit"] == "a9e6e947ce79c9581138fcda0e20ea66191c528c"
    assert record["qualification_record_sha256"] == digest(QUALIFICATION)
    assert record["artifacts"]["campaign_harness_sha256"] == digest(CAMPAIGN)
    assert record["environment"]["cpu_threads"] == 1
    assert record["u4d_to_u4e_cross_campaign_comparison_is_descriptive"] is True

    for device in ("cpu", "cuda"):
        resident = record["resident"][device]
        assert resident["workers_per_lane"] == 6
        assert resident["warmups_per_worker"] == 5
        assert resident["samples_per_worker"] == 20
        assert len(resident["randomized_blocks"]) == 6
        assert all(set(block["order"]) == set(LANES) for block in resident["randomized_blocks"])
        assert resident["analysis"]["overall_winner"] == "dveb"
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
            if lane == "dveb":
                for worker in workers[lane]:
                    check_dveb_policy(worker, device)
            if lane == "gradflow":
                assert all(
                    worker["graph"] == {"unique_graphs": 1, "graph_break_count": 0}
                    for worker in workers[lane]
                )
            if device == "cuda":
                for worker in workers[lane]:
                    check_telemetry(worker["telemetry_before"])
                    check_telemetry(worker["telemetry_after"])
        for left, right in itertools.combinations(LANES, 2):
            pair = resident["analysis"]["paired_worker_median_ratios"][f"{left}_over_{right}"]
            expected = [a / b for a, b in zip(medians[left], medians[right])]
            assert pair["values"] == expected
            assert pair["decision"] != "unresolved"
        assert resident["analysis"]["paired_worker_median_ratios"]["dveb_over_opensbli"]["decision"] == "dveb_win"
        assert resident["analysis"]["paired_worker_median_ratios"]["dveb_over_gradflow"]["decision"] == "dveb_win"

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
        check_telemetry(lane_record["telemetry_before"])
        check_telemetry(lane_record["telemetry_after"])
        if lane == "dveb":
            check_dveb_policy(lane_record, "cuda")
    for left, right in itertools.combinations(LANES, 2):
        expected = transfer["analysis"][left]["median"] / transfer["analysis"][right]["median"]
        assert transfer["analysis"][f"median_ratio_{left}_over_{right}"] == expected

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
            check_telemetry(item["telemetry_before"])
            check_telemetry(item["telemetry_after"])
            if lane == "dveb":
                check_dveb_policy(item, "cuda")
    for left, right in itertools.combinations(launch_lanes, 2):
        expected = launch["analysis"][left]["median"] / launch["analysis"][right]["median"]
        assert launch["analysis"][f"median_ratio_{left}_over_{right}"] == expected

    state = ROOT / "experiments" / "academic_u4c" / "evidence" / "u4c_c2_20260830" / "qualification_arrays" / f"n{SIZE}_state.bin"
    canonical = state.with_name(f"n{SIZE}_canonical.bin")
    assert record["artifacts"]["frozen_state_sha256"] == digest(state)
    assert record["artifacts"]["canonical_rhs_sha256"] == digest(canonical)
    print("U4-E E2/E3 three-way campaign evidence verified")


if __name__ == "__main__":
    main()
