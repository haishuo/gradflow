#!/usr/bin/env python3
"""Offline verifier for the frozen U4-F evidence bundle."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVIDENCE = HERE / "evidence" / "u4f_20260831"
BATCHES = (1, 4, 16, 64, 256, 1024)
DEVICES = ("cpu", "cuda")
LANES = ("dveb_native", "pytorch_inductor")
PROTOCOL_COMMIT = "ef1ac91f1d0c3ddbaa59c4e8b9f6b4eef9685195"
AMENDMENT_COMMIT = "de11c8171d54fb34b1de848947bf31cc09b01f99"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_checksums() -> None:
    for line in (EVIDENCE / "SHA256SUMS").read_text().splitlines():
        wanted, relative = line.split("  ", 1)
        path = EVIDENCE / relative
        assert path.is_file(), path
        assert digest(path) == wanted, path


def expected_decision(ratios: list[float], interval: list[float]) -> str:
    median = statistics.median(ratios)
    if median < 0.95 and interval[1] < 1.0:
        return "pytorch_inductor_win"
    if median > 1.05 and interval[0] > 1.0:
        return "dveb_native_win"
    return "unresolved"


def main() -> None:
    verify_checksums()
    document = json.loads((EVIDENCE / "campaign.json").read_text())
    assert document["schema"] == "gradflow.academic_u4f.campaign.v1"
    assert document["complete"] is True
    assert document["protocol_commit"] == PROTOCOL_COMMIT
    assert document["protocol_amendment_commit"] == AMENDMENT_COMMIT
    assert document["size"] == 8192
    assert tuple(document["batches"]) == BATCHES
    assert set(map(int, document["cells"])) == set(BATCHES)
    assert document["sources"]["adapter_sha256"] == digest(
        HERE / "adapter" / "dveb_u4f_batch_driver.cpp"
    )
    assert document["sources"]["pytorch_worker_sha256"] == digest(
        HERE / "pytorch_batch_worker.py"
    )
    assert (EVIDENCE / "COMMANDS.txt").stat().st_size > 0

    for batch in BATCHES:
        cell = document["cells"][str(batch)]
        assert cell["batch"] == batch
        assert cell["points"] == batch * 8192
        qualification = cell["qualification"]
        expected_keys = {
            f"{lane}_{device}" for lane in LANES for device in DEVICES
        } | {f"{lane}_cpu_cuda" for lane in LANES}
        assert set(qualification) == expected_keys
        for device in DEVICES:
            admitted = all(
                qualification[f"{lane}_{device}"]["passed"] for lane in LANES
            )
            assert cell["admitted"][device] == admitted
            assert cell["status"][device] == (
                "timed" if admitted else "correctness_excluded"
            )
            if not admitted:
                assert device not in cell["resident"]
                continue
            resident = cell["resident"][device]
            assert resident["workers_per_lane"] == 6
            assert resident["warmups_per_worker"] == 5
            assert resident["samples_per_worker"] == 20
            assert len(resident["randomized_blocks"]) == 6
            assert set(resident["worker_records"]) == set(LANES)
            medians = {}
            for lane in LANES:
                workers = resident["worker_records"][lane]
                assert len(workers) == 6
                assert all(len(row["samples_milliseconds"]) == 20 for row in workers)
                medians[lane] = [
                    statistics.median(row["samples_milliseconds"]) for row in workers
                ]
                retained = resident["analysis"]["lanes"][lane]["worker_medians"]
                assert len(retained["values"]) == 6
                assert math.isclose(retained["median"], statistics.median(medians[lane]))
            ratios = [
                pytorch / dveb
                for pytorch, dveb in zip(
                    medians["pytorch_inductor"], medians["dveb_native"]
                )
            ]
            paired = resident["analysis"][
                "paired_worker_median_ratio_pytorch_over_dveb"
            ]
            assert all(math.isclose(a, b) for a, b in zip(ratios, paired["values"]))
            assert paired["decision"] == expected_decision(
                ratios, paired["bootstrap_median_95_ci"]
            )
            assert resident["analysis"]["decision"] == paired["decision"]
    print("GradFlow academic U4-F evidence verified")


if __name__ == "__main__":
    main()
