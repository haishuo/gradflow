#!/usr/bin/env python3
"""Offline checksum and semantic verifier for Academic U5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BATCHES = (1, 4, 16, 64, 256, 1024)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    root = arguments.evidence.resolve()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert digest(root / relative) == expected, relative

    environment = load(root / "environment.json")
    assert environment["schema"] == "gradflow.academic_u5.environment.v1"
    assert environment["torch"] == "2.13.0+cu130"
    assert environment["torch_git_version"] == "cf30153c4c131c8164ee7798e5022d810682e2cb"
    assert environment["cuda_available"]
    assert environment["compute_capability"] == [12, 0]

    a1 = load(root / "a1/numerical_limits.json")
    assert a1["complete"] and a1["qualified_orders"] == [5, 7, 9, 11, 13, 15]
    assert all(record["all_finite"] and record["all_conservative"] for record in a1["roundoff_sweeps"])
    assert all(record["all_finite"] and record["all_conservative"] for record in a1["epsilon_sweeps"])

    campaign = load(root / "a2/campaign.json")
    assert campaign["complete"] and len(campaign["configurations"]) == 46
    eligible = [record for record in campaign["workers"].values() if record.get("protocol_eligible", True)]
    assert len(eligible) == 90
    assert all(record["returncode"] == 0 and record["record"]["status"] == "complete" for record in eligible)
    graphs = [record["record"]["correctness"]["compiled"]["graph"] for record in eligible]
    assert all(graph["unique_graphs"] == 1 and graph["graph_break_count"] == 0 for graph in graphs)

    aot = load(root / "a2/aot.json")
    assert aot["complete"]
    for order in (5, 11, 15):
        record = aot["orders"][str(order)]
        assert record["build_returncode"] == 0 and record["worker_returncode"] == 0
        assert record["qualification"]["correctness"]["jit"]["admitted"]
        assert record["qualification"]["correctness"]["aot"]["admitted"]
    for name in ("deployment.json", "deployment_isolated_cache.json"):
        deployment = load(root / "a2" / name)
        assert deployment["complete"]
        admitted = [record for record in deployment["configurations"] if record["eligible"]]
        assert len(admitted) == 8
        assert all(len(record["records"]) == 3 for record in admitted)
        assert all("failure" not in record for record in admitted)

    a3 = load(root / "a3/campaign.json")
    assert a3["complete"] and a3["derivative_gate"]["registered_window_passed"]
    assert a3["inverse_gate"]["passed"]
    for device in ("cpu", "cuda"):
        record = a3["benchmarks"][device]
        assert record["returncode"] == 0
        assert record["record"]["eager"]["admitted"]
        assert record["record"]["compiled"]["admitted"]
        graph = record["record"]["compiled"]["graph"]
        assert graph["unique_graphs"] == 1 and graph["graph_break_count"] == 0

    u4f = load(root / "u4f/campaign.json")
    assert u4f["complete"] and tuple(u4f["batches"]) == BATCHES
    expected_cuda = {
        1: "dveb_native_win", 4: "dveb_native_win", 16: "dveb_native_win",
        64: "pytorch_inductor_win", 256: "pytorch_inductor_win",
        1024: "pytorch_inductor_win",
    }
    for batch in BATCHES:
        cell = u4f["cells"][str(batch)]
        assert cell["admitted"] == {"cpu": True, "cuda": True}
        assert cell["resident"]["cpu"]["analysis"]["decision"] == "dveb_native_win"
        assert cell["resident"]["cuda"]["analysis"]["decision"] == expected_cuda[batch]

    comparison = load(root / "comparison.json")
    assert comparison["schema"] == "gradflow.academic_u5.comparison.v1"
    assert comparison["a1"]["roundoff_sweeps_identical"]
    assert comparison["a1"]["epsilon_sweeps_identical"]
    assert comparison["u4f"]["batched_cpu_compiler_failure_fixed"]
    assert comparison["claim_boundary"]["canonical_math_changed"] is False
    print("GradFlow academic U5 evidence verified")


if __name__ == "__main__":
    main()

