#!/usr/bin/env python3
"""Offline checksum and semantic verification for Academic A2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORDERS = (5, 7, 9, 11, 13, 15)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(evidence: Path) -> None:
    for line in (evidence / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        actual = sha256(evidence / relative)
        assert actual == expected, f"checksum mismatch for {relative}: {actual}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()
    verify_checksums(evidence)

    campaign = json.loads((evidence / "campaign.json").read_text())
    assert campaign["schema"] == "gradflow-academic-a2-campaign-v1"
    assert campaign["complete"]
    assert campaign["protocol_commit"] == "6464c50"
    assert not campaign["canonical_source_changed"]
    assert len(campaign["configurations"]) == 46
    assert len(campaign["workers"]) == 91
    assert campaign["excluded_unregistered_workers"] == [
        "characteristic_o5_float32_d3_n64_cpu"
    ]
    eligible = [
        wrapper
        for wrapper in campaign["workers"].values()
        if wrapper.get("protocol_eligible", True)
    ]
    assert len(eligible) == 90
    assert all(wrapper["returncode"] == 0 for wrapper in eligible)
    assert all(wrapper["record"]["status"] == "complete" for wrapper in eligible)
    compiled_graphs = [
        wrapper["record"]["correctness"]["compiled"]["graph"] for wrapper in eligible
    ]
    assert all(graph["unique_graphs"] == 1 for graph in compiled_graphs)
    assert all(graph["graph_break_count"] == 0 for graph in compiled_graphs)

    aot = json.loads((evidence / "aot.json").read_text())
    assert aot["schema"] == "gradflow-academic-a2-aot-campaign-v1"
    assert aot["complete"]
    assert tuple(int(order) for order in aot["orders"]) == (5, 11, 15)
    decisions = []
    for order in (5, 11, 15):
        entry = aot["orders"][str(order)]
        assert entry["build_returncode"] == 0
        assert entry["worker_returncode"] == 0
        assert entry["build_record"]["status"] == "complete"
        qualification = entry["qualification"]
        assert qualification["status"] == "complete"
        assert qualification["correctness"]["jit"]["admitted"]
        assert qualification["correctness"]["aot"]["admitted"]
        assert qualification["jit_graph"]["unique_graphs"] == 1
        assert qualification["jit_graph"]["graph_break_count"] == 0
        decisions.append(
            qualification["resident_timing"]["paired_analysis"]["decision"]
        )
    assert decisions == ["aot_win", "aot_win", "unresolved"]

    prepared = json.loads((evidence / "deployment.json").read_text())
    isolated = json.loads((evidence / "deployment_isolated_cache.json").read_text())
    assert prepared["complete"] and isolated["complete"]
    assert "torchinductor_cache_policy" not in prepared
    assert isolated["torchinductor_cache_policy"] == "isolated"
    for document in (prepared, isolated):
        admitted = [item for item in document["configurations"] if item["eligible"]]
        assert len(admitted) == 8
        assert all(len(item["records"]) == 3 for item in admitted)
        assert all(
            record["returncode"] == 0 and record["worker"]["finite"]
            for item in admitted
            for record in item["records"]
        )
    assert all(
        item["all_checksums_identical"]
        for item in prepared["configurations"]
        if item["eligible"]
    )
    isolated_variability = [
        (item["order"], item["dimensions"], item["lane"])
        for item in isolated["configurations"]
        if item["eligible"] and not item["all_checksums_identical"]
    ]
    assert isolated_variability == [(15, 3, "cuda_compiled")]

    analysis = json.loads((evidence / "analysis.json").read_text())
    assert analysis["schema"] == "gradflow-academic-a2-analysis-v1"
    assert all(analysis["complete_inputs"].values())
    assert analysis["core_worker_counts"] == {
        "total": 91,
        "protocol_eligible": 90,
        "unregistered_excluded": 1,
    }
    assert len(analysis["cross_order"]) == 24
    assert len(analysis["scale"]) == 18
    assert len(analysis["characteristic"]) == 8
    assert len(analysis["correctness_exclusions"]) == 35
    for relative, expected in analysis["input_sha256"].items():
        assert sha256(ROOT / relative) == expected
    for relative, expected in analysis["source_sha256"].items():
        assert sha256(ROOT / relative) == expected

    scale = {
        (item["order"], item["dimensions"], item["n"]): item
        for item in analysis["scale"]
    }
    assert scale[(5, 3, 16)]["cuda_transfer_over_cpu_resident"] > 1
    assert scale[(5, 3, 32)]["cuda_transfer_over_cpu_resident"] < 1
    assert scale[(15, 3, 16)]["cuda_transfer_over_cpu_resident"] < 1
    assert scale[(5, 1, 32768)]["cuda_transfer_over_cpu_resident"] > 1
    assert scale[(15, 1, 512)]["best_cuda_resident"] is None

    results = (ROOT / "docs/ACADEMIC_A2_RESULTS.md").read_text()
    assert "correctness-admitted" in results
    assert "isolated empty compiler cache" in results.lower()
    assert "does not establish" in results
    assert "AOT win" in results and "unresolved" in results
    print("Academic A2 evidence, cache endpoints, exclusions, and conclusions verify.")


if __name__ == "__main__":
    main()
