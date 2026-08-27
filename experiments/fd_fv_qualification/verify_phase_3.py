#!/usr/bin/env python3
"""Verify the frozen FD/FV Phase-3 qualification record and decision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = (
    ROOT / "experiments/fd_fv_qualification/results/phase_3_20260827"
)
RECORD_PATH = RESULT_DIR / "qualification.json"
SOURCE_COMMIT = "1d920ea97ed7abec9e4e451b377343cf72316f4c"
EXPECTED_FAILED_GATES = ["smooth_spatial", "transfer_evidence"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    record = json.loads(RECORD_PATH.read_text())
    assert record["schema_version"] == 1
    assert record["phase"] == "fd_fv_phase_3"
    assert record["formulation_id"] == (
        "fv_dimensional_js5_global_lf_periodic_v1"
    )
    assert record["source_commit"] == SOURCE_COMMIT
    assert record["source_dirty"] is False
    assert record["performance_measurements_collected"] is False

    for relative, expected in record["source_hashes"].items():
        assert sha256(ROOT / relative) == expected
    for name, expected in record["phase_2_hashes"].items():
        path = ROOT / "experiments/fd_fv_contract/results/phase_2_20260827" / name
        assert sha256(path) == expected

    assert record["oracle_parity"]["passed"] is True
    assert record["refusal_contract"]["passed"] is True
    assert all(record["refusal_contract"]["checks"].values())

    spatial = record["smooth_spatial"]
    positive = spatial["directions"]["1"]
    negative = spatial["directions"]["-1"]
    assert positive["monotone"] is True
    assert positive["maximum_rate"] >= 4.7
    assert positive["passed"] is True
    assert negative["monotone"] is True
    assert negative["maximum_rate"] < 4.7
    assert negative["passed"] is False
    assert spatial["passed"] is False

    complete = record["smooth_complete_solve"]
    assert complete["passed"] is True
    assert max(complete["l2_rates"]) >= 4.0
    assert all(run["conservation_passed"] for run in complete["runs"])

    discontinuity = record["discontinuity"]
    assert discontinuity["monotone_l1"] is True
    assert discontinuity["passed"] is True
    assert all(run["passed"] for run in discontinuity["runs"])

    gradients = record["differentiation"]
    assert gradients["rhs_gradcheck"] is True
    assert gradients["fixed_three_step"]["passed"] is True
    assert gradients["passed"] is True

    assert record["eager_cpu"]["passed"] is True
    assert record["cuda"]["status"] in {"passed", "untested_unavailable"}
    assert record["mps"]["status"] in {
        "passed",
        "not_executed",
        "untested_unavailable",
    }
    compiler = record["compiler"]
    assert compiler["passed"] is True
    assert compiler["compilation_latency_timed"] is False
    for case in compiler["cpu"].values():
        assert case["graph_count"] == 1
        assert case["graph_break_count"] == 0
        assert case["passed"] is True

    transfers = record["transfer_evidence"]
    assert transfers["static_forbidden_calls"] == []
    assert transfers["profiler_forbidden_events"] == ["aten::to"]
    assert transfers["passed"] is False

    assert record["failed_gates"] == EXPECTED_FAILED_GATES
    assert record["passed"] is False
    for name, decision in record["gate_decisions"].items():
        assert decision == (name not in EXPECTED_FAILED_GATES)

    expected_hash, expected_name = (
        RESULT_DIR / "SHA256SUMS"
    ).read_text().strip().split("  ", maxsplit=1)
    assert expected_name == "qualification.json"
    assert sha256(RECORD_PATH) == expected_hash

    print(
        "FD/FV Phase 3 verified: 9 gates passed, CUDA/MPS unavailable, and "
        "the frozen smooth-spatial and profiler-event gates remain failed."
    )


if __name__ == "__main__":
    main()
