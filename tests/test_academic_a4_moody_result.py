from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments/academic_a4/evidence/moody_20260831"


def test_moody_result_preserves_complete_scientific_surface() -> None:
    record = json.loads((EVIDENCE / "second_machine.json").read_text())
    qualification = record["qualification"]
    assert record["status"] == "fail_needs_investigation"
    assert len(record["a2_workers"]) == 36
    assert qualification["a1_completed"]
    assert qualification["a3_agreement_passed"]
    assert qualification["a2_worker_surface_complete"]
    assert qualification["a2_graph_contract_passed"]
    assert qualification["binary32_cuda_materially_useful"]
    assert not qualification["admission_failures"]


def test_moody_result_keeps_packet_failures_visible() -> None:
    record = json.loads((EVIDENCE / "second_machine.json").read_text())
    sentinels = {item["name"]: item for item in record["sentinels"]}
    assert sentinels["pytest"]["returncode"] == 1
    for name in ("verify_a1", "verify_a2", "verify_a3", "verify_u5", "verify_a4_rc2"):
        assert sentinels[name]["returncode"] == 0
    pytest_text = (EVIDENCE / "raw/pytest.stdout").read_text()
    assert "4 failed, 351 passed, 12 skipped" in pytest_text


def test_moody_fastest_cuda_wins_registered_cells() -> None:
    record = json.loads((EVIDENCE / "second_machine.json").read_text())
    cells = record["a2_analysis"]["cells"]
    assert {(cell["order"], cell["dtype"]) for cell in cells} == {
        (order, dtype)
        for order in (5, 11, 15)
        for dtype in ("float32", "float64")
    }
    assert all(0.0 < cell["fastest_cuda_over_fastest_cpu"] < 1.0 for cell in cells)
