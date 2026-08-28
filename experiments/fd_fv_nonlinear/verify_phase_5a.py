#!/usr/bin/env python3
"""Independently verify the frozen nonlinear Phase-5A oracle records."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fd_fv_nonlinear.freeze_phase_5a import freeze
from experiments.infrastructure.device_admission import classify_device_admission


FROZEN = ROOT / "experiments/fd_fv_nonlinear/results/phase_5a_20260828"
ORACLE = ROOT / "experiments/fd_fv_nonlinear/burgers_oracle.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_oracle_imports() -> None:
    tree = ast.parse(ORACLE.read_text())
    forbidden = {"torch", "numpy", "gradflow"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots = {node.module.split(".")[0]}
        else:
            continue
        assert roots.isdisjoint(forbidden), roots & forbidden


def _verify_manifest() -> None:
    entries = {}
    for line in (FROZEN / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    assert entries == {
        "contract.json": sha256(FROZEN / "contract.json"),
        "oracle_cases.json": sha256(FROZEN / "oracle_cases.json"),
    }


def _verify_infrastructure_classifier() -> None:
    assert classify_device_admission(
        process_visible=True, host_inventory="present", admission="passed"
    ) == "admitted"
    assert classify_device_admission(
        process_visible=True, host_inventory="present", admission="failed"
    ) == "visible_admission_failed"
    assert classify_device_admission(
        process_visible=True, host_inventory="unknown"
    ) == "visible_unqualified"
    assert classify_device_admission(
        process_visible=False, host_inventory="present"
    ) == "process_hidden_host_present"
    assert classify_device_admission(
        process_visible=False, host_inventory="absent"
    ) == "host_confirmed_absent"
    assert classify_device_admission(
        process_visible=False, host_inventory="unknown"
    ) == "not_visible_host_unknown"
    assert classify_device_admission(
        process_visible=None, host_inventory="unknown"
    ) == "probe_failed"


def main() -> None:
    _verify_oracle_imports()
    _verify_manifest()
    _verify_infrastructure_classifier()
    contract = json.loads((FROZEN / "contract.json").read_text())
    cases = json.loads((FROZEN / "oracle_cases.json").read_text())
    assert contract["schema"] == "gradflow.fd_fv.nonlinear.phase_5a.contract.v1"
    assert contract["performance_measurements_collected"] is False
    assert contract["production_burgers_implementation_added"] is False
    assert cases["summary"]["all_oracle_checks_passed"] is True
    assert float.fromhex(
        cases["analytic"]["minimum_characteristic_jacobian_hex"]
    ) > 0.0

    with tempfile.TemporaryDirectory() as directory:
        regenerated = Path(directory) / "phase_5a"
        freeze(regenerated)
        for name in ("contract.json", "oracle_cases.json", "SHA256SUMS"):
            assert (regenerated / name).read_bytes() == (
                FROZEN / name
            ).read_bytes()

    print(
        "FD/FV nonlinear Phase 5A verified: exact pre-shock point and "
        "cell-average oracles passed, no performance was collected, and "
        "device visibility vocabulary is unambiguous."
    )


if __name__ == "__main__":
    main()
