#!/usr/bin/env python3
"""Independently verify the frozen nonlinear Phase-5A oracle records."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fd_fv_nonlinear.freeze_phase_5a import build_cases  # noqa: E402
from experiments.infrastructure.device_admission import (  # noqa: E402
    classify_device_admission,
)


FROZEN = ROOT / "experiments/fd_fv_nonlinear/results/phase_5a_20260828"
ORACLE = ROOT / "experiments/fd_fv_nonlinear/burgers_oracle.py"
SOURCE_COMMIT = "0c974ab"
CONTRACT_SHA256 = "3250bf7aa4647324d683fdacf86117e3668cb54de3f84652c6a9c29c49ad46ab"
CASES_SHA256 = "0aaf646eb65e9112dce1d69ea802a4ce4e0487c18d705368c29ac714767f1df0"
REGENERATION_ABSOLUTE_TOLERANCE = 5.0e-14


def _compare_regenerated(actual: object, expected: object, path: str = "root") -> None:
    """Compare independently regenerated floats without demanding bit identity."""
    assert type(actual) is type(expected), path
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), path
        for key in actual:
            _compare_regenerated(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare_regenerated(actual_item, expected_item, f"{path}[{index}]")
    elif isinstance(actual, str) and (
        actual.startswith(("0x", "-0x")) and expected.startswith(("0x", "-0x"))
    ):
        assert math.isclose(
            float.fromhex(actual),
            float.fromhex(expected),
            rel_tol=0.0,
            abs_tol=REGENERATION_ABSOLUTE_TOLERANCE,
        ), path
    elif isinstance(actual, float):
        assert math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=REGENERATION_ABSOLUTE_TOLERANCE,
        ), path
    else:
        assert actual == expected, path


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
        "contract.json": CONTRACT_SHA256,
        "oracle_cases.json": CASES_SHA256,
    }


def _verify_frozen_sources(contract: dict) -> None:
    for relative, expected in contract["source_sha256"].items():
        content = subprocess.check_output(
            ("git", "show", f"{SOURCE_COMMIT}:{relative}"), cwd=ROOT
        )
        assert hashlib.sha256(content).hexdigest() == expected


def _verify_infrastructure_classifier() -> None:
    assert (
        classify_device_admission(
            process_visible=True, host_inventory="present", admission="passed"
        )
        == "admitted"
    )
    assert (
        classify_device_admission(
            process_visible=True, host_inventory="present", admission="failed"
        )
        == "visible_admission_failed"
    )
    assert (
        classify_device_admission(process_visible=True, host_inventory="unknown")
        == "visible_unqualified"
    )
    assert (
        classify_device_admission(process_visible=False, host_inventory="present")
        == "process_hidden_host_present"
    )
    assert (
        classify_device_admission(process_visible=False, host_inventory="absent")
        == "host_confirmed_absent"
    )
    assert (
        classify_device_admission(process_visible=False, host_inventory="unknown")
        == "not_visible_host_unknown"
    )
    assert (
        classify_device_admission(process_visible=None, host_inventory="unknown")
        == "probe_failed"
    )


def main() -> None:
    _verify_oracle_imports()
    _verify_manifest()
    _verify_infrastructure_classifier()
    contract = json.loads((FROZEN / "contract.json").read_text())
    cases = json.loads((FROZEN / "oracle_cases.json").read_text())
    assert contract["schema"] == "gradflow.fd_fv.nonlinear.phase_5a.contract.v1"
    assert contract["performance_measurements_collected"] is False
    assert contract["production_burgers_implementation_added"] is False
    _verify_frozen_sources(contract)
    assert cases["summary"]["all_oracle_checks_passed"] is True
    assert float.fromhex(cases["analytic"]["minimum_characteristic_jacobian_hex"]) > 0.0

    with tempfile.TemporaryDirectory() as directory:
        regenerated = Path(directory) / "oracle_cases.json"
        regenerated_payload = build_cases()
        regenerated.write_text(
            json.dumps(regenerated_payload, indent=2, sort_keys=True) + "\n"
        )
        _compare_regenerated(regenerated_payload, cases)

    print(
        "FD/FV nonlinear Phase 5A verified: exact pre-shock point and "
        "cell-average oracles passed, no performance was collected, and "
        "device visibility vocabulary is unambiguous."
    )


if __name__ == "__main__":
    main()
