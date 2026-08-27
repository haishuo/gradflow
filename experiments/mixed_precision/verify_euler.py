#!/usr/bin/env python3
"""Independently verify the Phase-D Tier-2 Euler qualification record."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ORDERS = (5, 7, 9, 11, 13, 15)
REPRESENTATIVE_ORDERS = (5, 11, 15)
POLICIES = (
    "all_f64",
    "indicators_f32",
    "weight_formation_f32",
    "indicators_and_weight_formation_f32",
)
LOCAL_THRESHOLDS = {
    "tight": {"linf": 1.0e-5, "rms": 1.0e-6},
    "engineering": {"linf": 5.0e-4, "rms": 1.0e-4},
}
TERMINAL_THRESHOLDS = {
    "tight": {"l1": 1.0e-4, "linf": 2.0e-3},
    "engineering": {"l1": 5.0e-4, "linf": 1.0e-2},
}
PHASE_A_RECORD = ROOT / "experiments/euler_boundary_shock/results/phase_a_20260827"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_checksums(directory: Path) -> None:
    checksum_path = directory / "SHA256SUMS"
    entries = checksum_path.read_text().splitlines()
    expected_names = {path.name for path in directory.iterdir() if path != checksum_path}
    recorded_names: set[str] = set()
    for line in entries:
        expected, name = line.split()
        if name in recorded_names:
            raise ValueError(f"duplicate checksum entry: {name}")
        recorded_names.add(name)
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"checksum mismatch: {name}")
    if recorded_names != expected_names:
        raise ValueError("checksum manifest does not cover exactly the result files")


def expected_keys(orders: tuple[int, ...]) -> set[str]:
    return {f"order{order}_{policy}" for order in orders for policy in POLICIES}


def inherited_class(policy: str, order: int) -> str:
    if policy in {"indicators_f32", "indicators_and_weight_formation_f32"} and order == 5:
        return "engineering"
    return "tight"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_numbers(values: Any) -> bool:
    if isinstance(values, bool) or values is None or isinstance(values, str):
        return True
    if isinstance(values, (int, float)):
        return math.isfinite(values)
    if isinstance(values, dict):
        return all(finite_numbers(value) for value in values.values())
    if isinstance(values, list):
        return all(finite_numbers(value) for value in values)
    return False


def verify_identity(key: str, record: dict[str, Any], *, problem: str | None = None) -> None:
    prefix = f"{problem}_" if problem else ""
    expected = f"{prefix}order{record['order']}_{record['policy']}"
    require(key == expected, f"record identity mismatch: {key}")
    require(record["policy"] in POLICIES, f"unknown policy: {key}")
    if "inherited_class" in record:
        require(
            record["inherited_class"] == inherited_class(record["policy"], record["order"]),
            f"inherited class mismatch: {key}",
        )


def verify_local(key: str, record: dict[str, Any]) -> bool:
    verify_identity(key, record)
    threshold = LOCAL_THRESHOLDS[record["inherited_class"]]
    expected = (
        all(
            case["finite"]
            and finite_numbers(case)
            and case["maximum_linf_normalized"] <= threshold["linf"]
            and case["maximum_rms_normalized"] <= threshold["rms"]
            for case in record["parity_cases"].values()
        )
        and math.isfinite(record["uniform_rhs_linf"])
        and record["uniform_rhs_linf"] <= 2.0e-12
        and all(math.isfinite(value) and value <= 64.0 for value in record["conservation_ratios"].values())
    )
    require(record["passed"] is expected, f"local decision mismatch: {key}")
    return expected


def verify_repeated(key: str, record: dict[str, Any]) -> bool:
    verify_identity(key, record)
    threshold = TERMINAL_THRESHOLDS[record["inherited_class"]]
    terminal = record["terminal_parity"]
    analytic_ok = (
        len(record["analytic_l1"]) == len(record["analytic_bound"])
        and all(
            actual <= bound
            for actual, bound in zip(
                record["analytic_l1"], record["analytic_bound"], strict=True
            )
        )
    )
    expected = (
        finite_numbers(record)
        and record["completed"]
        and record["minimum_density"] > 0.0
        and record["minimum_pressure"] > 0.0
        and terminal["maximum_l1_normalized"] <= threshold["l1"]
        and terminal["maximum_linf_normalized"] <= threshold["linf"]
        and analytic_ok
    )
    require(record["passed"] is expected, f"repeated-step decision mismatch: {key}")
    return expected


def verify_gradient(key: str, record: dict[str, Any]) -> bool:
    verify_identity(key, record)
    expected = (
        finite_numbers(record)
        and record["finite"]
        and record["nonzero"]
        and (
            record["relative_error"] <= 2.0e-5
            or record["absolute_error"] <= 2.0e-7
        )
        and record["gradient_l2_normalized"] <= 5.0e-4
        and record["gradient_linf_normalized"] <= 2.0e-3
    )
    require(record["passed"] is expected, f"gradient decision mismatch: {key}")
    return expected


def verify_compiler_device(key: str, record: dict[str, Any]) -> bool:
    verify_identity(key, record)
    require(
        set(record["compiled"]) == {"cpu", "cuda"},
        f"compiler device set mismatch: {key}",
    )
    compiled_passes = []
    for device, result in record["compiled"].items():
        if not result["available"]:
            require(device == "cuda", f"CPU compile result unavailable: {key}")
            continue
        parity = result["parity"]
        expected_compile = (
            finite_numbers(parity)
            and result["graph_count"] == 1
            and result["graph_break_count"] == 0
            and parity["maximum_linf_normalized"] <= 5.0e-5
            and parity["maximum_rms_normalized"] <= 1.0e-5
        )
        require(
            result["passed"] is expected_compile,
            f"compile decision mismatch: {key}/{device}",
        )
        compiled_passes.append(expected_compile)
    metric = record["cpu_cuda"]
    expected_device = metric is None or (
        finite_numbers(metric)
        and metric["maximum_linf_normalized"] <= 5.0e-4
    )
    require(
        record["cpu_cuda_passed"] is expected_device,
        f"device decision mismatch: {key}",
    )
    expected = expected_device and all(compiled_passes)
    require(
        record["passed"] is expected,
        f"compiler/device decision mismatch: {key}",
    )
    return expected


def independent_shock_passed(
    record: dict[str, Any], thresholds: dict[str, Any]
) -> bool:
    metrics = record["independent_metrics"]
    if record["problem"] == "sod":
        return all(
            metrics["l1"][name] <= thresholds["sod"]["l1_max"][name]
            for name in ("density", "velocity", "pressure")
        ) and all(
            wave["error_cells"]
            <= thresholds["sod"]["wave_location_error_cells_max"]
            for wave in metrics["wave_locations"].values()
        )
    structure = metrics["structure"]
    return all(
        metrics["l1"][name]
        <= thresholds["shu_osher"]["l1_max_to_n12800"][name]
        for name in ("density", "velocity", "pressure")
    ) and (
        structure["density_correlation"]
        >= thresholds["shu_osher"]["density_correlation_min"]
        and thresholds["shu_osher"]["density_total_variation_ratio_min"]
        <= structure["density_total_variation_ratio"]
        <= thresholds["shu_osher"]["density_total_variation_ratio_max"]
    )


def verify_shock(
    key: str, record: dict[str, Any], thresholds: dict[str, Any]
) -> bool:
    verify_identity(key, record, problem=record["problem"])
    if record["policy"] == "all_f64":
        control = ROOT / record["control_artifact"]
        expected = (
            control.is_file()
            and sha256(control) == record["control_sha256"]
            and record["completed"]
            and record["independent_passed"]
            and record["terminal_parity"]["maximum_l1_normalized"] == 0.0
            and record["terminal_parity"]["maximum_linf_normalized"] == 0.0
        )
    else:
        threshold = TERMINAL_THRESHOLDS[record["inherited_class"]]
        independent = independent_shock_passed(record, thresholds)
        require(
            record["independent_passed"] is independent,
            f"independent shock decision mismatch: {key}",
        )
        terminal = record["terminal_parity"]
        expected = (
            finite_numbers(record)
            and record["completed"]
            and record["minimum_density"] > 0.0
            and record["minimum_pressure"] > 0.0
            and terminal["maximum_l1_normalized"] <= threshold["l1"]
            and terminal["maximum_linf_normalized"] <= threshold["linf"]
            and independent
        )
    require(record["passed"] is expected, f"shock decision mismatch: {key}")
    return expected


def verify(directory: Path) -> dict[str, Any]:
    verify_checksums(directory)
    payload = json.loads((directory / "qualification.json").read_text())
    require(payload["schema_version"] == 1, "unknown Tier-2 schema")
    require(tuple(payload["orders"]) == ORDERS, "order list mismatch")
    require(
        tuple(payload["representative_orders"]) == REPRESENTATIVE_ORDERS,
        "representative order list mismatch",
    )
    require(tuple(payload["policies"]) == POLICIES, "policy list mismatch")
    require(payload["local_thresholds"] == LOCAL_THRESHOLDS, "local thresholds mismatch")
    require(
        payload["terminal_thresholds"] == TERMINAL_THRESHOLDS,
        "terminal thresholds mismatch",
    )

    threshold_path = PHASE_A_RECORD / "thresholds.json"
    require(
        sha256(threshold_path) == payload["phase_a_thresholds_sha256"],
        "Phase-A threshold hash mismatch",
    )
    thresholds = json.loads(threshold_path.read_text())
    for name, digest in payload["source_hashes"].items():
        path = ROOT / name
        require(
            path.is_file() and sha256(path) == digest,
            f"source hash mismatch: {name}",
        )

    require(
        set(payload["local"]) == expected_keys(ORDERS),
        "local matrix is incomplete",
    )
    for section in ("repeated_step", "gradients", "compiler_device"):
        require(
            set(payload[section]) == expected_keys(REPRESENTATIVE_ORDERS),
            f"{section} matrix is incomplete",
        )
    shock_keys = {
        f"{problem}_order{order}_{policy}"
        for problem in ("sod", "shu_osher")
        for order in REPRESENTATIVE_ORDERS
        for policy in POLICIES
    }
    require(set(payload["shocks"]) == shock_keys, "shock matrix is incomplete")

    decisions = []
    decisions.extend(
        verify_local(key, record) for key, record in payload["local"].items()
    )
    decisions.extend(
        verify_repeated(key, record)
        for key, record in payload["repeated_step"].items()
    )
    decisions.extend(
        verify_gradient(key, record)
        for key, record in payload["gradients"].items()
    )
    decisions.extend(
        verify_compiler_device(key, record)
        for key, record in payload["compiler_device"].items()
    )
    decisions.extend(
        verify_shock(key, record, thresholds)
        for key, record in payload["shocks"].items()
    )
    require(
        payload["static"]["passed"]
        is (not payload["static"]["forbidden_tokens_found"]),
        "static decision mismatch",
    )
    expected_decision = (
        "PASS" if all(decisions) and payload["static"]["passed"] else "FAIL"
    )
    require(payload["decision"] == expected_decision, "top-level decision mismatch")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    payload = verify(args.result_dir.resolve())
    print(
        f"verified Tier 2 decision={payload['decision']} "
        f"source={payload['source_commit']}"
    )


if __name__ == "__main__":
    main()
