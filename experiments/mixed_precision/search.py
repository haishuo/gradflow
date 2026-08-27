#!/usr/bin/env python3
"""Execute the frozen Phase-D scalar mixed-precision search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from gradflow import (  # noqa: E402
    PRECISION_BLOCKS,
    QUALIFIED_ORDERS,
    WENOJS,
    WENOJSPrecisionPolicy,
)

CLASS_THRESHOLDS = {
    "tight": {"linf": 1.0e-5, "rms": 1.0e-6},
    "engineering": {"linf": 5.0e-4, "rms": 1.0e-4},
    "coarse": {"linf": 1.0e-2, "rms": 2.0e-3},
}
CONSTANT_BOUND = 5.0e-5
CONSERVATION_FACTOR = 64.0
SEARCH_SEED = 20260827


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def policy_for_mask(mask: int) -> WENOJSPrecisionPolicy:
    if not 0 <= mask < 2 ** len(PRECISION_BLOCKS):
        raise ValueError("precision mask is outside the exhaustive search")
    choices = {
        block: torch.float32 if mask & (1 << index) else torch.float64
        for index, block in enumerate(PRECISION_BLOCKS)
    }
    return WENOJSPrecisionPolicy(**choices)


def policy_id(mask: int) -> str:
    bits = format(mask, f"0{len(PRECISION_BLOCKS)}b")[::-1]
    return "".join(
        f"{block[0]}{32 if bit == '1' else 64}"
        for block, bit in zip(PRECISION_BLOCKS, bits)
    )


def parity_metrics(
    actual: torch.Tensor,
    oracle: torch.Tensor,
    *,
    signal_scale: float,
) -> dict[str, float | bool]:
    difference = actual - oracle
    scale = max(abs(signal_scale), torch.finfo(torch.float64).tiny)
    return {
        "finite": bool(torch.all(torch.isfinite(actual))),
        "linf_absolute": float(torch.max(torch.abs(difference))),
        "rms_absolute": float(torch.sqrt(torch.mean(difference.square()))),
        "linf_normalized": float(torch.max(torch.abs(difference))) / scale,
        "rms_normalized": float(torch.sqrt(torch.mean(difference.square()))) / scale,
        "signal_scale": scale,
    }


def rates(errors: list[float], sizes: tuple[int, ...]) -> list[float]:
    return [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors, errors[1:], sizes, sizes[1:]
        )
        if coarse > 0.0 and fine > 0.0
    ]


def rk3(
    scheme: WENOJS,
    state: torch.Tensor,
    dx: float,
    dt: float,
    steps: int,
) -> torch.Tensor:
    result = state
    for _ in range(steps):
        rhs0 = scheme.rhs(result, dx, lambda q: q, alpha=1.0)
        stage1 = result + dt * rhs0
        rhs1 = scheme.rhs(stage1, dx, lambda q: q, alpha=1.0)
        stage2 = 0.75 * result + 0.25 * (stage1 + dt * rhs1)
        rhs2 = scheme.rhs(stage2, dx, lambda q: q, alpha=1.0)
        result = (result + 2.0 * (stage2 + dt * rhs2)) / 3.0
    return result


def build_oracles(order: int) -> dict[str, Any]:
    scheme = WENOJS(order)
    oracles: dict[str, Any] = {}

    smooth_sizes = (48, 96, 192)
    smooth = []
    for n in smooth_sizes:
        x = torch.arange(n, dtype=torch.float64) / n
        state = torch.sin(6.0 * math.pi * x) + 0.15 * torch.cos(8.0 * math.pi * x)
        exact = -(
            6.0 * math.pi * torch.cos(6.0 * math.pi * x)
            - 1.2 * math.pi * torch.sin(8.0 * math.pi * x)
        )
        rhs = scheme.rhs(state, 1.0 / n, lambda q: q, alpha=1.0)
        smooth.append((state, exact, rhs))
    oracles["smooth"] = (smooth_sizes, smooth)

    critical_sizes = (64, 128, 256)
    critical = []
    for n in critical_sizes:
        x = torch.arange(n, dtype=torch.float64) / n
        sine = torch.sin(2.0 * math.pi * x)
        state = sine.pow(3)
        exact = -6.0 * math.pi * sine.square() * torch.cos(2.0 * math.pi * x)
        rhs = scheme.rhs(state, 1.0 / n, lambda q: q, alpha=1.0)
        critical.append((state, exact, rhs))
    oracles["critical"] = (critical_sizes, critical)

    near_constant = []
    n = 256
    x = torch.arange(n, dtype=torch.float64) / n
    for amplitude in (1.0e-4, 1.0e-6, 1.0e-7):
        state = 1.0 + amplitude * torch.sin(2.0 * math.pi * x)
        rhs = scheme.rhs(state, 1.0 / n, lambda q: q, alpha=1.0)
        near_constant.append((amplitude, state, rhs))
    oracles["near_constant"] = near_constant

    n = 257
    x = torch.arange(n, dtype=torch.float64) / n
    square = torch.where(x < 0.5, torch.ones_like(x), -0.25 * torch.ones_like(x))
    oracles["square"] = (
        square,
        scheme.rhs(square, 1.0 / n, lambda q: q, alpha=1.0),
    )

    generator = torch.Generator().manual_seed(SEARCH_SEED)
    random_state = 0.8 * torch.randn(n, generator=generator, dtype=torch.float64)
    random_rhs = scheme.rhs(
        random_state,
        1.0 / n,
        lambda q: 0.5 * q.square(),
        alpha=2.5,
    )
    oracles["burgers"] = (random_state, random_rhs)

    n = 128
    x = torch.arange(n, dtype=torch.float64) / n
    initial = 0.3 + torch.sin(2.0 * math.pi * x) + 0.1 * torch.cos(6.0 * math.pi * x)
    oracles["rk3"] = (
        initial,
        rk3(scheme, initial, 1.0 / n, 0.2 / n, 40),
    )
    return oracles


def evaluate(order: int, mask: int, oracles: dict[str, Any]) -> dict[str, Any]:
    policy = policy_for_mask(mask)
    scheme = WENOJS(order, precision=policy)
    cases: dict[str, dict[str, float | bool]] = {}
    analytic: dict[str, Any] = {}

    smooth_sizes, smooth_data = oracles["smooth"]
    actual_errors = []
    oracle_errors = []
    for n, (state, exact, oracle) in zip(smooth_sizes, smooth_data):
        actual = scheme.rhs(state, 1.0 / n, lambda q: q, alpha=1.0)
        name = f"smooth_n{n}"
        scale = float(torch.max(torch.abs(exact)))
        cases[name] = parity_metrics(actual, oracle, signal_scale=scale)
        actual_errors.append(float(torch.sqrt(torch.mean((actual - exact).square()))))
        oracle_errors.append(float(torch.sqrt(torch.mean((oracle - exact).square()))))
    analytic["smooth"] = {
        "sizes": smooth_sizes,
        "actual_l2_errors": actual_errors,
        "oracle_l2_errors": oracle_errors,
        "actual_rates": rates(actual_errors, smooth_sizes),
        "oracle_rates": rates(oracle_errors, smooth_sizes),
    }

    critical_sizes, critical_data = oracles["critical"]
    actual_errors = []
    oracle_errors = []
    for n, (state, exact, oracle) in zip(critical_sizes, critical_data):
        actual = scheme.rhs(state, 1.0 / n, lambda q: q, alpha=1.0)
        name = f"critical_n{n}"
        scale = float(torch.max(torch.abs(exact)))
        cases[name] = parity_metrics(actual, oracle, signal_scale=scale)
        actual_errors.append(float(torch.sqrt(torch.mean((actual - exact).square()))))
        oracle_errors.append(float(torch.sqrt(torch.mean((oracle - exact).square()))))
    analytic["critical"] = {
        "sizes": critical_sizes,
        "actual_l2_errors": actual_errors,
        "oracle_l2_errors": oracle_errors,
        "actual_rates": rates(actual_errors, critical_sizes),
        "oracle_rates": rates(oracle_errors, critical_sizes),
    }

    for amplitude, state, oracle in oracles["near_constant"]:
        actual = scheme.rhs(state, 1.0 / 256, lambda q: q, alpha=1.0)
        name = f"near_constant_a{amplitude:.0e}"
        cases[name] = parity_metrics(
            actual, oracle, signal_scale=2.0 * math.pi * amplitude
        )

    square, square_oracle = oracles["square"]
    square_actual = scheme.rhs(square, 1.0 / 257, lambda q: q, alpha=1.0)
    cases["square"] = parity_metrics(
        square_actual,
        square_oracle,
        signal_scale=float(torch.max(torch.abs(square_oracle))),
    )

    random_state, random_oracle = oracles["burgers"]
    random_actual = scheme.rhs(
        random_state,
        1.0 / 257,
        lambda q: 0.5 * q.square(),
        alpha=2.5,
    )
    cases["burgers"] = parity_metrics(
        random_actual,
        random_oracle,
        signal_scale=float(torch.max(torch.abs(random_oracle))),
    )

    initial, rk_oracle = oracles["rk3"]
    rk_actual = rk3(scheme, initial, 1.0 / 128, 0.2 / 128, 40)
    cases["rk3_40_steps"] = parity_metrics(
        rk_actual,
        rk_oracle,
        signal_scale=float(torch.max(torch.abs(rk_oracle))),
    )

    constant = torch.full(
        (max(32, order + 2),), 1.25, dtype=torch.float64
    )
    constant_rhs = scheme.rhs(
        constant, 1.0 / constant.shape[-1], lambda q: q, alpha=1.0
    )
    constant_error = float(torch.max(torch.abs(constant_rhs)))
    conservation_residual = float(torch.abs(torch.sum(random_actual, dtype=torch.float64)))
    conservation_scale = float(torch.sum(torch.abs(random_actual), dtype=torch.float64))
    conservation_bound = (
        CONSERVATION_FACTOR * torch.finfo(torch.float32).eps * conservation_scale
    )

    maximum_linf = max(float(case["linf_normalized"]) for case in cases.values())
    maximum_rms = max(float(case["rms_normalized"]) for case in cases.values())
    finite = all(bool(case["finite"]) for case in cases.values()) and bool(
        torch.all(torch.isfinite(constant_rhs))
    )
    safety = {
        "finite": finite,
        "constant_rhs_linf": constant_error,
        "constant_bound": CONSTANT_BOUND,
        "conservation_residual": conservation_residual,
        "conservation_bound": conservation_bound,
        "passed": (
            finite
            and constant_error <= CONSTANT_BOUND
            and conservation_residual <= conservation_bound
        ),
    }

    classification = "failed"
    if safety["passed"]:
        for name, threshold in CLASS_THRESHOLDS.items():
            if maximum_linf <= threshold["linf"] and maximum_rms <= threshold["rms"]:
                classification = name
                break

    return {
        "order": order,
        "mask": mask,
        "policy_id": policy_id(mask),
        "assignment": policy.as_names(),
        "float32_blocks": sum(
            dtype == "float32" for dtype in policy.as_names().values()
        ),
        "classification": classification,
        "maximum_linf_normalized": maximum_linf,
        "maximum_rms_normalized": maximum_rms,
        "safety": safety,
        "cases": cases,
        "analytic": analytic,
    }


def summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for record in records:
        name = str(record["classification"])
        counts[name] = counts.get(name, 0) + 1
    per_order = {}
    for order in sorted({int(record["order"]) for record in records}):
        selected = [record for record in records if record["order"] == order]
        order_counts: dict[str, int] = {}
        for record in selected:
            name = str(record["classification"])
            order_counts[name] = order_counts.get(name, 0) + 1
        per_order[str(order)] = {
            "counts": order_counts,
            "maximum_float32_blocks_by_class": {
                class_name: max(
                    (
                        int(record["float32_blocks"])
                        for record in selected
                        if record["classification"] == class_name
                    ),
                    default=None,
                )
                for class_name in (*CLASS_THRESHOLDS, "failed")
            },
            "most_demoted_tight_policies": [
                record["policy_id"]
                for record in selected
                if record["classification"] == "tight"
                and record["float32_blocks"]
                == max(
                    (
                        int(candidate["float32_blocks"])
                        for candidate in selected
                        if candidate["classification"] == "tight"
                    ),
                    default=-1,
                )
            ],
        }
    return {"counts": counts, "per_order": per_order}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--orders", type=int, nargs="+", default=list(QUALIFIED_ORDERS)
    )
    parser.add_argument(
        "--masks",
        type=int,
        nargs="+",
        default=list(range(2 ** len(PRECISION_BLOCKS))),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output directory: {args.output}")
    if any(order not in QUALIFIED_ORDERS for order in args.orders):
        raise SystemExit(f"orders must be drawn from {QUALIFIED_ORDERS}")
    if any(not 0 <= mask < 2 ** len(PRECISION_BLOCKS) for mask in args.masks):
        raise SystemExit("masks must be in the exhaustive 0--63 range")

    torch.set_num_threads(1)
    records = []
    for order in args.orders:
        oracles = build_oracles(order)
        for mask in args.masks:
            records.append(evaluate(order, mask, oracles))

    complete = (
        tuple(args.orders) == QUALIFIED_ORDERS
        and args.masks == list(range(2 ** len(PRECISION_BLOCKS)))
    )
    dirty = bool(git_text("status", "--porcelain"))
    payload = {
        "schema_version": 1,
        "phase": "D-tier-1-scalar-mixed-precision",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/MIXED_PRECISION_PHASE_D_PROTOCOL.md",
        "source_commit": git_text("rev-parse", "HEAD"),
        "source_dirty": dirty,
        "command": " ".join(sys.argv),
        "complete_frozen_matrix": complete,
        "precision_blocks": PRECISION_BLOCKS,
        "class_thresholds": CLASS_THRESHOLDS,
        "class_order": list(CLASS_THRESHOLDS),
        "constant_bound": CONSTANT_BOUND,
        "conservation_factor": CONSERVATION_FACTOR,
        "seed": SEARCH_SEED,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "summary": summary(records),
        "records": records,
    }
    args.output.mkdir(parents=True)
    result_path = args.output / "search.json"
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    checksum_path = args.output / "SHA256SUMS"
    checksum_path.write_text(f"{sha256(result_path)}  {result_path.name}\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
