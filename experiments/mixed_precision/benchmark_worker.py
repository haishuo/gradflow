#!/usr/bin/env python3
"""Measure one frozen CUDA order/policy pair in an isolated process."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from gradflow import PRECISION_BLOCKS, WENOJS, WENOJSPrecisionPolicy  # noqa: E402

POINTS = 2**20
WARMUPS = 5
REPETITIONS = 30
POLICY_MASKS = {
    "all_f64": 0,
    "indicators_f32": 1 << PRECISION_BLOCKS.index("indicators"),
    "weight_formation_f32": 1 << PRECISION_BLOCKS.index("weight_formation"),
    "indicators_and_weight_formation_f32": (
        (1 << PRECISION_BLOCKS.index("indicators"))
        | (1 << PRECISION_BLOCKS.index("weight_formation"))
    ),
    "all_internal_f32": 2 ** len(PRECISION_BLOCKS) - 1,
}


def policy_for_name(name: str) -> WENOJSPrecisionPolicy:
    try:
        mask = POLICY_MASKS[name]
    except KeyError as error:
        raise ValueError(f"unknown frozen benchmark policy: {name}") from error
    return WENOJSPrecisionPolicy(
        **{
            block: torch.float32 if mask & (1 << index) else torch.float64
            for index, block in enumerate(PRECISION_BLOCKS)
        }
    )


def quantile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (1.0 - weight) * sorted_values[lower] + weight * sorted_values[upper]


def statistics(milliseconds: list[float]) -> dict[str, Any]:
    ordered = sorted(milliseconds)
    return {
        "samples_ms": milliseconds,
        "median_ms": quantile(ordered, 0.5),
        "q1_ms": quantile(ordered, 0.25),
        "q3_ms": quantile(ordered, 0.75),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "mean_ms": sum(ordered) / len(ordered),
    }


def measure_calls(call: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor):
    output = None
    for _ in range(WARMUPS):
        output = call(state)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(REPETITIONS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(REPETITIONS)]
    for start, end in zip(starts, ends):
        start.record()
        output = call(state)
        end.record()
    torch.cuda.synchronize()
    elapsed = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    assert output is not None
    result = statistics(elapsed)
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["output_finite"] = bool(torch.all(torch.isfinite(output)).item())
    result["output_max_abs"] = float(torch.max(torch.abs(output)).item())
    return result


def execute(order: int, policy_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    policy = policy_for_name(policy_name)
    scheme = WENOJS(order, precision=policy)
    x = torch.arange(POINTS, dtype=torch.float64, device="cuda") / POINTS
    state = 0.3 + 0.6 * torch.sin(2.0 * math.pi * x) + 0.1 * torch.cos(
        6.0 * math.pi * x
    )
    dx = 1.0 / POINTS

    def call(values: torch.Tensor) -> torch.Tensor:
        return scheme.rhs(
            values,
            dx,
            lambda q: 0.5 * q.square(),
            alpha=1.5,
        )

    eager = measure_calls(call, state)
    torch._dynamo.reset()
    compiled_call = torch.compile(call, fullgraph=True, dynamic=False)
    torch.cuda.synchronize()
    started = time.perf_counter()
    first_output = compiled_call(state)
    torch.cuda.synchronize()
    compile_first_call_ms = 1000.0 * (time.perf_counter() - started)
    compiled = measure_calls(compiled_call, state)
    compiled["first_call_ms"] = compile_first_call_ms
    compiled["first_output_finite"] = bool(
        torch.all(torch.isfinite(first_output)).item()
    )
    return {
        "status": "completed",
        "order": order,
        "policy": policy_name,
        "mask": POLICY_MASKS[policy_name],
        "assignment": policy.as_names(),
        "points": POINTS,
        "warmups": WARMUPS,
        "repetitions": REPETITIONS,
        "device": torch.cuda.get_device_name(0),
        "eager": eager,
        "compiled": compiled,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--policy", choices=tuple(POLICY_MASKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    try:
        payload = execute(args.order, args.policy)
    except Exception as error:  # failures are experimental data
        payload = {
            "status": "failed",
            "order": args.order,
            "policy": args.policy,
            "mask": POLICY_MASKS[args.policy],
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
