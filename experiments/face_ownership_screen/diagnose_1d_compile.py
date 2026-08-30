#!/usr/bin/env python3
"""Timing-free prospective diagnostic for the large-1D compile failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch

from run_screen import (
    comparison_passes,
    error_metrics,
    make_rhs,
    output_health,
    smooth_input,
    threshold,
)


SIZES = (65_536, 262_144, 524_288, 786_432, 884_736, 1_048_576)


def compile_probe(
    function: Callable[[torch.Tensor], torch.Tensor], state: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(function, fullgraph=True)
    started = time.perf_counter()
    output = compiled(state)
    torch.cuda.synchronize()
    stats = torch._dynamo.utils.counters.get("stats", {})
    breaks = torch._dynamo.utils.counters.get("graph_break", {})
    return output, {
        "first_call_wall_seconds": time.perf_counter() - started,
        "unique_graphs": int(stats.get("unique_graphs", 0)),
        "calls_captured": int(stats.get("calls_captured", 0)),
        "graph_break_count": int(sum(breaks.values())),
        "graph_break_reasons": {str(key): int(value) for key, value in breaks.items()},
    }


def scalar_record(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_value = float(actual)
    reference_value = float(reference)
    return {
        "eager": reference_value,
        "compiled": actual_value,
        "absolute_error": abs(actual_value - reference_value),
        "bitwise_identical": bool(torch.equal(actual, reference)),
    }


def run_case(order: int, dtype: torch.dtype, n: int) -> dict[str, Any]:
    state = smooth_input(n, 1, dtype).cuda()
    dx = 2.0 * torch.pi / n
    eager_alpha = torch.amax(torch.abs(state))
    compiled_alpha, alpha_compile = compile_probe(
        lambda value: torch.amax(torch.abs(value)), state
    )

    eager: dict[str, torch.Tensor] = {}
    compiled: dict[str, torch.Tensor] = {}
    compilation: dict[str, Any] = {}
    own_comparisons: dict[str, Any] = {}
    health: dict[str, Any] = {}
    bounds = threshold(dtype, compiled=True)

    for representation in ("face_once", "cell_recompute"):
        function = make_rhs(representation, order, 1, dx)
        eager[representation] = function(state)
        compiled[representation], compilation[representation] = compile_probe(
            function, state
        )
        metrics = error_metrics(compiled[representation], eager[representation])
        own_comparisons[representation] = {
            **metrics,
            "bounds": {"maximum_normalized": bounds[0], "rms_normalized": bounds[1]},
            "passed": comparison_passes(metrics, bounds),
        }
        health[representation] = {
            "eager": output_health(eager[representation]),
            "compiled": output_health(compiled[representation]),
        }

    parity = error_metrics(compiled["cell_recompute"], compiled["face_once"])
    del state, eager, compiled
    torch.cuda.empty_cache()
    return {
        "order": order,
        "dtype": str(dtype).removeprefix("torch."),
        "n": n,
        "alpha": {**scalar_record(compiled_alpha, eager_alpha), "compilation": alpha_compile},
        "compilation": compilation,
        "compiled_versus_eager": own_comparisons,
        "compiled_representation_parity": parity,
        "health": health,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    document: dict[str, Any] = {
        "schema": "gradflow.face_ownership_1d_compile_diagnostic.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "records": [],
        "complete": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for order in (5, 15):
        for dtype in (torch.float32, torch.float64):
            for n in SIZES:
                record = run_case(order, dtype, n)
                document["records"].append(record)
                args.output.write_text(json.dumps(document, indent=2) + "\n")
                face = record["compiled_versus_eager"]["face_once"]
                cell = record["compiled_versus_eager"]["cell_recompute"]
                print(
                    f"order={order} dtype={dtype} N={n}: "
                    f"alpha_err={record['alpha']['absolute_error']:.3e} "
                    f"face={face['maximum_normalized']:.3e}/{face['passed']} "
                    f"cell={cell['maximum_normalized']:.3e}/{cell['passed']}",
                    flush=True,
                )
    document["complete"] = True
    document["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
