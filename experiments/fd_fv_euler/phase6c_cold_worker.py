#!/usr/bin/env python3
"""Run one Phase-6C process-entry-to-host-answer smooth cold pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from experiments.fd_fv_euler.phase6c_problem import (
    FINAL_SMOOTH_TIME,
    METHOD_IDS,
    adaptive_solve,
    conservation,
    error_norms,
    smooth_expected,
    smooth_initial,
    stage_function,
    tensor_hash,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=("eager", "compiled"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    try:
        if arguments.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible to the worker")
        initial_cpu = smooth_initial(arguments.method, arguments.cells)
        initial = initial_cpu.to(arguments.device)
        expected = smooth_expected(arguments.method, arguments.cells).to(
            arguments.device
        )
        stages = stage_function(arguments.method, arguments.cells, "periodic")
        if arguments.mode == "compiled":
            stages = torch.compile(stages, fullgraph=True, dynamic=False)
        final, diagnostics = adaptive_solve(
            arguments.method,
            initial,
            FINAL_SMOOTH_TIME,
            "periodic",
            stages,
            check_stages=False,
        )
        final_cpu = final.cpu()
        expected_cpu = expected.cpu()
        checks = {
            **error_norms(final_cpu, expected_cpu),
            "conservation": conservation(
                initial_cpu,
                final_cpu,
                1.0 / arguments.cells,
                diagnostics["steps"],
            ),
            "finite": bool(torch.isfinite(final_cpu).all()),
            "host_visible_answer": final_cpu.device.type == "cpu",
            "terminal_sha256": tensor_hash(final_cpu),
        }
        eligible = (
            diagnostics["completed"]
            and checks["finite"]
            and checks["host_visible_answer"]
            and checks["conservation"]["passed"]
        )
        result = {
            "status": "completed",
            "kind": "cold",
            "method": arguments.method,
            "formulation_id": METHOD_IDS[arguments.method],
            "device": arguments.device,
            "mode": arguments.mode,
            "cells": arguments.cells,
            "diagnostics": diagnostics,
            "checks": checks,
            "peak_process_rss_bytes": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "eligible": eligible,
        }
    except Exception as error:
        result = {
            "status": "failed",
            "kind": "cold",
            "method": arguments.method,
            "device": arguments.device,
            "mode": arguments.mode,
            "cells": arguments.cells,
            "error_type": type(error).__name__,
            "error": str(error),
            "eligible": False,
        }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
