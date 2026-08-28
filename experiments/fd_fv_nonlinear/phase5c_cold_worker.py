#!/usr/bin/env python3
"""Run one Phase-5C process-entry-to-host-answer cold pilot point."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import sys


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from experiments.fd_fv_nonlinear.performance_problem import (
    FINAL_TIME,
    METHOD_IDS,
    conservation,
    errors,
    solve,
    state,
    step_function,
    timestep,
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
    initial_cpu = state(arguments.method, arguments.cells)
    initial = initial_cpu.to(arguments.device)
    expected = state(arguments.method, arguments.cells, FINAL_TIME).to(
        arguments.device
    )
    step = step_function(arguments.method, arguments.cells)
    if arguments.mode == "compiled":
        step = torch.compile(step, fullgraph=True, dynamic=False)
    _, steps = timestep(arguments.cells)
    final = solve(initial, step, steps)
    final_cpu = final.cpu()
    expected_cpu = expected.cpu()
    l1, l2 = errors(final_cpu, expected_cpu)
    mass_change, mass_bound, mass_passed = conservation(
        initial_cpu, final_cpu, arguments.cells
    )
    finite = math.isfinite(l1) and math.isfinite(l2)
    result = {
        "status": "completed",
        "kind": "cold",
        "method": arguments.method,
        "formulation_id": METHOD_IDS[arguments.method],
        "device": arguments.device,
        "mode": arguments.mode,
        "cells": arguments.cells,
        "steps": steps,
        "l1_error": l1,
        "l2_error": l2,
        "mass_change": mass_change,
        "mass_bound": mass_bound,
        "conservation_passed": mass_passed,
        "finite": finite,
        "host_visible_answer": final_cpu.device.type == "cpu",
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "eligible": finite and mass_passed and final_cpu.device.type == "cpu",
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
