#!/usr/bin/env python3
"""Execute one full process-entry-to-answer Phase-4B cold point."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource

import torch

from problem import FINAL_TIME, METHOD_IDS, conservation, errors, projected_state, solve, step_function


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("fd", "fv"), required=True)
    parser.add_argument("--dimension", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    torch.set_num_threads(6)
    torch.set_num_interop_threads(1)
    initial = projected_state(arguments.method, arguments.dimension, arguments.cells)
    expected = projected_state(
        arguments.method,
        arguments.dimension,
        arguments.cells,
        time=FINAL_TIME,
    )
    step, steps = step_function(
        arguments.method, arguments.dimension, arguments.cells
    )
    compiled = torch.compile(step, fullgraph=True, dynamic=False)
    final = solve(initial, compiled, steps)
    l1, l2 = errors(final, expected)
    mass_change, mass_bound, mass_passed = conservation(
        initial, final, arguments.dimension, arguments.cells
    )
    result = {
        "status": "completed",
        "method": arguments.method,
        "formulation_id": METHOD_IDS[arguments.method],
        "dimension": arguments.dimension,
        "cells_per_axis": arguments.cells,
        "logical_cells": arguments.cells**arguments.dimension,
        "steps": steps,
        "l1_error": l1,
        "l2_error": l2,
        "finite": math.isfinite(l1) and math.isfinite(l2),
        "mass_change": mass_change,
        "mass_bound": mass_bound,
        "conservation_passed": mass_passed,
        "host_visible_answer": True,
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "eligible": math.isfinite(l1) and math.isfinite(l2) and mass_passed,
    }
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
