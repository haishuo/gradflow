#!/usr/bin/env python3
"""Compare one frozen recovery candidate with the G2 oracle contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_frozen_u0 import (
    comparison,
    duplicate_periodic,
    forward_euler,
    load_unique,
    sha256,
    ssp_rk3,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()

    timing = json.loads(options.timing.read_text())
    size = int(timing["size"])
    steps = int(timing["steps"])

    initial_unique = load_unique(options.initial, size)
    actual = load_unique(options.actual, size)
    initial = duplicate_periodic(initial_unique)
    spacing = (10.0 / size,) * 3

    frozen_dt = float(timing["final_dt"])
    forward, forward_dt = forward_euler(
        initial, spacing, steps, first_dt=frozen_dt if steps == 1 else None
    )
    qualified, qualified_dt = ssp_rk3(initial, spacing, steps)

    record = {
        "study": "g3_recovery_candidate",
        "contract": timing["contract"],
        "grid": [size, size, size],
        "steps": steps,
        "dtype_candidate": "float32",
        "dtype_oracle": "float64",
        "initial_sha256": sha256(options.initial),
        "actual_sha256": sha256(options.actual),
        "candidate_final_timestep": frozen_dt,
        "forward_euler_oracle": comparison(
            actual, forward, initial_unique, forward_dt
        ),
        "qualified_ssp_rk3": comparison(
            actual, qualified, initial_unique, qualified_dt
        ),
    }
    options.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
