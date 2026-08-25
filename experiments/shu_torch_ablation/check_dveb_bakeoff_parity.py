#!/usr/bin/env python3
"""Full-state correctness gate for the hash-frozen DVEB bakeoff artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch

from shu_euler_torch import cfl_timestep, periodic_vortex, ssp_rk3_step


POINTS = ((6, 1), (6, 10), (32, 1), (128, 1))
BOUND = 2.0e-5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def torch_result(size: int, steps: int) -> np.ndarray:
    state, spacing = periodic_vortex((size,) * 3, dtype=torch.float32)
    for _ in range(steps):
        dt = cfl_timestep(state, spacing, 0.1)
        state = ssp_rk3_step(state, spacing, dt)
    return state.numpy()


def native_result(
    binary: Path, family: str, target: str, size: int, steps: int, output: Path
) -> tuple[np.ndarray, dict[str, object]]:
    environment = os.environ.copy()
    environment.update({"OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})
    command = [str(binary)]
    if family == "dveb":
        environment["DVEB_CALIBRATION"] = "1"
        candidate = "cpu_simd[6]" if target == "cpu" else "cuda"
        command += ["--internal-calibration", "--candidate", candidate]
    else:
        command += ["--target", target]
    command += [
        "--size", str(size), "--steps", str(steps), "--output", str(output)
    ]
    completed = subprocess.run(
        command, env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True,
    )
    record = json.loads(completed.stdout.splitlines()[-1])
    state = np.fromfile(output, dtype=np.float32).reshape(
        5, size + 1, size + 1, size + 1
    )
    return state, record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite result: {arguments.output}")
    manifest = json.loads(arguments.manifest.read_text())
    binaries = {
        family: Path(manifest["native"][family]["frozen_copy"])
        for family in ("dveb", "ceiling")
    }
    for family, binary in binaries.items():
        expected = manifest["native"][family]["sha256"]
        if not binary.is_file() or sha256(binary) != expected:
            raise SystemExit(f"{family} artifact failed its hash check")

    results: list[dict[str, object]] = []
    maximum = 0.0
    with tempfile.TemporaryDirectory(prefix="gradflow-dveb-parity-") as temporary:
        scratch = Path(temporary)
        for size, steps in POINTS:
            reference = torch_result(size, steps)
            states: dict[str, np.ndarray] = {}
            records: dict[str, object] = {}
            for family in ("dveb", "ceiling"):
                for target in ("cpu", "cuda"):
                    name = f"{family}-{target}"
                    state, record = native_result(
                        binaries[family], family, target, size, steps,
                        scratch / f"{name}-n{size}-s{steps}.bin",
                    )
                    states[name] = state
                    records[name] = record
            errors = {
                f"{name}_vs_torch": float(np.max(np.abs(state - reference)))
                for name, state in states.items()
            }
            errors.update({
                "dveb_cpu_vs_cuda": float(np.max(np.abs(
                    states["dveb-cpu"] - states["dveb-cuda"]
                ))),
                "dveb_vs_ceiling_cpu": float(np.max(np.abs(
                    states["dveb-cpu"] - states["ceiling-cpu"]
                ))),
                "dveb_vs_ceiling_cuda": float(np.max(np.abs(
                    states["dveb-cuda"] - states["ceiling-cuda"]
                ))),
            })
            maximum = max(maximum, *errors.values())
            point = {
                "size": size, "steps": steps, "errors": errors,
                "records": records,
            }
            results.append(point)
            print(json.dumps(point), flush=True)

    report = {
        "schema_version": 1,
        "manifest": str(arguments.manifest.resolve()),
        "bound": BOUND,
        "maximum_error": maximum,
        "passed": maximum <= BOUND,
        "points": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if maximum > BOUND:
        raise SystemExit("FAIL: full-state parity bound exceeded")


if __name__ == "__main__":
    main()
