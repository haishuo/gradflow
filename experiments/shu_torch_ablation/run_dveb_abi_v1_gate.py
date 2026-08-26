#!/usr/bin/env python3
"""Record the bounded arbitrary-state DVEB ABI v1 parity gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from gradflow import (
    DvebArtifact,
    Solver,
    periodic_vortex,
    synchronize_duplicate_endpoints,
)


CASES = ((6, 1), (6, 10), (32, 1))
TOLERANCE = 2.0e-5


def arbitrary_state(intervals: int) -> torch.Tensor:
    state, _ = periodic_vortex((intervals,) * 3)
    coordinate = torch.arange(intervals + 1, dtype=state.dtype) * (
        2.0 * torch.pi / intervals
    )
    z, y, x = torch.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    factor = 1.0 + 0.01 * torch.sin(x) * torch.cos(y) * torch.sin(z)
    density0 = state[0]
    velocity = state[1:4] / density0
    kinetic0 = 0.5 * state[1:4].square().sum(dim=0) / density0
    pressure = 0.4 * (state[4] - kinetic0)
    density = density0 * factor
    momentum = density.unsqueeze(0) * velocity
    energy = pressure / 0.4 + 0.5 * momentum.square().sum(dim=0) / density
    return synchronize_duplicate_endpoints(
        torch.cat((density.unsqueeze(0), momentum, energy.unsqueeze(0)))
    ).contiguous()


def run(arguments: argparse.Namespace) -> dict:
    artifact = DvebArtifact.from_manifest(arguments.manifest)
    rows = []
    for intervals, steps in CASES:
        state = arbitrary_state(intervals)
        solver = Solver(
            equations="euler", dimension=3, weno=("JS", 5),
            flux_split="global_lf", boundaries="periodic_duplicated",
            dtype=torch.float32, spacing=(10.0 / intervals,) * 3,
            dveb_artifact=artifact, cpu_workers=6,
        )
        started = time.perf_counter()
        pytorch = solver.run(state, steps=steps, backend="pytorch-eager")
        pytorch_wall = time.perf_counter() - started

        observations = {}
        native_states = {}
        for backend in ("cpu-simd", "cuda-native"):
            started = time.perf_counter()
            native_states[backend] = solver.run(state, steps=steps, backend=backend)
            wall = time.perf_counter() - started
            diagnostics = solver.last_run
            assert diagnostics is not None
            observations[backend] = {
                "selected": diagnostics.backend.selected,
                "execution_seconds": diagnostics.native_execution_seconds,
                "abi_total_seconds": diagnostics.native_total_seconds,
                "python_call_seconds": wall,
                "abi_copy_and_adapter_seconds": (
                    diagnostics.native_total_seconds
                    - diagnostics.native_execution_seconds
                ),
                "python_above_abi_seconds": wall - diagnostics.native_total_seconds,
                "peak_bytes": diagnostics.native_peak_bytes,
                "linf_vs_pytorch": float(
                    (native_states[backend] - pytorch).abs().max()
                ),
            }
        cpu_cuda = float(
            (native_states["cpu-simd"] - native_states["cuda-native"])
            .abs().max()
        )
        passed = all(
            item["linf_vs_pytorch"] <= TOLERANCE
            for item in observations.values()
        ) and cpu_cuda <= TOLERANCE
        rows.append({
            "intervals": intervals,
            "steps": steps,
            "elements": state.numel(),
            "pytorch_eager_wall_seconds": pytorch_wall,
            "cpu_cuda_linf": cpu_cuda,
            "backends": observations,
            "passed": passed,
        })
    return {
        "schema": "gradflow-dveb-abi-v1-gate-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "correctness and ABI-overhead observation; not a benchmark",
        "tolerance": {"rtol": 0.0, "atol": TOLERANCE},
        "state": "non-vortex physically admissible periodic perturbation",
        "artifact": {
            "manifest": str(Path(arguments.manifest).resolve()),
            "library": str(artifact.library),
            "library_sha256": artifact.library_sha256,
            "program_sha256": artifact.program_sha256,
            "module_sha256": artifact.module_sha256,
        },
        "environment": {
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "cases": rows,
        "passed": all(row["passed"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = run(arguments)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
