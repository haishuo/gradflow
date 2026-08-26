#!/usr/bin/env python3
"""Full-array correctness gate for the frozen DVEB device E4 addendum."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gradflow import (  # noqa: E402
    DvebArtifact,
    DvebDeviceContext,
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    periodic_vortex,
)


POINTS = ((6, 1), (6, 10), (32, 1))
BOUND = 2.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def advance(state: torch.Tensor, spacing: tuple[float, float, float],
            steps: int) -> torch.Tensor:
    result = state
    with torch.inference_mode():
        for _ in range(steps):
            dt = euler_cfl_timestep(result, spacing, 0.1)
            result = euler_ssp_rk3_step(result, spacing, dt)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing result: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "gradflow-dveb-device-e4-preparation-v1":
        raise SystemExit("unexpected preparation manifest")
    artifact = DvebArtifact.from_manifest(manifest["device_artifact_manifest"])
    args.archive_dir.mkdir(parents=True, exist_ok=True)

    points = []
    maximum = 0.0
    for size, steps in POINTS:
        state_cpu, spacing_raw = periodic_vortex((size,) * 3, dtype=torch.float32)
        state = state_cpu.cuda().contiguous()
        spacing = tuple(float(value) for value in spacing_raw)
        expected = advance(state.clone(), spacing, steps)
        stream = torch.cuda.Stream()
        with DvebDeviceContext(artifact, size) as context, torch.cuda.stream(stream):
            result = context.run(state, steps=steps)
        error = float((result.state - expected).abs().max())
        maximum = max(maximum, error)
        archive = args.archive_dir / f"device_e4_correctness_n{size}_s{steps}.npz"
        np.savez_compressed(
            archive,
            pytorch=expected.detach().cpu().numpy(),
            dveb_device=result.state.detach().cpu().numpy(),
        )
        points.append({
            "size": size,
            "steps": steps,
            "maximum_absolute_error": error,
            "finite": bool(torch.isfinite(result.state).all()),
            "archive": {"path": str(archive.resolve()), "sha256": sha256(archive),
                        "bytes": archive.stat().st_size},
            "dveb_execution_seconds": result.execution_seconds,
            "dveb_total_seconds": result.total_seconds,
        })

    state_cpu, _ = periodic_vortex((6,) * 3, dtype=torch.float32)
    aliased = state_cpu.cuda().contiguous()
    pointer = aliased.data_ptr()
    stream = torch.cuda.Stream()
    with DvebDeviceContext(artifact, 6) as context, torch.cuda.stream(stream):
        alias_result = context.run(aliased, steps=1, out=aliased)
    alias_pass = alias_result.state.data_ptr() == pointer and bool(
        torch.isfinite(alias_result.state).all()
    )
    report = {
        "schema": "gradflow-dveb-device-e4-correctness-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "points": points,
        "bound": BOUND,
        "maximum_absolute_error": maximum,
        "nondefault_stream_exact_alias_pass": alias_pass,
        "pass": maximum <= BOUND and alias_pass and all(item["finite"] for item in points),
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if not report["pass"]:
        raise SystemExit("device E4 correctness gate failed")


if __name__ == "__main__":
    main()
