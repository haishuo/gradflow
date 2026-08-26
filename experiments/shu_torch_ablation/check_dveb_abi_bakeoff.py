#!/usr/bin/env python3
"""Full-state gate for every lane in the frozen forced-target ABI bakeoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gradflow import (  # noqa: E402
    euler_cfl_timestep,
    euler_ssp_rk3_step,
    periodic_vortex,
)


EXPERIMENT = Path(__file__).resolve().parent
WORKER = EXPERIMENT / "abi_bakeoff_worker.py"
POINTS = ((6, 1), (6, 10), (32, 1), (128, 1))
BOUND = 2.0e-5
LANES = (
    "authority-cpu",
    "fortran",
    "dveb-cpu6",
    "dveb-cpu12",
    "dveb-cuda",
    "direct-eager",
    "persistent-compile",
    "aot-inductor",
    "ceiling-cpu",
    "ceiling-cuda",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def last_json(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"no JSON record in output:\n{text}")


def run_checked(
    command: list[str], *, environment: dict[str, str] | None = None,
    stdin: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=EXPERIMENT,
        env=environment,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"correctness command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed, elapsed


def authority(path: Path, size: int, steps: int) -> dict[str, object]:
    started = time.perf_counter()
    state, spacing = periodic_vortex((size,) * 3, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(steps):
            dt = euler_cfl_timestep(state, spacing, 0.1)
            state = euler_ssp_rk3_step(state, spacing, dt)
    state.contiguous().numpy().tofile(path)
    return {
        "lane": "authority-cpu",
        "wall_seconds": time.perf_counter() - started,
        "finite": bool(torch.isfinite(state).all()),
        "checksum_float64": float(state.to(torch.float64).sum()),
    }


def worker_result(
    manifest: dict[str, object], lane: str, path: Path, size: int, steps: int,
) -> dict[str, object]:
    command = [
        sys.executable, str(WORKER), "--lane", lane,
        "--endpoint", "correctness", "--size", str(size),
        "--steps", str(steps), "--output-state", str(path),
    ]
    environment = os.environ.copy()
    environment.update({
        "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores", "OMP_SCHEDULE": "static",
    })
    if lane.startswith("dveb-"):
        command += [
            "--artifact-manifest",
            manifest["dveb"]["manifest"]["frozen_copy"],
        ]
    elif lane == "persistent-compile":
        environment["TORCHINDUCTOR_CACHE_DIR"] = manifest["compile_caches"][str(size)]["path"]
    elif lane == "aot-inductor":
        package = manifest["aot_packages"][str(size)]
        command += ["--package", package["path"]]
        environment["TORCHINDUCTOR_CACHE_DIR"] = package["runtime_cache"]
    completed, elapsed = run_checked(command, environment=environment)
    record = last_json(completed.stdout)
    record["external_seconds"] = elapsed
    if completed.stderr.strip():
        record["stderr"] = completed.stderr.strip()
    return record


def native_result(
    manifest: dict[str, object], lane: str, path: Path, size: int, steps: int,
) -> tuple[dict[str, object], int]:
    environment = os.environ.copy()
    environment.update({
        "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores", "OMP_SCHEDULE": "static",
    })
    if lane == "fortran":
        binary = manifest["native"]["fortran"]["path"]
        environment["WENO_WRITE_STATE"] = str(path)
        completed, elapsed = run_checked(
            [binary], environment=environment,
            stdin=f"{size} {size} {size}\n0.1\n{steps}\n1.0e6\n",
        )
        record = {
            "lane": lane, "external_seconds": elapsed,
            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip(),
            "finite": not any(flag in completed.stderr for flag in (
                "IEEE_INVALID", "IEEE_OVERFLOW", "IEEE_DIVIDE_BY_ZERO"
            )),
        }
        return record, 1
    target = lane.removeprefix("ceiling-")
    binary = manifest["native"]["ceiling"]["frozen_copy"]
    completed, elapsed = run_checked([
        binary, "--target", target, "--size", str(size),
        "--steps", str(steps), "--output", str(path),
    ], environment=environment)
    record = last_json(completed.stdout)
    record["external_seconds"] = elapsed
    if completed.stderr.strip():
        record["stderr"] = completed.stderr.strip()
    return record, 0


def state_view(path: Path, size: int, offset_floats: int = 0) -> np.memmap:
    shape = (5, size + 1, size + 1, size + 1)
    return np.memmap(path, dtype=np.float32, mode="r", offset=4 * offset_floats, shape=shape)


def endpoint_error(state: np.ndarray) -> float:
    return max(
        float(np.max(np.abs(state[..., 0] - state[..., -1]))),
        float(np.max(np.abs(state[..., 0, :] - state[..., -1, :]))),
        float(np.max(np.abs(state[..., 0, :, :] - state[..., -1, :, :]))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing result: {args.output}")
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "gradflow-dveb-abi-bakeoff-preparation-v1":
        raise SystemExit("unexpected preparation manifest")

    point_records = []
    overall_maximum = 0.0
    with tempfile.TemporaryDirectory(prefix="gradflow-abi-correctness-") as temporary:
        scratch = Path(temporary)
        for size, steps in POINTS:
            raw_paths = {lane: scratch / f"{lane}-n{size}-s{steps}.bin" for lane in LANES}
            records = {"authority-cpu": authority(raw_paths["authority-cpu"], size, steps)}
            offsets = {lane: 0 for lane in LANES}
            for lane in ("fortran", "ceiling-cpu", "ceiling-cuda"):
                records[lane], offsets[lane] = native_result(
                    manifest, lane, raw_paths[lane], size, steps
                )
            for lane in (
                "dveb-cpu6", "dveb-cpu12", "dveb-cuda", "direct-eager",
                "persistent-compile", "aot-inductor",
            ):
                records[lane] = worker_result(manifest, lane, raw_paths[lane], size, steps)

            views = {
                lane: state_view(raw_paths[lane], size, offsets[lane])
                for lane in LANES
            }
            errors = {}
            for left, right in itertools.combinations(LANES, 2):
                error = float(np.max(np.abs(views[left] - views[right])))
                errors[f"{left}__{right}"] = error
                overall_maximum = max(overall_maximum, error)
            endpoint_errors = {lane: endpoint_error(view) for lane, view in views.items()}
            overall_maximum = max(overall_maximum, *endpoint_errors.values())

            archive_path = args.archive_dir / f"correctness_n{size}_s{steps}.npz"
            np.savez_compressed(archive_path, **{lane: np.asarray(view) for lane, view in views.items()})
            archive = {
                "path": str(archive_path.resolve()),
                "sha256": sha256(archive_path),
                "bytes": archive_path.stat().st_size,
                "arrays": list(LANES),
            }
            point = {
                "size": size, "steps": steps, "records": records,
                "pairwise_linf": errors, "endpoint_linf": endpoint_errors,
                "archive": archive,
            }
            point_records.append(point)
            print(json.dumps({
                "size": size, "steps": steps,
                "maximum_pairwise_linf": max(errors.values()),
                "maximum_endpoint_linf": max(endpoint_errors.values()),
                "archive": archive,
            }, sort_keys=True), flush=True)

    report = {
        "schema": "gradflow-dveb-abi-bakeoff-correctness-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "bound": BOUND,
        "maximum_error": overall_maximum,
        "passed": overall_maximum <= BOUND,
        "points": point_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"], "maximum_error": overall_maximum}))
    if not report["passed"]:
        raise SystemExit("full-state correctness gate failed")


if __name__ == "__main__":
    main()
