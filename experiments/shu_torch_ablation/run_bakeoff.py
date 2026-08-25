#!/usr/bin/env python3
"""Orchestrate one-run, fresh-process matched 3-D Shu WENO bake-offs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
WORKER = EXPERIMENT_DIR / "bakeoff_worker.py"
AOT_BUILDER = EXPERIMENT_DIR / "build_aot_package.py"
DEFAULT_FORTRAN = EXPERIMENT_DIR / "build" / "shu_euler_3d"


def _last_json(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"subprocess emitted no JSON record:\n{stdout}")


def _run(
    command: list[str], *, environment: dict[str, str] | None = None
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=EXPERIMENT_DIR,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    process_seconds = time.perf_counter() - started
    record = _last_json(completed.stdout)
    record["fresh_process_seconds"] = process_seconds
    if completed.stderr.strip():
        record["stderr"] = completed.stderr.strip()
    return record, process_seconds


def _run_fortran(binary: Path, size: int, steps: int) -> dict[str, object]:
    stdin = f"{size} {size} {size}\n0.1\n{steps}\n1.0e6\n"
    started = time.perf_counter()
    completed = subprocess.run(
        [str(binary)],
        cwd=EXPERIMENT_DIR,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return {
        "lane": "fortran",
        "size": size,
        "steps": steps,
        "cfl": 0.1,
        "fresh_process_seconds": time.perf_counter() - started,
        "finite": not any(
            flag in completed.stderr
            for flag in ("IEEE_INVALID", "IEEE_OVERFLOW", "IEEE_DIVIDE_BY_ZERO")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fortran", type=Path, default=DEFAULT_FORTRAN)
    arguments = parser.parse_args()
    if not arguments.fortran.is_file():
        raise SystemExit(f"missing Fortran executable: {arguments.fortran}")

    records: list[dict[str, object]] = []
    builds: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="gradflow-bakeoff-") as temporary:
        temporary_path = Path(temporary)
        for size in arguments.sizes:
            fortran = _run_fortran(arguments.fortran, size, arguments.steps)
            records.append(fortran)
            print(json.dumps(fortran), flush=True)

            base_worker = [
                sys.executable,
                str(WORKER),
                "--size",
                str(size),
                "--steps",
                str(arguments.steps),
            ]
            for lane in ("direct-eager", "conv-eager"):
                record, _ = _run(base_worker + ["--lane", lane])
                records.append(record)
                print(json.dumps(record), flush=True)

            cold_environment = os.environ.copy()
            cold_environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                temporary_path / f"cold-cache-n{size}"
            )
            cold_environment["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
            cold, _ = _run(
                base_worker + ["--lane", "compile"],
                environment=cold_environment,
            )
            cold["lane"] = "compile-cold"
            records.append(cold)
            print(json.dumps(cold), flush=True)

            cached_environment = os.environ.copy()
            cached_environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                temporary_path / f"persistent-cache-n{size}"
            )
            cached_environment["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            cached_environment["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
            preparation, _ = _run(
                base_worker + ["--lane", "compile"],
                environment=cached_environment,
            )
            builds.append(
                {
                    "lane": "compile-cache-preparation",
                    "size": size,
                    "seconds": preparation["fresh_process_seconds"],
                }
            )
            cached, _ = _run(
                base_worker + ["--lane", "compile"],
                environment=cached_environment,
            )
            cached["lane"] = "compile-persistent-cache"
            records.append(cached)
            print(json.dumps(cached), flush=True)

            package = temporary_path / f"shu-weno3d-n{size}.pt2"
            aot_environment = os.environ.copy()
            aot_environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                temporary_path / f"aot-build-cache-n{size}"
            )
            build, _ = _run(
                [
                    sys.executable,
                    str(AOT_BUILDER),
                    "--size",
                    str(size),
                    "--output",
                    str(package),
                ],
                environment=aot_environment,
            )
            build["lane"] = "aot-package-build"
            builds.append(build)
            aot_cold_environment = os.environ.copy()
            aot_cold_environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                temporary_path / f"aot-cold-runtime-cache-n{size}"
            )
            aot_cold, _ = _run(
                base_worker
                + ["--lane", "aot", "--package", str(package)],
                environment=aot_cold_environment,
            )
            aot_cold["lane"] = "aot-inductor-cold-package"
            records.append(aot_cold)
            print(json.dumps(aot_cold), flush=True)

            aot_cached_environment = os.environ.copy()
            aot_cached_environment["TORCHINDUCTOR_CACHE_DIR"] = str(
                temporary_path / f"aot-cached-runtime-cache-n{size}"
            )
            aot_preparation, _ = _run(
                base_worker
                + ["--lane", "aot", "--package", str(package)],
                environment=aot_cached_environment,
            )
            builds.append(
                {
                    "lane": "aot-load-cache-preparation",
                    "size": size,
                    "seconds": aot_preparation["fresh_process_seconds"],
                }
            )
            aot_cached, _ = _run(
                base_worker
                + ["--lane", "aot", "--package", str(package)],
                environment=aot_cached_environment,
            )
            aot_cached["lane"] = "aot-inductor-cached-package"
            records.append(aot_cached)
            print(json.dumps(aot_cached), flush=True)

    output = {
        "schema_version": 1,
        "runs_per_configuration": 1,
        "precision": "float32",
        "dimension": 3,
        "steps": arguments.steps,
        "sizes": arguments.sizes,
        "timing_endpoint": "fresh process to final state materialized in host memory",
        "timestep_policy": (
            "recompute Shu sum-of-directional-speeds CFL with cfl=0.1 "
            "before every step"
        ),
        "aot_build_excluded_from_run": True,
        "persistent_cache_preparation_excluded_from_run": True,
        "aot_load_cache_preparation_excluded_from_cached_run": True,
        "records": records,
        "builds": builds,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
