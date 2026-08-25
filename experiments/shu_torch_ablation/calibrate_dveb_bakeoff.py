#!/usr/bin/env python3
"""Calibrate a frozen automatic-placement DVEB artifact without touching DVEB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import time
from urllib.parse import quote


CANDIDATES = (
    "cpu_simd[4]",
    "cpu_simd[6]",
    "cpu_simd[12]",
    "cuda",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def run_candidate(
    binary: Path, candidate: str, size: int, steps: int, environment: dict[str, str]
) -> dict[str, object]:
    command = [
        str(binary), "--internal-calibration", "--candidate", candidate,
        "--size", str(size), "--steps", str(steps),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"calibration failed for {candidate}, N={size}, steps={steps}:\n"
            f"{completed.stderr}"
        )
    record = json.loads(completed.stdout.splitlines()[-1])
    record.update({
        "candidate": candidate,
        "external_seconds": elapsed,
        "stderr": completed.stderr.strip(),
    })
    return record


def main() -> None:
    calibration_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sizes-one-step", type=int, nargs="+", required=True)
    parser.add_argument("--sizes-ten-steps", type=int, nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0xD0EB)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite calibration: {arguments.output}")
    if arguments.repetitions < 3 or arguments.warmups < 0:
        raise SystemExit("calibration requires at least three repetitions")

    manifest = json.loads(arguments.manifest.read_text())
    binary = Path(manifest["native"]["dveb"]["frozen_copy"])
    artifact_hash = manifest["native"]["dveb"]["sha256"]
    if not binary.is_file() or sha256(binary) != artifact_hash:
        raise SystemExit("frozen DVEB artifact failed its hash check")

    environment = os.environ.copy()
    environment.update({
        "DVEB_CALIBRATION": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    })
    rng = random.Random(arguments.seed)
    raw: list[dict[str, object]] = []
    rows: list[tuple[str, int, int, str, float, float, float]] = []
    points = [*( (size, 1) for size in arguments.sizes_one_step ),
              *( (size, 10) for size in arguments.sizes_ten_steps )]
    for size, steps in points:
        order = list(CANDIDATES)
        rng.shuffle(order)
        for candidate in order:
            for _ in range(arguments.warmups):
                run_candidate(binary, candidate, size, steps, environment)
            observations = [
                run_candidate(binary, candidate, size, steps, environment)
                for _ in range(arguments.repetitions)
            ]
            raw.extend(observations)
            for endpoint, field in (
                ("resident", "execution_seconds"),
                ("cpu-resident", "process_seconds_after_main"),
            ):
                values = [float(item[field]) for item in observations]
                rows.append((
                    endpoint, size, steps, candidate, statistics.median(values),
                    percentile(values, 0.05), percentile(values, 0.95),
                ))
            print(json.dumps({
                "calibrated": candidate, "size": size, "steps": steps,
                "median_cpu_resident": rows[-1][4],
            }), flush=True)

    program_hash = (
        "c6e5bd916f951ff412eac99863a74f8c98e5e14b044097a7ad59fe26f704c381"
    )
    module_hash = (
        "555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260"
    )
    machine = {
        "platform": platform.platform(),
        "cpu": command_output(["lscpu"]),
        "gpu": command_output([
            "nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ]),
        "affinity": "OMP_PROC_BIND=close;OMP_PLACES=cores;schedule=static",
    }
    lines = [
        "schema\tdveb-placement-v1",
        f"program_sha256\t{program_hash}",
        f"module_sha256\t{module_hash}",
        f"artifact_sha256\t{artifact_hash}",
        "interpolation\tpiecewise-log-linear-by-size-exact-steps",
        "hysteresis\tcpu-family=2pct-or-0.1ms;cuda=5pct-or-0.25ms",
    ]
    for key, value in machine.items():
        lines.append(f"machine\t{key}\t{quote(str(value), safe='-_.[]')}")
    lines.extend(f"candidate\t{candidate}" for candidate in CANDIDATES)
    for endpoint, size, steps, candidate, median, p05, p95 in rows:
        lines.append(
            f"sample\t{endpoint}\t{size}\t{steps}\t{candidate}\t"
            f"{median:.17g}\t{p05:.17g}\t{p95:.17g}"
        )
    body = "\n".join(lines) + "\n"
    model_hash = hashlib.sha256(body.encode()).hexdigest()
    model_path = arguments.output.with_suffix(".placement.tsv")
    raw_path = arguments.output.with_suffix(".raw.json")
    model_path.write_text(body + f"model_sha256\t{model_hash}\n")
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    report = {
        "schema_version": 1,
        "purpose": "excluded DVEB automatic-placement preparation",
        "manifest": str(arguments.manifest.resolve()),
        "artifact_sha256": artifact_hash,
        "program_sha256": program_hash,
        "module_sha256": module_hash,
        "model": str(model_path.resolve()),
        "model_sha256": model_hash,
        "raw": str(raw_path.resolve()),
        "raw_sha256": sha256(raw_path),
        "sizes_one_step": arguments.sizes_one_step,
        "sizes_ten_steps": arguments.sizes_ten_steps,
        "repetitions": arguments.repetitions,
        "warmups": arguments.warmups,
        "seed": arguments.seed,
        "candidates": CANDIDATES,
        "machine": machine,
        "calibration_wall_seconds_including_warmups": (
            time.perf_counter() - calibration_started
        ),
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
