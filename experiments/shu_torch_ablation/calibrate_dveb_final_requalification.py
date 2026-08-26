#!/usr/bin/env python3
"""Build a disjoint-training WENO placement model for the final DVEB artifact."""

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
    "cpu_simd[1]", "cpu_simd[2]", "cpu_simd[4]",
    "cpu_simd[6]", "cpu_simd[12]", "cuda",
)
ENDPOINTS = (("resident", "execution_seconds"),
             ("cpu-resident", "process_seconds_after_main"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output(command: list[str]) -> str:
    return subprocess.run(
        command, check=False, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def run(binary: Path, candidate: str, size: int, steps: int,
        environment: dict[str, str]) -> dict[str, object]:
    command = [
        str(binary), "--internal-calibration", "--candidate", candidate,
        "--size", str(size), "--steps", str(steps),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, env=environment, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"failed {candidate} N={size} steps={steps}: {completed.stderr}"
        )
    record = json.loads(completed.stdout.splitlines()[-1])
    record.update(candidate=candidate, size=size, steps=steps,
                  external_seconds=elapsed, stderr=completed.stderr.strip())
    return record


def dominance(rows: list[dict[str, object]]) -> tuple[list[str], list[dict[str, str]]]:
    medians = {
        (str(row["endpoint"]), int(row["size"]), int(row["steps"]),
         str(row["candidate"])): float(row["median"])
        for row in rows
    }
    points = sorted({(str(row["endpoint"]), int(row["size"]), int(row["steps"]))
                     for row in rows})
    cpus = [candidate for candidate in CANDIDATES if candidate != "cuda"]
    exclusions: list[dict[str, str]] = []
    excluded: set[str] = set()
    for candidate in cpus:
        for challenger in cpus:
            if challenger == candidate:
                continue
            ratios = [medians[(*point, challenger)] / medians[(*point, candidate)]
                      for point in points]
            if all(ratio <= 0.98 for ratio in ratios) and any(
                ratio <= 0.95 for ratio in ratios
            ):
                excluded.add(candidate)
                exclusions.append({"candidate": candidate, "dominated_by": challenger})
                break
    return [candidate for candidate in CANDIDATES if candidate not in excluded], exclusions


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0xD0EB2026)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.repetitions != 7 or arguments.warmups != 1:
        raise SystemExit("protocol requires a new output, seven repetitions, and one warmup")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(arguments.manifest.read_text())
    binary = Path(manifest["native"]["dveb"]["frozen_copy"])
    if not binary.is_file() or sha256(binary) != manifest["native"]["dveb"]["sha256"]:
        raise SystemExit("frozen DVEB artifact failed verification")

    environment = os.environ.copy()
    environment.update(DVEB_CALIBRATION="1", OMP_DYNAMIC="FALSE",
                       OMP_PROC_BIND="close", OMP_PLACES="cores")
    points = [(size, steps) for steps in (1, 10)
              for size in (7, 12, 24, 40, 56, 72)]
    rng = random.Random(arguments.seed)
    raw: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for size, steps in points:
        order = list(CANDIDATES)
        rng.shuffle(order)
        for candidate in order:
            run(binary, candidate, size, steps, environment)
            observations = [run(binary, candidate, size, steps, environment)
                            for _ in range(arguments.repetitions)]
            raw.extend(observations)
            for endpoint, field in ENDPOINTS:
                values = [float(item[field]) for item in observations]
                rows.append({
                    "endpoint": endpoint, "size": size, "steps": steps,
                    "candidate": candidate, "median": statistics.median(values),
                    "p05": percentile(values, 0.05),
                    "p95": percentile(values, 0.95),
                })
            print(json.dumps({"calibrated": candidate, "size": size,
                              "steps": steps}), flush=True)

    retained, exclusions = dominance(rows)
    machine = {
        "platform": platform.platform(), "cpu": output(["lscpu"]),
        "gpu": output(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                       "--format=csv,noheader,nounits"]),
        "affinity": "OMP_PROC_BIND=close;OMP_PLACES=cores;schedule=static",
    }
    lines = [
        "schema\tdveb-placement-v1",
        f"program_sha256\t{manifest['program_sha256']}",
        f"module_sha256\t{manifest['module_sha256']}",
        f"artifact_sha256\t{manifest['native']['dveb']['sha256']}",
        "interpolation\tpiecewise-log-linear-by-size-exact-steps",
        "hysteresis\tcpu-family=2pct-or-0.1ms;cuda=5pct-or-0.25ms",
    ]
    for key, value in machine.items():
        lines.append(f"machine\t{key}\t{quote(str(value), safe='-_.[]')}")
    lines.extend(f"candidate\t{candidate}" for candidate in retained)
    for exclusion in exclusions:
        lines.append(f"excluded\t{exclusion['candidate']}\tdominated-by={exclusion['dominated_by']}")
    for row in rows:
        if row["candidate"] in retained:
            lines.append(
                f"sample\t{row['endpoint']}\t{row['size']}\t{row['steps']}\t"
                f"{row['candidate']}\t{row['median']:.17g}\t{row['p05']:.17g}\t"
                f"{row['p95']:.17g}"
            )
    body = "\n".join(lines) + "\n"
    model_hash = hashlib.sha256(body.encode()).hexdigest()
    model_path = arguments.output.with_suffix(".placement.tsv")
    raw_path = arguments.output.with_suffix(".raw.json")
    model_path.write_text(body + f"model_sha256\t{model_hash}\n")
    raw_path.write_text(json.dumps(raw, indent=2) + "\n")
    report = {
        "schema_version": 1, "purpose": "excluded WENO placement preparation",
        "manifest": str(arguments.manifest.resolve()),
        "artifact_sha256": manifest["native"]["dveb"]["sha256"],
        "program_sha256": manifest["program_sha256"],
        "module_sha256": manifest["module_sha256"],
        "training_sizes": [7, 12, 24, 40, 56, 72], "training_steps": [1, 10],
        "repetitions": arguments.repetitions, "warmups": arguments.warmups,
        "seed": arguments.seed, "initial_candidates": list(CANDIDATES),
        "retained_candidates": retained, "exclusions": exclusions,
        "dominance_rule": "another CPU <=0.98x at all points/endpoints and <=0.95x somewhere",
        "model": str(model_path.resolve()), "model_sha256": model_hash,
        "raw": str(raw_path.resolve()), "raw_sha256": sha256(raw_path),
        "machine": machine,
        "calibration_wall_seconds_including_warmups": time.perf_counter() - started,
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
