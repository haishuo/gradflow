#!/usr/bin/env python3
"""Run one held-out selector or large-grid confirmation point."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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


def run_lane(lane: str, size: int, steps: int, manifest: dict[str, object],
             calibration: dict[str, object], environment: dict[str, str]) -> dict[str, object]:
    dveb = Path(manifest["native"]["dveb"]["frozen_copy"])
    if lane == "auto":
        command = [
            str(dveb), "--model", str(calibration["model"]),
            "--verified-model", str(calibration["model_sha256"]),
            "--endpoint", "cpu-resident", "--explain-placement",
        ]
        run_environment = environment
    elif lane == "ceiling-cuda":
        command = [str(manifest["native"]["ceiling"]["frozen_copy"]),
                   "--target", "cuda"]
        run_environment = environment
    else:
        command = [str(dveb), "--internal-calibration", "--candidate", lane]
        run_environment = environment.copy()
        run_environment["DVEB_CALIBRATION"] = "1"
    command += ["--size", str(size), "--steps", str(steps)]
    started = time.perf_counter()
    completed = subprocess.run(
        command, env=run_environment, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    elapsed = time.perf_counter() - started
    record = last_json(completed.stdout) if completed.returncode == 0 else {}
    record.update(lane=lane, size=size, steps=steps, returncode=completed.returncode,
                  fresh_process_seconds=elapsed,
                  success=completed.returncode == 0 and bool(record.get("finite", True)))
    if lane == "auto" and "selected=" in completed.stderr:
        record["selected"] = completed.stderr.split("selected=", 1)[1].split()[0]
    if completed.stderr.strip():
        record["stderr"] = completed.stderr.strip()
    return record


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values), "median": statistics.median(values),
        "minimum": ordered[0], "maximum": ordered[-1],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    }


def summarize(records: list[dict[str, object]], lanes: list[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for lane in lanes:
        accepted = [item for item in records if item["lane"] == lane and item["success"]]
        lane_result: dict[str, object] = {
            "successes": len(accepted),
            "failures": sum(item["lane"] == lane and not item["success"] for item in records),
        }
        for endpoint in ("fresh_process_seconds", "process_seconds_after_main",
                         "execution_seconds"):
            values = [float(item[endpoint]) for item in accepted if endpoint in item]
            if values:
                lane_result[endpoint] = stats(values)
        result[lane] = lane_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--mode", choices=("selector", "large"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--steps", type=int, choices=(1, 10), required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists() or arguments.repetitions != 30 or arguments.warmups != 1:
        raise SystemExit("protocol requires a new output, 30 repetitions, and one warmup")
    valid_sizes = {8, 16, 32, 48, 64} if arguments.mode == "selector" else {96, 128}
    if arguments.size not in valid_sizes:
        raise SystemExit(f"N={arguments.size} is not declared for {arguments.mode}")

    manifest = json.loads(arguments.manifest.read_text())
    calibration = json.loads(arguments.calibration.read_text())
    if calibration["artifact_sha256"] != manifest["native"]["dveb"]["sha256"]:
        raise SystemExit("model and artifact do not match")
    for family in ("dveb", "ceiling"):
        binary = Path(manifest["native"][family]["frozen_copy"])
        if not binary.is_file() or sha256(binary) != manifest["native"][family]["sha256"]:
            raise SystemExit(f"changed frozen {family} artifact")
    model = Path(calibration["model"])
    lines = model.read_text().splitlines(keepends=True)
    observed_model = hashlib.sha256("".join(lines[:-1]).encode()).hexdigest()
    if observed_model != calibration["model_sha256"]:
        raise SystemExit("placement model failed verification")

    lanes = (["auto", *calibration["retained_candidates"]]
             if arguments.mode == "selector" else ["auto", "cuda", "ceiling-cuda"])
    environment = os.environ.copy()
    environment.update(OMP_DYNAMIC="FALSE", OMP_PROC_BIND="close", OMP_PLACES="cores")
    warmups = [run_lane(lane, arguments.size, arguments.steps, manifest,
                        calibration, environment) for lane in lanes]
    rng = random.Random(arguments.seed + arguments.size * 1000 + arguments.steps)
    records: list[dict[str, object]] = []
    blocks: list[dict[str, object]] = []
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    for repetition in range(arguments.repetitions):
        order = list(lanes)
        rng.shuffle(order)
        for lane in order:
            record = run_lane(lane, arguments.size, arguments.steps, manifest,
                              calibration, environment)
            record["repetition"] = repetition
            records.append(record)
            print(json.dumps(record), flush=True)
        blocks.append({"repetition": repetition, "order": order})
        checkpoint = {
            "schema_version": 1, "complete": False, "mode": arguments.mode,
            "size": arguments.size, "steps": arguments.steps,
            "repetitions_complete": repetition + 1, "lanes": lanes,
            "manifest": str(arguments.manifest.resolve()),
            "calibration": str(arguments.calibration.resolve()),
            "warmup_records": warmups, "blocks": blocks, "records": records,
        }
        arguments.output.write_text(json.dumps(checkpoint, indent=2) + "\n")

    result = {
        "schema_version": 1, "complete": True,
        "protocol": "DVEB_FINAL_REQUALIFICATION_PROTOCOL.md", "mode": arguments.mode,
        "size": arguments.size, "steps": arguments.steps,
        "repetitions": arguments.repetitions, "warmups": arguments.warmups,
        "seed": arguments.seed, "lanes": lanes,
        "manifest": str(arguments.manifest.resolve()),
        "calibration": str(arguments.calibration.resolve()),
        "environment": {"platform": platform.platform(), "omp": {
            key: environment[key] for key in ("OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES")
        }},
        "warmup_records": warmups, "blocks": blocks, "records": records,
        "summary": summarize(records, lanes),
    }
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "summary": result["summary"]},
                     indent=2))


if __name__ == "__main__":
    main()
