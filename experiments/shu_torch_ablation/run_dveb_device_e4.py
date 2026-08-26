#!/usr/bin/env python3
"""Run one randomized point of the frozen device-ABI E4 addendum."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKER = HERE / "device_abi_e4_worker.py"
LANES = ("dveb-device", "direct-eager", "persistent-compile", "aot-inductor")
POINTS = {
    (8, 1),
    (16, 1),
    (32, 1),
    (64, 1),
    (96, 1),
    (128, 1),
    (16, 10),
    (32, 10),
    (64, 10),
    (128, 10),
}


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
    raise RuntimeError(f"worker emitted no JSON:\n{text}")


def command_output(command: list[str]) -> str:
    return subprocess.run(command, text=True, capture_output=True).stdout.strip()


def telemetry() -> dict[str, str]:
    return {
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,pstate,power.draw",
                "--format=csv,noheader,nounits",
            ]
        )
    }


def validate(manifest: dict[str, object]) -> dict[str, object]:
    if manifest.get("schema") != "gradflow-dveb-device-e4-preparation-v1":
        raise SystemExit("unexpected preparation manifest")
    for item in manifest["device_artifacts"].values():
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise SystemExit(f"device artifact identity failed: {path}")
    base_record = manifest["base_manifest"]
    base_path = Path(base_record["path"])
    if sha256(base_path) != base_record["sha256"]:
        raise SystemExit("base manifest identity failed")
    base = json.loads(base_path.read_text())
    for package in base["aot_packages"].values():
        path = Path(package["path"])
        if not path.is_file() or sha256(path) != package["sha256"]:
            raise SystemExit(f"AOT package identity failed: {path}")
    return base


def run_lane(
    manifest: dict[str, object],
    base: dict[str, object],
    lane: str,
    size: int,
    steps: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(WORKER),
        "--lane",
        lane,
        "--size",
        str(size),
        "--steps",
        str(steps),
        "--warmups",
        "5",
        "--repetitions",
        "5",
    ]
    environment = os.environ.copy()
    if lane == "dveb-device":
        command += ["--artifact-manifest", manifest["device_artifact_manifest"]]
    elif lane == "persistent-compile":
        environment["TORCHINDUCTOR_CACHE_DIR"] = base["compile_caches"][str(size)][
            "path"
        ]
    elif lane == "aot-inductor":
        package = base["aot_packages"][str(size)]
        command += ["--package", package["path"]]
        environment["TORCHINDUCTOR_CACHE_DIR"] = package["runtime_cache"]
    completed = subprocess.run(
        command, cwd=HERE, env=environment, text=True, capture_output=True, timeout=1800
    )
    try:
        record = last_json(completed.stdout)
    except RuntimeError:
        record = {"lane": lane, "stdout": completed.stdout}
    record.update(
        returncode=completed.returncode,
        success=completed.returncode == 0 and bool(record.get("finite")),
    )
    if completed.stderr.strip():
        record["stderr"] = completed.stderr.strip()
    return record


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    summary = {}
    for lane in LANES:
        selected = [item for item in records if item["lane"] == lane]
        values = [
            float(value["call_seconds"])
            for item in selected
            if item["success"]
            for value in item["observations"]
        ]
        entry = {
            "workers": len(selected),
            "failures": sum(not item["success"] for item in selected),
            "count": len(values),
        }
        if values:
            entry.update(
                minimum=min(values),
                median=statistics.median(values),
                mean=statistics.fmean(values),
                maximum=max(values),
            )
        summary[lane] = entry
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.size, args.steps) not in POINTS:
        raise SystemExit("point is outside the frozen protocol")
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    base = validate(manifest)
    rng = random.Random(20260827 + args.size * 1000 + args.steps * 10)
    records = []
    blocks = []
    for block in range(6):
        order = list(LANES)
        rng.shuffle(order)
        block_record = {"block": block, "order": order, "telemetry_before": telemetry()}
        for lane in order:
            record = run_lane(manifest, base, lane, args.size, args.steps)
            record["block"] = block
            records.append(record)
            print(
                json.dumps(
                    {"block": block, "lane": lane, "success": record["success"]}
                ),
                flush=True,
            )
        block_record["telemetry_after"] = telemetry()
        blocks.append(block_record)
    report = {
        "schema": "gradflow-dveb-device-e4-timing-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "size": args.size,
        "steps": args.steps,
        "seed": 20260827,
        "lanes": LANES,
        "blocks": blocks,
        "records": records,
        "summary": summarize(records),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "summary": report["summary"]}, sort_keys=True
        )
    )
    if any(not item["success"] for item in records):
        raise SystemExit("a counted lane failed")


if __name__ == "__main__":
    main()
