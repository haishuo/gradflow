#!/usr/bin/env python3
"""Run one resumable randomized point of the frozen ABI bakeoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time


EXPERIMENT = Path(__file__).resolve().parent
WORKER = EXPERIMENT / "abi_bakeoff_worker.py"
PRIMARY = (
    "fortran", "dveb-cpu6", "dveb-cpu12", "dveb-cuda",
    "direct-eager", "persistent-compile", "aot-inductor",
)
CEILINGS = ("ceiling-cpu", "ceiling-cuda")
CALLABLE = PRIMARY[1:]
RESIDENT = ("direct-eager", "persistent-compile", "aot-inductor")


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
    raise RuntimeError(f"worker emitted no JSON object:\n{text}")


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def telemetry() -> dict[str, str]:
    return {
        "gpu": command_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,pstate,power.draw,clocks_throttle_reasons.active",
            "--format=csv,noheader,nounits",
        ]),
        "cpu_mhz": command_output([
            "awk", "/cpu MHz/ {s+=$4;n++} END {if(n) printf \"%.1f\",s/n}",
            "/proc/cpuinfo",
        ]),
    }


def worker_command(
    manifest: dict[str, object], lane: str, endpoint: str,
    size: int, steps: int, warmups: int, repetitions: int,
) -> tuple[list[str], dict[str, str]]:
    command = [
        sys.executable, str(WORKER), "--lane", lane,
        "--endpoint", endpoint, "--size", str(size), "--steps", str(steps),
        "--warmups", str(warmups), "--repetitions", str(repetitions),
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
    return command, environment


def run_lane(
    manifest: dict[str, object], lane: str, endpoint: str,
    size: int, steps: int, *, warmups: int = 0, repetitions: int = 1,
    timeout: float = 1800.0, environment_override: dict[str, str] | None = None,
) -> dict[str, object]:
    if lane == "fortran":
        command = [manifest["native"]["fortran"]["path"]]
        environment = os.environ.copy()
        environment.update({
            "OMP_DYNAMIC": "FALSE", "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores", "OMP_SCHEDULE": "static",
        })
        stdin = f"{size} {size} {size}\n0.1\n{steps}\n1.0e6\n"
    elif lane.startswith("ceiling-"):
        target = lane.removeprefix("ceiling-")
        command = [
            manifest["native"]["ceiling"]["frozen_copy"],
            "--target", target, "--size", str(size), "--steps", str(steps),
        ]
        environment = os.environ.copy()
        environment.update({
            "OMP_DYNAMIC": "FALSE", "OMP_NUM_THREADS": "12",
            "OMP_PROC_BIND": "close", "OMP_PLACES": "cores",
            "OMP_SCHEDULE": "static",
        })
        stdin = None
    else:
        command, environment = worker_command(
            manifest, lane,
            "resident" if endpoint == "E4" else ("warm" if endpoint == "E3" else "single"),
            size, steps, warmups, repetitions,
        )
        stdin = None
    if environment_override:
        environment.update(environment_override)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, cwd=EXPERIMENT, env=environment, input=stdin, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            timeout=timeout,
        )
        external_seconds = time.perf_counter() - started
    except subprocess.TimeoutExpired as error:
        return {
            "lane": lane, "endpoint": endpoint, "size": size, "steps": steps,
            "success": False, "failure": "timeout", "timeout_seconds": timeout,
            "stdout": error.stdout or "", "stderr": error.stderr or "",
        }
    if lane == "fortran":
        serious = ("IEEE_INVALID", "IEEE_OVERFLOW", "IEEE_DIVIDE_BY_ZERO")
        record: dict[str, object] = {
            "lane": lane,
            "finite": not any(flag in completed.stderr for flag in serious),
            "stdout": completed.stdout.strip(),
        }
    else:
        try:
            record = last_json(completed.stdout)
        except RuntimeError:
            record = {"lane": lane}
    record.update({
        "lane": lane, "endpoint": endpoint, "size": size, "steps": steps,
        "external_seconds": external_seconds, "returncode": completed.returncode,
    })
    if endpoint == "E1":
        record["fresh_process_seconds"] = external_seconds
    if completed.stderr.strip():
        record["stderr"] = completed.stderr.strip()
    record["success"] = completed.returncode == 0 and bool(record.get("finite", True))
    if not record["success"]:
        record["failure"] = "process-error" if completed.returncode else "nonfinite"
    return record


def validate_manifest(manifest: dict[str, object]) -> None:
    if manifest.get("schema") != "gradflow-dveb-abi-bakeoff-preparation-v1":
        raise SystemExit("unexpected preparation manifest schema")
    checks = [
        manifest["dveb"]["library"], manifest["dveb"]["header"],
        manifest["native"]["ceiling"], manifest["native"]["fortran"],
    ]
    for record in checks:
        path = Path(record.get("frozen_copy", record.get("path")))
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise SystemExit(f"artifact identity failed: {path}")
    for package in manifest["aot_packages"].values():
        path = Path(package["path"])
        if not path.is_file() or sha256(path) != package["sha256"]:
            raise SystemExit(f"AOT package identity failed: {path}")


def summarize(records: list[dict[str, object]], endpoint: str) -> dict[str, object]:
    by_lane = {}
    for lane in sorted({record["lane"] for record in records}):
        lane_records = [record for record in records if record["lane"] == lane]
        values = []
        for record in lane_records:
            if not record.get("success"):
                continue
            if endpoint == "E1":
                values.append(float(record["fresh_process_seconds"]))
            elif "observations" in record:
                values.extend(float(item["call_seconds"]) for item in record["observations"])
        item: dict[str, object] = {
            "processes": len(lane_records),
            "failures": sum(not bool(record.get("success")) for record in lane_records),
            "counted_observations": len(values),
        }
        if values:
            ordered = sorted(values)
            item.update({
                "minimum": ordered[0], "median": statistics.median(values),
                "mean": statistics.fmean(values), "maximum": ordered[-1],
            })
        by_lane[lane] = item
    return by_lane


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("capacity", "E1", "E2", "E3", "E4", "cold"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".partial.jsonl").exists():
        raise SystemExit(f"refusing existing output/checkpoint: {args.output}")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if str(args.size) not in manifest["aot_packages"]:
        raise SystemExit(f"no prepared package for N={args.size}")

    if args.mode == "capacity":
        lanes, blocks, warmups, per_worker = PRIMARY + CEILINGS, 1, 0, 1
    elif args.mode == "E1":
        lanes, blocks, warmups, per_worker = PRIMARY + CEILINGS, args.repetitions or 30, 0, 1
    elif args.mode == "E2":
        lanes, blocks, warmups, per_worker = CALLABLE, args.repetitions or 30, 0, 1
    elif args.mode == "E3":
        lanes, blocks, warmups, per_worker = CALLABLE, 6, 1, 5
    elif args.mode == "E4":
        lanes, blocks, warmups, per_worker = RESIDENT, 6, 5, 5
    else:
        lanes, blocks, warmups, per_worker = ("persistent-compile", "aot-inductor"), 5, 0, 1

    expected = None
    if args.mode in {"E1", "E2"}:
        expected = 30 if args.repetitions is None else args.repetitions
        if expected != 30:
            raise SystemExit("counted E1/E2 protocol requires exactly 30 repetitions")
    if args.mode in {"E3", "E4"} and args.repetitions is not None:
        raise SystemExit("E3/E4 block counts are frozen")

    checkpoint = args.output.with_suffix(args.output.suffix + ".partial.jsonl")
    rng = random.Random(args.seed + args.size * 1000 + args.steps * 10 + sum(map(ord, args.mode)))
    records = []
    block_records = []
    for block in range(blocks):
        order = list(lanes)
        if args.mode == "cold" and block >= 3:
            order.remove("persistent-compile")
        rng.shuffle(order)
        block_record = {"block": block, "order": order, "telemetry_before": telemetry()}
        for lane in order:
            overrides = None
            if args.mode == "cold":
                cache = args.output.parent / "cold-cache" / f"{lane}-n{args.size}-s{args.steps}-b{block}"
                overrides = {"TORCHINDUCTOR_CACHE_DIR": str(cache)}
            record = run_lane(
                manifest, lane, args.mode, args.size, args.steps,
                warmups=warmups, repetitions=per_worker,
                environment_override=overrides,
            )
            record["block"] = block
            records.append(record)
            with checkpoint.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps({
                "block": block, "lane": lane, "success": record["success"],
                "external_seconds": record.get("external_seconds"),
                "observations": record.get("observations"),
            }, sort_keys=True), flush=True)
        block_record["telemetry_after"] = telemetry()
        block_records.append(block_record)

    report = {
        "schema": "gradflow-dveb-abi-bakeoff-timing-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()), "mode": args.mode,
        "size": args.size, "steps": args.steps, "seed": args.seed,
        "lanes": lanes, "blocks": block_records, "records": records,
        "summary": summarize(records, args.mode),
        "environment": {
            "platform": platform.platform(), "python": sys.version,
            "cpu": command_output(["lscpu"]),
            "governor": command_output([
                "sh", "-c", "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do test -r \"$f\" && cat \"$f\"; done",
            ]),
            "gpu": command_output([
                "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    checkpoint.unlink()
    print(json.dumps({"output": str(args.output), "summary": report["summary"]}, sort_keys=True))
    if any(not bool(record.get("success")) for record in records) and args.mode != "capacity":
        raise SystemExit("a counted lane failed")


if __name__ == "__main__":
    main()
