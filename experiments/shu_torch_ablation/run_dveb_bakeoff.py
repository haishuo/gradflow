#!/usr/bin/env python3
"""Run one randomized, repeated DVEB-inclusive matched configuration."""

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
import sys
import time


EXPERIMENT = Path(__file__).resolve().parent
WORKER = EXPERIMENT / "bakeoff_worker.py"
FORTRAN = EXPERIMENT / "build" / "shu_euler_3d"
PRIMARY_LANES = (
    "fortran",
    "dveb-auto",
    "direct-eager",
    "aot-inductor",
)
DIAGNOSTIC_LANES = (
    "dveb-cpu6",
    "dveb-cuda",
    "ceiling-cpu",
    "ceiling-cuda",
)


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
    raise RuntimeError(f"subprocess emitted no JSON object:\n{text}")


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def run_process(
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
    return completed, time.perf_counter() - started


def run_lane(
    lane: str, size: int, steps: int, manifest: dict[str, object],
    calibration: dict[str, object], native_environment: dict[str, str],
) -> dict[str, object]:
    if lane == "fortran":
        completed, elapsed = run_process(
            [str(FORTRAN)],
            environment=native_environment,
            stdin=f"{size} {size} {size}\n0.1\n{steps}\n1.0e6\n",
        )
        serious = ("IEEE_INVALID", "IEEE_OVERFLOW", "IEEE_DIVIDE_BY_ZERO")
        record: dict[str, object] = {
            "lane": lane,
            "fresh_process_seconds": elapsed,
            "finite": not any(flag in completed.stderr for flag in serious),
        }
    elif lane.startswith("dveb-"):
        binary = Path(manifest["native"]["dveb"]["frozen_copy"])
        command = [str(binary)]
        environment = native_environment.copy()
        if lane == "dveb-auto":
            command += [
                "--model", calibration["model"],
                "--verified-model", calibration["model_sha256"],
                "--endpoint", "cpu-resident", "--explain-placement",
            ]
        else:
            candidate = "cpu_simd[6]" if lane == "dveb-cpu6" else "cuda"
            command += ["--internal-calibration", "--candidate", candidate]
            environment["DVEB_CALIBRATION"] = "1"
        command += ["--size", str(size), "--steps", str(steps)]
        completed, elapsed = run_process(command, environment=environment)
        record = last_json(completed.stdout) if completed.returncode == 0 else {"lane": lane}
        record["fresh_process_seconds"] = elapsed
        record["lane"] = lane
        if lane == "dveb-auto" and "selected=" in completed.stderr:
            record["selected"] = completed.stderr.split("selected=", 1)[1].split()[0]
    elif lane.startswith("ceiling-"):
        family, target = lane.split("-", 1)
        binary = Path(manifest["native"][family]["frozen_copy"])
        completed, elapsed = run_process(
            [str(binary), "--target", target, "--size", str(size),
             "--steps", str(steps)],
            environment=native_environment,
        )
        record = last_json(completed.stdout) if completed.returncode == 0 else {"lane": lane}
        record["fresh_process_seconds"] = elapsed
        record["lane"] = lane
    else:
        worker_lane = "direct-eager" if lane == "direct-eager" else "aot"
        command = [
            sys.executable,
            str(WORKER),
            "--lane",
            worker_lane,
            "--size",
            str(size),
            "--steps",
            str(steps),
        ]
        environment = os.environ.copy()
        if lane == "aot-inductor":
            package = manifest["aot_packages"][str(size)]
            command += ["--package", package["path"]]
            environment["TORCHINDUCTOR_CACHE_DIR"] = package["runtime_cache"]
        completed, elapsed = run_process(command, environment=environment)
        record = last_json(completed.stdout) if completed.returncode == 0 else {"lane": lane}
        record["fresh_process_seconds"] = elapsed
        record["lane"] = lane

    record["size"] = size
    record["steps"] = steps
    record["returncode"] = completed.returncode
    if completed.stderr.strip():
        record["stderr"] = completed.stderr.strip()
    record["success"] = completed.returncode == 0 and bool(record.get("finite", True))
    return record


def summarize(
    records: list[dict[str, object]], lanes: tuple[str, ...]
) -> dict[str, object]:
    summary: dict[str, object] = {}
    endpoints = (
        "fresh_process_seconds",
        "execution_seconds",
        "end_to_host_after_import_seconds",
        "process_seconds_after_main",
    )
    for lane in lanes:
        lane_records = [r for r in records if r["lane"] == lane and r["success"]]
        lane_summary: dict[str, object] = {
            "successes": len(lane_records),
            "failures": sum(1 for r in records if r["lane"] == lane and not r["success"]),
        }
        for endpoint in endpoints:
            values = [float(r[endpoint]) for r in lane_records if endpoint in r]
            if values:
                ordered = sorted(values)
                lane_summary[endpoint] = {
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "minimum": ordered[0],
                    "maximum": ordered[-1],
                    "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
                }
        summary[lane] = lane_summary
    return summary


def telemetry() -> dict[str, str]:
    return {
        "gpu": command_output([
            "nvidia-smi",
            "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,pstate,power.draw,clocks_throttle_reasons.active",
            "--format=csv,noheader,nounits",
        ]),
        "cpu_mhz": command_output([
            "bash", "-lc", "awk '/cpu MHz/ {s+=$4;n++} END {if(n) printf \"%.1f\",s/n}' /proc/cpuinfo",
        ]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing to overwrite result: {arguments.output}")
    if not FORTRAN.is_file():
        raise SystemExit(f"missing Fortran executable: {FORTRAN}")

    manifest = json.loads(arguments.manifest.read_text())
    calibration = json.loads(arguments.calibration.read_text())
    if calibration["artifact_sha256"] != manifest["native"]["dveb"]["sha256"]:
        raise SystemExit("calibration belongs to a different DVEB artifact")
    model = Path(calibration["model"])
    if not model.is_file():
        raise SystemExit("missing DVEB placement model")
    model_lines = model.read_text().splitlines(keepends=True)
    if not model_lines or not model_lines[-1].startswith("model_sha256\t"):
        raise SystemExit("invalid DVEB placement model")
    observed_model_hash = hashlib.sha256("".join(model_lines[:-1]).encode()).hexdigest()
    if observed_model_hash != calibration["model_sha256"]:
        raise SystemExit("DVEB placement model failed its integrity check")
    if str(arguments.size) not in manifest["aot_packages"]:
        raise SystemExit(f"manifest has no AOT package for N={arguments.size}")
    for family in ("dveb", "ceiling"):
        frozen = Path(manifest["native"][family]["frozen_copy"])
        expected = manifest["native"][family]["sha256"]
        if not frozen.is_file() or sha256(frozen) != expected:
            raise SystemExit(f"refusing changed frozen {family} binary")
    package = manifest["aot_packages"][str(arguments.size)]
    if sha256(Path(package["path"])) != package["sha256"]:
        raise SystemExit("refusing changed AOT package")

    native_environment = os.environ.copy()
    native_environment.update({
        "OMP_DYNAMIC": "FALSE",
        "OMP_PROC_BIND": "CLOSE",
        "OMP_PLACES": "CORES",
    })
    lanes = PRIMARY_LANES + DIAGNOSTIC_LANES if arguments.include_diagnostics else PRIMARY_LANES
    warmup_records: list[dict[str, object]] = []
    for lane in lanes:
        for _ in range(arguments.warmups):
            warmup_records.append(run_lane(
                lane, arguments.size, arguments.steps, manifest, calibration,
                native_environment
            ))

    rng = random.Random(arguments.seed + arguments.size * 1000 + arguments.steps)
    records: list[dict[str, object]] = []
    blocks: list[dict[str, object]] = []
    for repetition in range(arguments.repetitions):
        order = list(lanes)
        rng.shuffle(order)
        before = telemetry()
        for lane in order:
            record = run_lane(
                lane, arguments.size, arguments.steps, manifest, calibration,
                native_environment
            )
            record["repetition"] = repetition
            records.append(record)
            print(json.dumps(record), flush=True)
        blocks.append({
            "repetition": repetition,
            "order": order,
            "telemetry_before": before,
            "telemetry_after": telemetry(),
        })

    result = {
        "schema_version": 1,
        "protocol": "DVEB_BAKEOFF_PROTOCOL.md",
        "size": arguments.size,
        "steps": arguments.steps,
        "repetitions": arguments.repetitions,
        "warmups": arguments.warmups,
        "seed": arguments.seed,
        "lanes": lanes,
        "manifest": str(arguments.manifest.resolve()),
        "calibration": str(arguments.calibration.resolve()),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu": command_output(["lscpu"]),
            "governor": command_output([
                "bash", "-lc",
                "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do test -r \"$f\" && cat \"$f\"; done | sort -u",
            ]),
            "gpu": command_output([
                "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]),
            "gfortran": command_output(["gfortran", "--version"]),
            "omp": {key: native_environment[key] for key in
                    ("OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES")},
        },
        "warmup_records": warmup_records,
        "blocks": blocks,
        "records": records,
        "summary": summarize(records, lanes),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
