#!/usr/bin/env python3
"""Run the frozen G6 exact-math occupancy performance campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time
from typing import Any

import numpy as np


SIZES = (64, 128, 256)
STEPS = (1, 10)
WARMUPS = 3
REPETITIONS = 30
SEED = 20260830
BOOTSTRAPS = 20_000
MAXIMUM_TEMPERATURE_C = 80
FROZEN = "frozen_r6q"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def telemetry() -> dict[str, Any]:
    fields = (
        "timestamp,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,"
        "power.limit,utilization.gpu,memory.used"
    )
    values = [
        item.strip()
        for item in command_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
        ).split(",")
    ]
    if len(values) != 9:
        raise RuntimeError(f"unexpected telemetry fields: {values}")
    return {
        "timestamp": values[0],
        "temperature_c": float(values[1]),
        "pstate": values[2],
        "sm_clock_mhz": float(values[3]),
        "memory_clock_mhz": float(values[4]),
        "power_w": float(values[5]),
        "power_limit_w": float(values[6]),
        "utilization_percent": float(values[7]),
        "memory_used_mib": float(values[8]),
    }


def check_temperature(sample: dict[str, Any]) -> None:
    if sample["temperature_c"] >= MAXIMUM_TEMPERATURE_C:
        raise RuntimeError(
            f"thermal stop at {sample['temperature_c']} C "
            f"(limit {MAXIMUM_TEMPERATURE_C} C)"
        )


def generate_vortex(path: Path, n: int) -> str:
    coordinate = np.arange(n, dtype=np.float64) * (10.0 / n)
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    coefficient = 5.0 / (2.0 * math.pi * math.exp(-0.5))
    radius_squared = (x - 5.0) ** 2 + (y - 5.0) ** 2
    exponential = np.exp(-0.5 * radius_squared)
    u = -coefficient * exponential * (y - 5.0)
    v = coefficient * exponential * (x - 5.0)
    temperature = 1.0 - 0.5 * coefficient**2 * exponential**2 * (0.4 / 1.4)
    pressure = temperature ** (1.4 / 0.4)
    density = pressure / temperature
    energy = pressure / 0.4 + 0.5 * density * (u * u + v * v)
    state = np.memmap(path, mode="w+", dtype=np.float32, shape=(5, n, n, n))
    state[0] = density
    state[1] = density * u
    state[2] = density * v
    state[3] = 0.0
    state[4] = energy
    state.flush()
    del state
    return sha256(path)


def run_lane(
    lane: str,
    executable: Path,
    input_path: Path,
    n: int,
    steps: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [
            str(executable),
            "--size", str(n),
            "--steps", str(steps),
            "--warmups", "0",
            "--repetitions", "1",
            "--input-state", str(input_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wall_seconds = time.perf_counter() - started
    native = json.loads(completed.stdout)
    expected = (
        "r6q_arbitrary_state_rhs_unique_strict_f32_shu_face_once_v1"
        if lane == FROZEN
        else f"g6_r6q_{lane}_v1"
    )
    if native.get("contract") != expected:
        raise RuntimeError(f"unexpected {lane} contract: {native.get('contract')}")
    if not native.get("finite"):
        raise RuntimeError(f"{lane} reported nonfinite output")
    return {
        "resident_seconds": float(native["median_device_ms"]) * 1.0e-3,
        "external_fresh_process_seconds": wall_seconds,
        "native": native,
    }


def statistics_record(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    indices = rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    bootstraps = np.median(array[indices], axis=1)
    return {
        "count": len(values),
        "values": values,
        "median": median,
        "mean": float(np.mean(array)),
        "sample_standard_deviation": float(np.std(array, ddof=1)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "median_absolute_deviation": float(np.median(np.abs(array - median))),
        "bootstrap_median_95_ci": [
            float(np.quantile(bootstraps, 0.025)),
            float(np.quantile(bootstraps, 0.975)),
        ],
    }


def analyze(
    n: int,
    steps: int,
    records: list[dict[str, Any]],
    lanes: tuple[str, ...],
    seed_offset: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + seed_offset)
    endpoints = {}
    for endpoint in ("resident_seconds", "external_fresh_process_seconds"):
        values = {
            lane: [record["lanes"][lane][endpoint] for record in records]
            for lane in lanes
        }
        frozen_values = values[FROZEN]
        candidates = {}
        for lane in lanes:
            if lane == FROZEN:
                continue
            ratios = [
                candidate / frozen
                for candidate, frozen in zip(values[lane], frozen_values, strict=True)
            ]
            candidates[lane] = {
                "lane": statistics_record(values[lane], rng),
                "paired_candidate_over_frozen_ratio": statistics_record(ratios, rng),
                "ratio_of_medians": float(np.median(values[lane]))
                / float(np.median(frozen_values)),
            }
        endpoints[endpoint] = {
            "frozen_r6q": statistics_record(frozen_values, rng),
            "candidates": candidates,
        }
    return {
        "n": n,
        "steps": steps,
        "primary_point": n == 128 and steps in (1, 10),
        "records": records,
        "endpoints": endpoints,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def machine_record() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hostname": command_output(["hostname"]),
        "cpu": command_output(["lscpu"]),
        "gpu": command_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
        "nvcc": command_output(["/home/haishuo/cuda-13.0/bin/nvcc", "--version"]),
    }


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    gate_path = Path(arguments.forward_gate).resolve()
    gate = json.loads(gate_path.read_text())
    if not gate.get("passed"):
        raise RuntimeError("G6 forward gate did not pass")
    candidates = tuple(gate["passing_candidates"])
    lanes = (FROZEN, *candidates)
    build = Path(arguments.build_dir).resolve()
    executables = {
        FROZEN: Path(arguments.frozen_r6q).resolve(),
        **{lane: build / f"gradflow_g6_{lane}" for lane in candidates},
    }
    output = Path(arguments.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    campaign_path = output / "campaign.json"
    checkpoint_path = output / "campaign_checkpoint.json"
    if campaign_path.exists() or checkpoint_path.exists():
        raise RuntimeError("G6 campaign output exists; refusing to overwrite")

    configurations = []
    checkpoint: dict[str, Any] = {
        "schema": "gradflow-g6-occupancy-campaign-checkpoint-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forward_gate_sha256": sha256(gate_path),
        "configurations": configurations,
    }
    order_rng = random.Random(SEED)
    with tempfile.TemporaryDirectory(prefix="gradflow-g6-") as directory:
        scratch = Path(directory)
        for index, (n, steps) in enumerate(
            (item for n in SIZES for item in ((n, 1), (n, 10)))
        ):
            input_path = scratch / f"vortex_n{n}.f32"
            input_hash = generate_vortex(input_path, n)
            for lane in lanes:
                for _ in range(WARMUPS):
                    run_lane(lane, executables[lane], input_path, n, steps)
            blocks = []
            for repetition in range(REPETITIONS):
                order = list(lanes)
                order_rng.shuffle(order)
                before = telemetry()
                check_temperature(before)
                observations = {
                    lane: run_lane(lane, executables[lane], input_path, n, steps)
                    for lane in order
                }
                after = telemetry()
                check_temperature(after)
                blocks.append({
                    "repetition": repetition,
                    "order": order,
                    "telemetry_before": before,
                    "lanes": observations,
                    "telemetry_after": after,
                })
                checkpoint["active"] = {
                    "n": n,
                    "steps": steps,
                    "completed_blocks": repetition + 1,
                }
                write_json(checkpoint_path, checkpoint)
            configuration = analyze(n, steps, blocks, lanes, index + 1)
            configuration["input_sha256"] = input_hash
            configuration["input_bytes"] = input_path.stat().st_size
            configurations.append(configuration)
            checkpoint["active"] = None
            write_json(checkpoint_path, checkpoint)

    primary = [item for item in configurations if item["primary_point"]]
    decisions = {}
    for candidate in candidates:
        ratios = []
        passes = []
        for point in primary:
            ratio = point["endpoints"]["resident_seconds"]["candidates"][candidate][
                "paired_candidate_over_frozen_ratio"
            ]
            ratios.append(ratio["median"])
            passes.append(
                ratio["median"] < 0.95
                and ratio["bootstrap_median_95_ci"][1] < 1.0
            )
        decisions[candidate] = {
            "primary_paired_median_ratios": ratios,
            "primary_geometric_mean_ratio": math.sqrt(ratios[0] * ratios[1]),
            "point_results": passes,
            "meaningful_improvement": all(passes),
        }
    fastest = min(
        candidates,
        key=lambda candidate: (
            decisions[candidate]["primary_geometric_mean_ratio"], candidate
        ),
    )
    result = {
        "schema": "gradflow-g6-occupancy-performance-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_backend_admitted": False,
        "protocol": {
            "sizes": list(SIZES),
            "steps": list(STEPS),
            "warmup_processes_per_lane": WARMUPS,
            "randomized_complete_lane_blocks": REPETITIONS,
            "random_seed": SEED,
            "bootstrap_resamples": BOOTSTRAPS,
            "thermal_stop_c": MAXIMUM_TEMPERATURE_C,
            "lane_order": list(lanes),
        },
        "forward_gate": {"path": str(gate_path), "sha256": sha256(gate_path)},
        "artifacts": {
            lane: {
                "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size
            }
            for lane, path in executables.items()
        },
        "machine": machine_record(),
        "configurations": configurations,
        "primary_decision": {
            "points": [[128, 1], [128, 10]],
            "candidate_results": decisions,
            "fastest_passing_candidate_by_frozen_rule": fastest,
            "any_meaningful_occupancy_improvement": any(
                item["meaningful_improvement"] for item in decisions.values()
            ),
            "backend_qualification_implication": False,
        },
    }
    write_json(campaign_path, result)
    checkpoint_path.rename(output / "campaign_checkpoint_complete.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--frozen-r6q", required=True)
    parser.add_argument("--forward-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    result = run(parser.parse_args())
    print(json.dumps({
        "schema": result["schema"],
        "primary_decision": result["primary_decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
