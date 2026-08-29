#!/usr/bin/env python3
"""Run the frozen G4 randomized face-once schedule campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIZES = (8, 16, 32, 64, 128, 192, 256)
STEPS = (1, 10)
WARMUPS = 3
REPETITIONS = 30
SEED = 20260829
BOOTSTRAPS = 20_000
MAXIMUM_TEMPERATURE_C = 80
FACE_CONTRACT = "r6q_arbitrary_state_rhs_unique_strict_f32_shu_face_once_v1"
CELL_CONTRACT = "g4_cell_recompute_interface_cuda_event_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def telemetry() -> dict[str, Any]:
    fields = (
        "timestamp,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,"
        "power.limit,utilization.gpu,memory.used"
    )
    output = command_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    )
    values = [item.strip() for item in output.split(",")]
    if len(values) != 9:
        raise RuntimeError(f"unexpected nvidia-smi telemetry: {output}")
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


def lane_command(
    lane: str,
    face: Path,
    cell: Path,
    input_path: Path,
    n: int,
    steps: int,
    output_path: Path | None = None,
) -> list[str]:
    if lane == "face_once":
        command = [
            str(face), "--size", str(n), "--steps", str(steps),
            "--warmups", "0", "--repetitions", "1",
            "--input-state", str(input_path),
        ]
        if output_path is not None:
            command.extend(("--output-state", str(output_path)))
        return command
    command = [
        str(cell), "--target", "cuda", "--size", str(n),
        "--steps", str(steps), "--input-state", str(input_path),
    ]
    if output_path is not None:
        command.extend(("--output", str(output_path)))
    return command


def run_lane(
    lane: str,
    face: Path,
    cell: Path,
    input_path: Path,
    n: int,
    steps: int,
    output_path: Path | None = None,
) -> dict[str, Any]:
    command = lane_command(lane, face, cell, input_path, n, steps, output_path)
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - started
    record = json.loads(completed.stdout)
    expected = FACE_CONTRACT if lane == "face_once" else CELL_CONTRACT
    if record.get("contract") != expected:
        raise RuntimeError(f"unexpected {lane} contract: {record.get('contract')}")
    if not record.get("finite"):
        raise RuntimeError(f"{lane} reported nonfinite output")
    if record.get("size") != n or record.get("steps") != steps:
        raise RuntimeError(f"{lane} returned the wrong configuration")
    resident = (
        float(record["median_device_ms"]) * 1.0e-3
        if lane == "face_once"
        else float(record["execution_seconds"])
    )
    return {
        "lane": lane,
        "resident_seconds": resident,
        "external_fresh_process_seconds": wall_seconds,
        "native": record,
    }


def pressure(state: np.ndarray) -> np.ndarray:
    density = state[0].astype(np.float64)
    momentum = state[1:4].astype(np.float64)
    energy = state[4].astype(np.float64)
    return 0.4 * (energy - 0.5 * np.sum(momentum * momentum, axis=0) / density)


def validate(face: Path, cell: Path, output: Path, scratch: Path) -> dict[str, Any]:
    n = 32
    input_path = scratch / "validity_input_n32.f32"
    input_hash = generate_vortex(input_path, n)
    face_path = output / "validity_face_once_n32_s1.f32"
    cell_path = output / "validity_cell_recompute_n32_s1.f32"
    face_record = run_lane("face_once", face, cell, input_path, n, 1, face_path)
    cell_record = run_lane("cell_recompute", face, cell, input_path, n, 1, cell_path)
    expected_face_bytes = 5 * n**3 * 4
    expected_cell_bytes = 5 * (n + 1) ** 3 * 4
    if face_path.stat().st_size != expected_face_bytes:
        raise RuntimeError("face-once validity output has the wrong length")
    if cell_path.stat().st_size != expected_cell_bytes:
        raise RuntimeError("cell-recompute validity output has the wrong length")
    face_state = np.fromfile(face_path, dtype=np.float32).reshape(5, n, n, n)
    duplicated = np.fromfile(cell_path, dtype=np.float32).reshape(
        5, n + 1, n + 1, n + 1
    )
    cell_state = duplicated[:, :-1, :-1, :-1]
    difference = face_state.astype(np.float64) - cell_state.astype(np.float64)
    results = {}
    for name, state in (("face_once", face_state), ("cell_recompute", cell_state)):
        p = pressure(state)
        results[name] = {
            "finite": bool(np.isfinite(state).all()),
            "minimum_density": float(np.min(state[0])),
            "minimum_pressure": float(np.min(p)),
            "positive": bool(np.min(state[0]) > 0.0 and np.min(p) > 0.0),
        }
    passed = bool(
        all(item["finite"] and item["positive"] for item in results.values())
        and np.max(np.abs(difference)) <= 2.0e-5
    )
    record = {
        "n": n,
        "steps": 1,
        "input_sha256": input_hash,
        "face_output_sha256": sha256(face_path),
        "cell_output_sha256": sha256(cell_path),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
        "health": results,
        "face_run": face_record,
        "cell_run": cell_record,
        "bound": 2.0e-5,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError("G4 pre-timing validity gate failed")
    return record


def statistics_record(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    bootstrap_indices = rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    bootstrap_medians = np.median(array[bootstrap_indices], axis=1)
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
            float(np.quantile(bootstrap_medians, 0.025)),
            float(np.quantile(bootstrap_medians, 0.975)),
        ],
    }


def analyze_configuration(
    n: int, steps: int, records: list[dict[str, Any]], seed_offset: int
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + seed_offset)
    lane_records = {
        lane: [pair["lanes"][lane] for pair in records]
        for lane in ("face_once", "cell_recompute")
    }
    endpoints = {}
    for endpoint in ("resident_seconds", "external_fresh_process_seconds"):
        face_values = [item[endpoint] for item in lane_records["face_once"]]
        cell_values = [item[endpoint] for item in lane_records["cell_recompute"]]
        ratios = [cell / face for cell, face in zip(cell_values, face_values, strict=True)]
        endpoint_record = {
            "face_once": statistics_record(face_values, rng),
            "cell_recompute": statistics_record(cell_values, rng),
            "paired_cell_over_face_ratio": statistics_record(ratios, rng),
            "ratio_of_medians": statistics.median(cell_values)
            / statistics.median(face_values),
        }
        endpoints[endpoint] = endpoint_record

    order_diagnostics = {}
    for lane in ("face_once", "cell_recompute"):
        order_diagnostics[lane] = {}
        for position in (0, 1):
            values = [
                pair["lanes"][lane]["resident_seconds"]
                for pair in records
                if pair["order"][position] == lane
            ]
            order_diagnostics[lane][f"position_{position + 1}"] = {
                "count": len(values),
                "median": float(np.median(values)) if values else None,
            }

    ratio = endpoints["resident_seconds"]["paired_cell_over_face_ratio"]
    primary = n == 128 and steps in (1, 10)
    hypothesis_supported = bool(
        ratio["median"] > 1.10 and ratio["bootstrap_median_95_ci"][0] > 1.0
    )
    return {
        "n": n,
        "steps": steps,
        "records": records,
        "endpoints": endpoints,
        "order_diagnostics": order_diagnostics,
        "primary_point": primary,
        "schedule_hypothesis_supported_at_point": hypothesis_supported,
    }


def machine_record() -> dict[str, Any]:
    commands = {
        "hostname": ["hostname"],
        "cpu": ["lscpu"],
        "gpu": [
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        "nvcc": ["/home/haishuo/cuda-13.0/bin/nvcc", "--version"],
    }
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        **{name: command_output(command) for name, command in commands.items()},
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    face = Path(arguments.face).resolve()
    cell = Path(arguments.cell).resolve()
    output = Path(arguments.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    campaign_path = output / "campaign.json"
    if campaign_path.exists() or (output / "campaign_checkpoint.json").exists():
        raise RuntimeError("G4 campaign output already exists; refusing to overwrite")
    scratch_root = Path(arguments.scratch).resolve() if arguments.scratch else None

    with tempfile.TemporaryDirectory(prefix="gradflow-g4-", dir=scratch_root) as directory:
        scratch = Path(directory)
        validity = validate(face, cell, output, scratch)
        rng = random.Random(SEED)
        configurations = []
        checkpoint = {
            "schema": "gradflow-g4-face-once-campaign-checkpoint-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "validity": validity,
            "configurations": configurations,
        }
        checkpoint_path = output / "campaign_checkpoint.json"

        for config_index, (n, steps) in enumerate(
            (item for n in SIZES for item in ((n, 1), (n, 10)))
        ):
            input_path = scratch / f"vortex_n{n}.f32"
            input_hash = generate_vortex(input_path, n)
            for lane in ("face_once", "cell_recompute"):
                for _ in range(WARMUPS):
                    run_lane(lane, face, cell, input_path, n, steps)

            pairs = []
            for repetition in range(REPETITIONS):
                order = ["face_once", "cell_recompute"]
                rng.shuffle(order)
                before = telemetry()
                check_temperature(before)
                lanes = {}
                for lane in order:
                    lanes[lane] = run_lane(lane, face, cell, input_path, n, steps)
                after = telemetry()
                check_temperature(after)
                pairs.append(
                    {
                        "repetition": repetition,
                        "order": order,
                        "telemetry_before": before,
                        "lanes": lanes,
                        "telemetry_after": after,
                    }
                )
                checkpoint["active"] = {
                    "n": n,
                    "steps": steps,
                    "completed_pairs": repetition + 1,
                }
                write_json(checkpoint_path, checkpoint)

            configuration = analyze_configuration(
                n, steps, pairs, seed_offset=config_index + 1
            )
            configuration["input_sha256"] = input_hash
            configuration["input_bytes"] = input_path.stat().st_size
            configurations.append(configuration)
            checkpoint["active"] = None
            write_json(checkpoint_path, checkpoint)

        primary = [
            item for item in configurations if item["primary_point"]
        ]
        result = {
            "schema": "gradflow-g4-face-once-performance-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "performance characterization of non-admitted R6Q",
            "candidate_backend_admitted": False,
            "protocol": {
                "sizes": list(SIZES),
                "steps": list(STEPS),
                "warmup_processes_per_lane": WARMUPS,
                "paired_repetitions": REPETITIONS,
                "random_seed": SEED,
                "bootstrap_resamples": BOOTSTRAPS,
                "thermal_stop_c": MAXIMUM_TEMPERATURE_C,
            },
            "artifacts": {
                "face_once": {
                    "path": str(face), "sha256": sha256(face),
                    "bytes": face.stat().st_size,
                },
                "cell_recompute": {
                    "path": str(cell), "sha256": sha256(cell),
                    "bytes": cell.stat().st_size,
                },
            },
            "machine": machine_record(),
            "validity": validity,
            "configurations": configurations,
            "primary_decision": {
                "points": [[128, 1], [128, 10]],
                "required_paired_median_ratio": 1.10,
                "required_bootstrap_lower_bound": 1.0,
                "individual_results": [
                    item["schedule_hypothesis_supported_at_point"]
                    for item in primary
                ],
                "schedule_hypothesis_supported": bool(
                    len(primary) == 2
                    and all(
                        item["schedule_hypothesis_supported_at_point"]
                        for item in primary
                    )
                ),
                "backend_qualification_implication": False,
            },
        }
        write_json(campaign_path, result)
        checkpoint_path.rename(output / "campaign_checkpoint_complete.json")
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--face", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scratch")
    arguments = parser.parse_args()
    result = run(arguments)
    summary = {
        "schema": result["schema"],
        "validity_passed": result["validity"]["passed"],
        "primary_decision": result["primary_decision"],
        "resident_ratio_of_medians": {
            f"n{item['n']}_s{item['steps']}":
                item["endpoints"]["resident_seconds"]["ratio_of_medians"]
            for item in result["configurations"]
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
