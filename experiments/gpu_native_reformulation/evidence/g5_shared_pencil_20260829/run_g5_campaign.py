#!/usr/bin/env python3
"""Run the frozen G5 shared-pencil three-lane performance campaign."""

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


ROOT = Path(__file__).resolve().parents[2]
SIZES = (32, 64, 128, 192, 256)
STEPS = (1, 10)
WARMUPS = 3
REPETITIONS = 30
SEED = 20260829
BOOTSTRAPS = 20_000
MAXIMUM_TEMPERATURE_C = 80
CONTRACTS = {
    "shared_pencil": "p1_shared_pencil_unique_strict_f32_shu_fused_update_v1",
    "global_face_once": "r6q_arbitrary_state_rhs_unique_strict_f32_shu_face_once_v1",
    "cell_recompute": "g4_cell_recompute_interface_cuda_event_v1",
}
LANES = tuple(CONTRACTS)


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
    lane: str, executables: dict[str, Path], input_path: Path, n: int, steps: int
) -> list[str]:
    if lane in ("shared_pencil", "global_face_once"):
        return [
            str(executables[lane]),
            "--size", str(n),
            "--steps", str(steps),
            "--warmups", "0",
            "--repetitions", "1",
            "--input-state", str(input_path),
        ]
    return [
        str(executables[lane]),
        "--target", "cuda",
        "--size", str(n),
        "--steps", str(steps),
        "--input-state", str(input_path),
    ]


def run_lane(
    lane: str, executables: dict[str, Path], input_path: Path, n: int, steps: int
) -> dict[str, Any]:
    command = lane_command(lane, executables, input_path, n, steps)
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    wall_seconds = time.perf_counter() - started
    native = json.loads(completed.stdout)
    if native.get("contract") != CONTRACTS[lane]:
        raise RuntimeError(f"unexpected {lane} contract: {native.get('contract')}")
    if not native.get("finite"):
        raise RuntimeError(f"{lane} reported nonfinite output")
    if native.get("size") != n or native.get("steps") != steps:
        raise RuntimeError(f"{lane} returned the wrong configuration")
    resident = (
        float(native["median_device_ms"]) * 1.0e-3
        if lane != "cell_recompute"
        else float(native["execution_seconds"])
    )
    return {
        "lane": lane,
        "resident_seconds": resident,
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


def analyze_configuration(
    n: int, steps: int, records: list[dict[str, Any]], seed_offset: int
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED + seed_offset)
    endpoints = {}
    for endpoint in ("resident_seconds", "external_fresh_process_seconds"):
        values = {
            lane: [record["lanes"][lane][endpoint] for record in records]
            for lane in LANES
        }
        shared_over_global = [
            shared / global_value
            for shared, global_value in zip(
                values["shared_pencil"], values["global_face_once"], strict=True
            )
        ]
        cell_over_shared = [
            cell / shared
            for cell, shared in zip(
                values["cell_recompute"], values["shared_pencil"], strict=True
            )
        ]
        endpoints[endpoint] = {
            **{
                lane: statistics_record(lane_values, rng)
                for lane, lane_values in values.items()
            },
            "paired_shared_over_global_ratio": statistics_record(
                shared_over_global, rng
            ),
            "paired_cell_over_shared_ratio": statistics_record(
                cell_over_shared, rng
            ),
            "shared_over_global_ratio_of_medians": (
                float(np.median(values["shared_pencil"]))
                / float(np.median(values["global_face_once"]))
            ),
            "cell_over_shared_ratio_of_medians": (
                float(np.median(values["cell_recompute"]))
                / float(np.median(values["shared_pencil"]))
            ),
        }

    order_diagnostics = {}
    for lane in LANES:
        order_diagnostics[lane] = {}
        for position in range(3):
            values = [
                record["lanes"][lane]["resident_seconds"]
                for record in records
                if record["order"][position] == lane
            ]
            order_diagnostics[lane][f"position_{position + 1}"] = {
                "count": len(values),
                "median": float(np.median(values)) if values else None,
            }

    ratio = endpoints["resident_seconds"]["paired_shared_over_global_ratio"]
    cell_ratio = endpoints["resident_seconds"]["paired_cell_over_shared_ratio"]
    primary = n == 128 and steps in (1, 10)
    return {
        "n": n,
        "steps": steps,
        "records": records,
        "endpoints": endpoints,
        "order_diagnostics": order_diagnostics,
        "primary_point": primary,
        "timing_criteria_passed_at_point": bool(
            ratio["median"] <= 1.10
            and ratio["bootstrap_median_95_ci"][1] <= 1.15
            and cell_ratio["median"] > 1.0
        ),
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
    executables = {
        "shared_pencil": Path(arguments.p1).resolve(),
        "global_face_once": Path(arguments.r6q).resolve(),
        "cell_recompute": Path(arguments.cell).resolve(),
    }
    output = Path(arguments.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    campaign_path = output / "campaign.json"
    checkpoint_path = output / "campaign_checkpoint.json"
    if campaign_path.exists() or checkpoint_path.exists():
        raise RuntimeError("G5 campaign output already exists; refusing to overwrite")
    gate = Path(arguments.forward_gate).resolve()
    gate_record = json.loads(gate.read_text())
    if not gate_record.get("passed"):
        raise RuntimeError("G5 forward gate did not pass")

    configurations = []
    checkpoint: dict[str, Any] = {
        "schema": "gradflow-g5-shared-pencil-campaign-checkpoint-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forward_gate_sha256": sha256(gate),
        "configurations": configurations,
    }
    rng = random.Random(SEED)
    with tempfile.TemporaryDirectory(prefix="gradflow-g5-") as directory:
        scratch = Path(directory)
        for config_index, (n, steps) in enumerate(
            (item for n in SIZES for item in ((n, 1), (n, 10)))
        ):
            input_path = scratch / f"vortex_n{n}.f32"
            input_hash = generate_vortex(input_path, n)
            for lane in LANES:
                for _ in range(WARMUPS):
                    run_lane(lane, executables, input_path, n, steps)

            triplets = []
            for repetition in range(REPETITIONS):
                order = list(LANES)
                rng.shuffle(order)
                before = telemetry()
                check_temperature(before)
                lanes = {
                    lane: run_lane(lane, executables, input_path, n, steps)
                    for lane in order
                }
                after = telemetry()
                check_temperature(after)
                triplets.append({
                    "repetition": repetition,
                    "order": order,
                    "telemetry_before": before,
                    "lanes": lanes,
                    "telemetry_after": after,
                })
                checkpoint["active"] = {
                    "n": n,
                    "steps": steps,
                    "completed_triplets": repetition + 1,
                }
                write_json(checkpoint_path, checkpoint)

            configuration = analyze_configuration(
                n, steps, triplets, seed_offset=config_index + 1
            )
            configuration["input_sha256"] = input_hash
            configuration["input_bytes"] = input_path.stat().st_size
            configurations.append(configuration)
            checkpoint["active"] = None
            write_json(checkpoint_path, checkpoint)

    primary = [item for item in configurations if item["primary_point"]]
    memory = gate_record["memory"]
    decision = bool(
        len(primary) == 2
        and memory["passed"]
        and all(item["timing_criteria_passed_at_point"] for item in primary)
    )
    result = {
        "schema": "gradflow-g5-shared-pencil-performance-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "speed-memory Pareto characterization of non-admitted P1",
        "candidate_backend_admitted": False,
        "protocol": {
            "sizes": list(SIZES),
            "steps": list(STEPS),
            "warmup_processes_per_lane": WARMUPS,
            "randomized_triplets_per_configuration": REPETITIONS,
            "random_seed": SEED,
            "bootstrap_resamples": BOOTSTRAPS,
            "thermal_stop_c": MAXIMUM_TEMPERATURE_C,
        },
        "artifacts": {
            lane: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for lane, path in executables.items()
        },
        "forward_gate": {"path": str(gate), "sha256": sha256(gate)},
        "machine": machine_record(),
        "configurations": configurations,
        "primary_decision": {
            "points": [[128, 1], [128, 10]],
            "maximum_paired_median_shared_over_global": 1.10,
            "maximum_bootstrap_upper_shared_over_global": 1.15,
            "maximum_memory_ratio": 0.70,
            "memory": memory,
            "individual_timing_results": [
                item["timing_criteria_passed_at_point"] for item in primary
            ],
            "successful_memory_recovery_pareto_result": decision,
            "backend_qualification_implication": False,
        },
    }
    write_json(campaign_path, result)
    checkpoint_path.rename(output / "campaign_checkpoint_complete.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", required=True)
    parser.add_argument("--r6q", required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--forward-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    result = run(parser.parse_args())
    print(json.dumps({
        "schema": result["schema"],
        "primary_decision": result["primary_decision"],
        "resident_ratios": {
            f"n{item['n']}_s{item['steps']}": {
                "shared_over_global": item["endpoints"]["resident_seconds"]
                    ["paired_shared_over_global_ratio"]["median"],
                "cell_over_shared": item["endpoints"]["resident_seconds"]
                    ["paired_cell_over_shared_ratio"]["median"],
            }
            for item in result["configurations"]
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
