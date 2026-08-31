#!/usr/bin/env python3
"""Run the frozen U4-F batched-line qualification and resident regime map."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradflow import weno5_rhs


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DRIVER = HERE / "adapter" / "dveb_u4f_batch_driver.cpp"
PYTORCH_WORKER = HERE / "pytorch_batch_worker.py"
PROTOCOL_COMMIT = "ef1ac91f1d0c3ddbaa59c4e8b9f6b4eef9685195"
AMENDMENT_COMMIT = "de11c8171d54fb34b1de848947bf31cc09b01f99"
LIBRARY_SHA256 = "9ff9172b1ac712b8bc97ca9523fd114b2637e5d7825259371ba9850459168443"
SIZE = 8192
BATCHES = (1, 4, 16, 64, 256, 1024)
LANES = ("dveb_native", "pytorch_inductor")
DEVICES = ("cpu", "cuda")
WORKERS = 6
WARMUPS = 5
SAMPLES = 20
BOOTSTRAPS = 20_000
SEED = 20260831
MAXIMUM_LIMIT = 5.0e-11
RMS_LIMIT = 5.0e-12
THERMAL_STOP_C = 80.0


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def execute(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    commands: list[str],
    timeout: int = 3600,
) -> tuple[subprocess.CompletedProcess[str], float]:
    commands.append(f"(cd {shlex.quote(str(cwd))} && {shlex.join(command)})")
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return completed, time.perf_counter() - started


def require(completed: subprocess.CompletedProcess[str], label: str) -> None:
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed ({completed.returncode})\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def stats(values: list[float]) -> dict[str, Any]:
    median = statistics.median(values)
    return {
        "count": len(values),
        "values": values,
        "minimum": min(values),
        "q1": quantile(values, 0.25),
        "median": median,
        "q3": quantile(values, 0.75),
        "maximum": max(values),
        "mean": statistics.mean(values),
        "median_absolute_deviation": statistics.median(
            abs(value - median) for value in values
        ),
        "sample_standard_deviation": statistics.stdev(values),
    }


def compare(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    difference = candidate - reference
    scale = max(float(np.max(np.abs(reference))), 1.0)
    maximum = float(np.max(np.abs(difference))) / scale
    rms = float(np.sqrt(np.mean(difference * difference))) / scale
    finite = bool(np.all(np.isfinite(candidate)))
    row_sums = np.sum(candidate, axis=1, dtype=np.float64)
    row_absolute = np.sum(np.abs(candidate), axis=1, dtype=np.float64)
    row_bounds = 32.0 * np.finfo(np.float64).eps * row_absolute
    conservation = np.abs(row_sums) <= row_bounds
    return {
        "maximum_normalized": maximum,
        "rms_normalized": rms,
        "finite": finite,
        "row_conservation": {
            "rows": int(candidate.shape[0]),
            "passed_rows": int(np.count_nonzero(conservation)),
            "maximum_absolute_sum": float(np.max(np.abs(row_sums))),
            "maximum_bound": float(np.max(row_bounds)),
            "passed": bool(np.all(conservation)),
        },
        "passed": bool(
            finite
            and np.all(conservation)
            and maximum <= MAXIMUM_LIMIT
            and rms <= RMS_LIMIT
        ),
    }


def state(batch: int, frozen_row: np.ndarray) -> np.ndarray:
    result = np.empty((batch, SIZE), dtype=np.float64)
    result[0] = frozen_row
    x = np.arange(SIZE, dtype=np.float64) / SIZE
    for b in range(1, batch):
        phase = (b % 127) / 127.0
        result[b] = (
            0.4
            + np.sin(2.0 * np.pi * (37.0 * x + phase))
            + 0.1 * np.cos(2.0 * np.pi * (91.0 * x + 3.0 * phase))
        )
    return result


def canonical(values: np.ndarray, frozen_rhs: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(values.copy())
    result = weno5_rhs(tensor, 1.0 / SIZE, lambda value: value, alpha=1.0)
    array = result.detach().numpy()
    if not compare(array[:1], frozen_rhs.reshape(1, SIZE))["passed"]:
        raise RuntimeError("U4-F row zero no longer agrees with frozen U4-E RHS")
    array[0] = frozen_rhs
    return array


def parse_json(stdout: str) -> dict[str, Any]:
    return json.loads(stdout.strip().splitlines()[-1])


def parse_samples(stdout: str) -> list[float]:
    values = [
        float(line.split("U4F_SAMPLE ", 1)[1])
        for line in stdout.splitlines()
        if line.startswith("U4F_SAMPLE ")
    ]
    if len(values) != SAMPLES:
        raise RuntimeError(f"expected {SAMPLES} U4-F samples, found {len(values)}")
    return values


def parse_policy(stdout: str, kind: str) -> dict[str, int | str]:
    prefix = f"U4F_{kind} "
    rows = [line for line in stdout.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise RuntimeError(f"expected one {prefix.strip()} row, found {len(rows)}")
    result: dict[str, int | str] = {}
    for item in rows[0].split()[1:]:
        key, value = item.split("=", 1)
        result[key] = value if key == "target" else int(value)
    return result


def validate_policy(
    query: dict[str, Any], run: dict[str, Any], *, batch: int, device: str
) -> None:
    for key in ("target", "cpu_loop", "cuda_block", "reuse", "launches",
                "scratch_bytes", "elements"):
        if query[key] != run[key]:
            raise RuntimeError(f"DVEB query/run disagreement for {key}")
    if query["target"] != device or query["synchronized"] != 0:
        raise RuntimeError("DVEB query target/synchronization contract failed")
    if run["synchronized"] != (1 if device == "cpu" else 0):
        raise RuntimeError("DVEB run synchronization contract failed")
    if query["elements"] != batch * SIZE:
        raise RuntimeError("DVEB elements-written contract failed")
    if query["reuse"] == 2:
        expected_scratch = 8 * batch * (SIZE + 6)
        if query["scratch_bytes"] != expected_scratch or query["launches"] != 2:
            raise RuntimeError("DVEB materialized schedule metadata failed")
    elif query["reuse"] == 1:
        if query["scratch_bytes"] != 0 or query["launches"] != 1:
            raise RuntimeError("DVEB recompute schedule metadata failed")
    else:
        raise RuntimeError("DVEB automatic reuse result is unknown")


def telemetry() -> dict[str, Any]:
    query = (
        "timestamp,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,"
        "power.limit,utilization.gpu,memory.used,clocks_throttle_reasons.active"
    )
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    result = {
        "timestamp": values[0],
        "temperature_c": float(values[1]),
        "pstate": values[2],
        "sm_clock_mhz": float(values[3]),
        "memory_clock_mhz": float(values[4]),
        "power_w": float(values[5]),
        "power_limit_w": float(values[6]),
        "utilization_percent": float(values[7]),
        "memory_used_mib": float(values[8]),
        "active_throttle_reasons": values[9],
    }
    if result["temperature_c"] >= THERMAL_STOP_C:
        raise RuntimeError(f"U4-F thermal stop at {result['temperature_c']} C")
    if result["active_throttle_reasons"] != "0x0000000000000000":
        raise RuntimeError("U4-F observed an active GPU throttle reason")
    return result


def analyze(records: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    medians = {
        lane: [statistics.median(row["samples_milliseconds"]) for row in rows]
        for lane, rows in records.items()
    }
    ratios = [
        pytorch / dveb
        for pytorch, dveb in zip(medians["pytorch_inductor"], medians["dveb_native"])
    ]
    generator = random.Random(seed)
    bootstrapped = []
    for _ in range(BOOTSTRAPS):
        sample = [ratios[generator.randrange(len(ratios))] for _ in ratios]
        bootstrapped.append(statistics.median(sample))
    interval = [quantile(bootstrapped, 0.025), quantile(bootstrapped, 0.975)]
    median = statistics.median(ratios)
    if median < 0.95 and interval[1] < 1.0:
        decision = "pytorch_inductor_win"
    elif median > 1.05 and interval[0] > 1.0:
        decision = "dveb_native_win"
    else:
        decision = "unresolved"
    return {
        "lanes": {
            lane: {
                "all_observations": stats(
                    [sample for row in rows for sample in row["samples_milliseconds"]]
                ),
                "worker_medians": stats(medians[lane]),
                "median_points_per_second": (
                    SIZE * records[lane][0]["batch"]
                    / (statistics.median(medians[lane]) / 1000.0)
                ),
            }
            for lane, rows in records.items()
        },
        "paired_worker_median_ratio_pytorch_over_dveb": {
            **stats(ratios),
            "bootstrap_median_95_ci": interval,
            "decision": decision,
        },
        "decision": decision,
    }


def write_checksums(directory: Path) -> None:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (directory / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.relative_to(directory)}\n" for path in files)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    handoff = args.handoff_dir.resolve()
    cuda = args.cuda_root.resolve()
    work = args.work_root.resolve()
    evidence = args.evidence_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    raw = evidence / "raw"
    raw.mkdir()
    commands = [
        f"(cd {shlex.quote(str(Path.cwd()))} && "
        f"{shlex.join([sys.executable, *sys.argv])})"
    ]

    for frozen_commit in (PROTOCOL_COMMIT, AMENDMENT_COMMIT):
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", frozen_commit, "HEAD"], cwd=ROOT
        ).returncode != 0:
            raise RuntimeError(f"U4-F frozen commit is not an ancestor: {frozen_commit}")
    library = handoff / "weno5_schedule_abi_v1.so"
    header = handoff / "weno5_schedule_abi_v1.h"
    if digest(library) != LIBRARY_SHA256 or not header.is_file():
        raise RuntimeError("U4-F DVEB handoff identity mismatch")

    native_env = os.environ.copy()
    native_env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "LD_LIBRARY_PATH": os.pathsep.join(
                [str(handoff), str(cuda / "lib64"), os.environ.get("LD_LIBRARY_PATH", "")]
            ),
        }
    )
    pytorch_env = os.environ.copy()
    pytorch_env["PYTHONPATH"] = str(ROOT / "src")
    pytorch_env["TORCHINDUCTOR_CACHE_DIR"] = str(work / "torchinductor_cache")

    executable = work / "u4f_dveb_batch"
    completed, adapter_build_seconds = execute(
        [
            "g++", "-O3", "-std=c++17", "-fopenmp", "-I", str(handoff),
            str(DRIVER), str(library), "-L", str(cuda / "lib64"), "-lcudart",
            f"-Wl,-rpath,{handoff}", f"-Wl,-rpath,{cuda / 'lib64'}",
            "-o", str(executable),
        ],
        cwd=work,
        env=native_env,
        commands=commands,
    )
    require(completed, "U4-F adapter build")
    (raw / "adapter_build.stdout").write_text(completed.stdout)
    (raw / "adapter_build.stderr").write_text(completed.stderr)

    u4c_arrays = ROOT / "experiments/academic_u4c/evidence/u4c_c2_20260830/qualification_arrays"
    frozen_state_path = u4c_arrays / "n8192_state.bin"
    frozen_rhs_path = u4c_arrays / "n8192_canonical.bin"
    frozen_state = np.fromfile(frozen_state_path, dtype=np.float64)
    frozen_rhs = np.fromfile(frozen_rhs_path, dtype=np.float64)
    if frozen_state.shape != (SIZE,) or frozen_rhs.shape != (SIZE,):
        raise RuntimeError("U4-F frozen U4-E anchor arrays are unavailable")

    campaign: dict[str, Any] = {
        "schema": "gradflow.academic_u4f.campaign.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_amendment_commit": AMENDMENT_COMMIT,
        "size": SIZE,
        "batches": list(BATCHES),
        "bounds": {"maximum_normalized": MAXIMUM_LIMIT, "rms_normalized": RMS_LIMIT},
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cpu_threads": 1,
        },
        "sources": {
            "dveb_library_sha256": digest(library),
            "dveb_header_sha256": digest(header),
            "adapter_sha256": digest(DRIVER),
            "pytorch_worker_sha256": digest(PYTORCH_WORKER),
            "frozen_state_sha256": digest(frozen_state_path),
            "frozen_rhs_sha256": digest(frozen_rhs_path),
        },
        "preparation": {
            "adapter_build_seconds": adapter_build_seconds,
            "adapter_executable_sha256": digest(executable),
        },
        "cells": {},
    }

    order_rng = random.Random(SEED)
    for batch in BATCHES:
        print(f"U4-F B={batch}: qualification", flush=True)
        batch_work = work / f"b{batch}"
        batch_work.mkdir()
        input_array = state(batch, frozen_state)
        reference = canonical(input_array, frozen_rhs)
        input_path = batch_work / "input.bin"
        reference_path = batch_work / "canonical.bin"
        input_array.tofile(input_path)
        reference.tofile(reference_path)
        candidates: dict[str, np.ndarray] = {}
        qualification: dict[str, Any] = {}

        for device in DEVICES:
            for lane in LANES:
                label = f"{lane}_{device}"
                output_path = batch_work / f"{label}.bin"
                if lane == "dveb_native":
                    command = [
                        str(executable), "--size", str(SIZE), "--batch", str(batch),
                        "--backend", device, "--mode", "qualify", "--input",
                        str(input_path), "--output", str(output_path),
                    ]
                    completed, wall = execute(
                        command, cwd=work, env=native_env, commands=commands
                    )
                else:
                    command = [
                        sys.executable, str(PYTORCH_WORKER), "--size", str(SIZE),
                        "--batch", str(batch), "--device", device, "--mode",
                        "qualify", "--input", str(input_path), "--output",
                        str(output_path),
                    ]
                    completed, wall = execute(
                        command, cwd=ROOT, env=pytorch_env, commands=commands
                    )
                (raw / f"b{batch}_{label}_qualification.stdout").write_text(completed.stdout)
                (raw / f"b{batch}_{label}_qualification.stderr").write_text(completed.stderr)
                if completed.returncode != 0:
                    qualification[label] = {
                        "status": "execution_failed",
                        "passed": False,
                        "process_seconds": wall,
                        "returncode": completed.returncode,
                        "stderr_tail": completed.stderr[-4000:],
                    }
                    continue
                values = np.fromfile(output_path, dtype=np.float64)
                if values.shape != (batch * SIZE,):
                    raise RuntimeError(f"U4-F B={batch} {label} output shape mismatch")
                values = values.reshape(batch, SIZE)
                candidates[label] = values
                record = {
                    **compare(values, reference),
                    "status": "qualified",
                    "input_sha256": digest(input_path),
                    "output_sha256": digest(output_path),
                    "canonical_sha256": digest(reference_path),
                    "process_seconds": wall,
                }
                if lane == "dveb_native":
                    query = parse_policy(completed.stdout, "QUERY")
                    run = parse_policy(completed.stdout, "RUN")
                    validate_policy(query, run, batch=batch, device=device)
                    record.update({"query": query, "run": run})
                else:
                    worker = parse_json(completed.stdout)
                    if worker["graph"] != {"unique_graphs": 1, "graph_break_count": 0}:
                        record["passed"] = False
                        record["status"] = "graph_gate_failed"
                    record["worker"] = worker
                qualification[label] = record

        for lane in LANES:
            cpu_key = f"{lane}_cpu"
            cuda_key = f"{lane}_cuda"
            if cpu_key in candidates and cuda_key in candidates:
                agreement = {
                    **compare(candidates[cuda_key], candidates[cpu_key]),
                    "status": "qualified",
                }
            else:
                agreement = {"status": "unavailable", "passed": False}
            qualification[f"{lane}_cpu_cuda"] = agreement
        admitted = {
            device: bool(
                all(
                    qualification[f"{lane}_{device}"]["passed"]
                    for lane in LANES
                )
            )
            for device in DEVICES
        }
        cell: dict[str, Any] = {
            "batch": batch,
            "points": batch * SIZE,
            "input_sha256": digest(input_path),
            "canonical_sha256": digest(reference_path),
            "qualification": qualification,
            "admitted": admitted,
        }
        campaign["cells"][str(batch)] = cell
        cell["status"] = {
            device: "timed" if admitted[device] else "correctness_excluded"
            for device in DEVICES
        }
        cell["resident"] = {}
        for device in DEVICES:
            if not admitted[device]:
                print(f"U4-F B={batch} {device}: correctness excluded", flush=True)
                continue
            records = {lane: [] for lane in LANES}
            blocks = []
            for worker_index in range(WORKERS):
                order = list(LANES)
                order_rng.shuffle(order)
                block = {"worker": worker_index, "order": order, "records": {}}
                for lane in order:
                    before = telemetry() if device == "cuda" else None
                    if lane == "dveb_native":
                        command = [
                            str(executable), "--size", str(SIZE), "--batch", str(batch),
                            "--backend", device, "--mode", "resident", "--input",
                            str(input_path), "--warmups", str(WARMUPS), "--samples",
                            str(SAMPLES),
                        ]
                        completed, wall = execute(
                            command, cwd=work, env=native_env, commands=commands
                        )
                        require(completed, f"U4-F B={batch} {device} {lane} worker")
                        query = parse_policy(completed.stdout, "QUERY")
                        run = parse_policy(completed.stdout, "RUN")
                        validate_policy(query, run, batch=batch, device=device)
                        record = {
                            "batch": batch,
                            "samples_milliseconds": parse_samples(completed.stdout),
                            "query": query,
                            "run": run,
                            "process_seconds": wall,
                        }
                    else:
                        command = [
                            sys.executable, str(PYTORCH_WORKER), "--size", str(SIZE),
                            "--batch", str(batch), "--device", device, "--mode",
                            "resident", "--input", str(input_path),
                        ]
                        completed, wall = execute(
                            command, cwd=ROOT, env=pytorch_env, commands=commands
                        )
                        require(completed, f"U4-F B={batch} {device} {lane} worker")
                        record = parse_json(completed.stdout)
                        if len(record["samples_milliseconds"]) != SAMPLES:
                            raise RuntimeError("U4-F PyTorch sample count failed")
                        if record["graph"] != {"unique_graphs": 1, "graph_break_count": 0}:
                            raise RuntimeError("U4-F PyTorch resident graph gate failed")
                        record["process_seconds"] = wall
                    after = telemetry() if device == "cuda" else None
                    if before is not None:
                        record["telemetry_before"] = before
                        record["telemetry_after"] = after
                    stdout_path = raw / f"b{batch}_{device}_w{worker_index}_{lane}.stdout"
                    stderr_path = raw / f"b{batch}_{device}_w{worker_index}_{lane}.stderr"
                    stdout_path.write_text(completed.stdout)
                    stderr_path.write_text(completed.stderr)
                    record["stdout"] = str(stdout_path.relative_to(evidence))
                    record["stderr"] = str(stderr_path.relative_to(evidence))
                    records[lane].append(record)
                    block["records"][lane] = {
                        "stdout": record["stdout"], "stderr": record["stderr"]
                    }
                blocks.append(block)
            cell["resident"][device] = {
                "workers_per_lane": WORKERS,
                "warmups_per_worker": WARMUPS,
                "samples_per_worker": SAMPLES,
                "randomized_blocks": blocks,
                "worker_records": records,
                "analysis": analyze(
                    records, SEED + batch + (1_000_000 if device == "cuda" else 0)
                ),
            }
            decision = cell["resident"][device]["analysis"]["decision"]
            print(f"U4-F B={batch} {device}: {decision}", flush=True)

    campaign["complete"] = True
    campaign["completed_utc"] = datetime.now(timezone.utc).isoformat()
    campaign_path = evidence / "campaign.json"
    campaign_path.write_text(json.dumps(campaign, indent=2) + "\n")
    (evidence / "COMMANDS.txt").write_text("\n".join(commands) + "\n")
    write_checksums(evidence)
    print(json.dumps({
        "complete": True,
        "decisions": {
            batch: {
                device: campaign["cells"][str(batch)].get("resident", {})
                .get(device, {}).get("analysis", {}).get("decision", "excluded")
                for device in DEVICES
            }
            for batch in BATCHES
        },
    }, indent=2))


if __name__ == "__main__":
    main()
