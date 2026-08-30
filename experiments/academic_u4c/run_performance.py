#!/usr/bin/env python3
"""Run the frozen U4-C C2 external resident-performance campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shlex
import shutil
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
ADAPTER = ROOT / "experiments" / "academic_u4b" / "adapter" / "opensbli_scalar_u4b.py"
PATCH = ROOT / "experiments" / "academic_u4b" / "adapter" / "opensbli-u4b.patch"
INSTRUMENTER = HERE / "adapter" / "instrument_performance.py"
MAKEFILE = HERE / "adapter" / "Makefile.ops"
GRADFLOW_WORKER = HERE / "gradflow_worker.py"
OPENSBLI_COMMIT = "e37dc377fa9b27d6bfa6e9da2968b96bcd736f1d"
OPENSBLI_TREE = "0ff053443f6b243b2bd42475f98122306151427d"
OPS_COMMIT = "c0af0f124469e5fd856b594a23ff1206c3e9c7a8"
OPS_TREE = "82c3fd0c0b4724c6e8474e16f730e7560845235f"
SIZES = (8192, 131072, 1048576, 8388608)
WORKERS = 6
SAMPLES = 20
WARMUPS = 5
BOOTSTRAPS = 20_000
SEED = 20260830
MAXIMUM_LIMIT = 5.0e-11
RMS_LIMIT = 5.0e-12
THERMAL_STOP_C = 80.0


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def git_value(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


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


def ensure_patch(root: Path, commands: list[str], env: dict[str, str]) -> None:
    reverse = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(PATCH)],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if reverse.returncode == 0:
        commands.append(f"git -C {shlex.quote(str(root))} apply # already applied")
        return
    check, _ = execute(
        ["git", "apply", "--check", str(PATCH)], cwd=root, env=env, commands=commands
    )
    require(check, "OpenSBLI patch check")
    apply, _ = execute(
        ["git", "apply", str(PATCH)], cwd=root, env=env, commands=commands
    )
    require(apply, "OpenSBLI patch")


def canonical_state(size: int) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float64) / size
    return 0.4 + torch.sin(2.0 * math.pi * 37.0 * x) + 0.1 * torch.cos(
        2.0 * math.pi * 91.0 * x
    )


def canonical_problem(size: int) -> tuple[np.ndarray, np.ndarray]:
    values = canonical_state(size)
    residual = (
        weno5_rhs(values, 1.0 / size, lambda value: value, alpha=1.0)
        .detach()
        .numpy()
    )
    return values.numpy(), residual


def conservation(values: np.ndarray) -> dict[str, float | bool]:
    total = float(np.sum(values, dtype=np.float64))
    absolute = float(np.sum(np.abs(values), dtype=np.float64))
    bound = float(32.0 * np.finfo(np.float64).eps * absolute)
    return {"sum": total, "sum_abs": absolute, "bound": bound, "passed": abs(total) <= bound}


def comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    difference = candidate - reference
    scale = max(float(np.max(np.abs(reference))), 1.0)
    maximum = float(np.max(np.abs(difference))) / scale
    rms = float(np.sqrt(np.mean(difference * difference))) / scale
    finite = bool(np.all(np.isfinite(candidate)))
    health = conservation(candidate)
    return {
        "maximum_normalized": maximum,
        "rms_normalized": rms,
        "finite": finite,
        "conservation": health,
        "passed": bool(
            finite
            and health["passed"]
            and maximum <= MAXIMUM_LIMIT
            and rms <= RMS_LIMIT
        ),
    }


def telemetry() -> dict[str, float | str]:
    query = (
        "timestamp,temperature.gpu,pstate,clocks.sm,clocks.mem,power.draw,"
        "power.limit,utilization.gpu,memory.used"
    )
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [value.strip() for value in completed.stdout.strip().split(",")]
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


def parse_json(stdout: str) -> dict[str, Any]:
    return json.loads(stdout.strip().splitlines()[-1])


def parse_external_samples(stdout: str) -> list[float]:
    values = []
    for line in stdout.splitlines():
        marker = "U4C_SAMPLE "
        if marker in line:
            values.append(float(line.split(marker, 1)[1].strip()))
    if len(values) != SAMPLES:
        raise RuntimeError(f"expected {SAMPLES} OpenSBLI samples, found {len(values)}")
    return values


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
        "median_absolute_deviation": statistics.median(abs(value - median) for value in values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def analyze_workers(records: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    medians = {
        lane: [statistics.median(worker["samples_milliseconds"]) for worker in workers]
        for lane, workers in records.items()
    }
    ratios = [left / right for left, right in zip(medians["opensbli"], medians["gradflow"])]
    generator = random.Random(seed)
    bootstrapped = []
    for _ in range(BOOTSTRAPS):
        sample = [ratios[generator.randrange(len(ratios))] for _ in ratios]
        bootstrapped.append(statistics.median(sample))
    interval = [quantile(bootstrapped, 0.025), quantile(bootstrapped, 0.975)]
    ratio_median = statistics.median(ratios)
    if ratio_median > 1.05 and interval[0] > 1.0:
        decision = "gradflow_win"
    elif ratio_median < 0.95 and interval[1] < 1.0:
        decision = "opensbli_win"
    else:
        decision = "unresolved"
    return {
        "lanes": {
            lane: {
                "all_observations": stats(
                    [sample for worker in workers for sample in worker["samples_milliseconds"]]
                ),
                "worker_medians": stats(medians[lane]),
            }
            for lane, workers in records.items()
        },
        "paired_worker_median_ratio_opensbli_over_gradflow": {
            **stats(ratios),
            "bootstrap_median_95_ci": interval,
        },
        "decision": decision,
    }


def write_checksums(directory: Path) -> None:
    paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (directory / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.relative_to(directory)}\n" for path in paths)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensbli-root", type=Path, required=True)
    parser.add_argument("--ops-root", type=Path, required=True)
    parser.add_argument("--sympy-root", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    opensbli = args.opensbli_root.resolve()
    ops = args.ops_root.resolve()
    cuda = args.cuda_root.resolve()
    hdf5 = args.hdf5_root.resolve()
    work = args.work_root.resolve()
    evidence = args.evidence_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    raw = evidence / "raw"
    arrays = evidence / "qualification_arrays"
    raw.mkdir()
    arrays.mkdir()
    commands = [f"(cd {shlex.quote(str(Path.cwd()))} && {shlex.join([sys.executable, *sys.argv])})"]

    if git_value(opensbli, "rev-parse", "HEAD") != OPENSBLI_COMMIT or git_value(opensbli, "rev-parse", "HEAD^{tree}") != OPENSBLI_TREE:
        raise RuntimeError("OpenSBLI source differs from frozen U4-C revision")
    if git_value(ops, "rev-parse", "HEAD") != OPS_COMMIT or git_value(ops, "rev-parse", "HEAD^{tree}") != OPS_TREE:
        raise RuntimeError("OPS source differs from frozen U4-C revision")

    environment = os.environ.copy()
    native_flags = "-Xcompiler=-fPIC -O3 -g -std=c++11 -gencode arch=compute_120,code=sm_120"
    environment.update(
        {
            "OPS_INSTALL_PATH": str(ops / "ops"),
            "OPS_COMPILER": "gnu",
            "CUDA_INSTALL_PATH": str(cuda),
            "HDF5_INSTALL_PATH": str(hdf5),
            "MPICXX": "g++",
            "NVCCFLAGS": native_flags,
            "PYTHONPATH": os.pathsep.join([str(args.sympy_root.resolve()), str(opensbli)]),
            "LD_LIBRARY_PATH": os.pathsep.join([str(cuda / "lib64"), str(hdf5 / "lib"), environment.get("LD_LIBRARY_PATH", "")]),
        }
    )
    hdf5_link = " ".join([f"-L{cuda / 'lib64'}", "-lops_hdf5_seq", f"-L{hdf5 / 'lib'}", "-lhdf5_hl -lhdf5 -lz"])
    ensure_patch(opensbli, commands, environment)

    result: dict[str, Any] = {
        "schema": "gradflow.academic_u4c.performance.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "protocol_commit": "2106a02",
        "c1_commit": "7a1f696",
        "sizes": {},
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_target": "sm_120",
        },
    }
    order_rng = random.Random(SEED)

    for size in SIZES:
        print(f"U4-C N={size}: preparing", flush=True)
        size_dir = work / f"n{size}"
        size_dir.mkdir()
        shutil.copy2(MAKEFILE, size_dir / "Makefile")
        preparation: dict[str, float] = {}
        generated, elapsed = execute(
            [sys.executable, str(ADAPTER), "--case", "state_a", "--size", str(size)],
            cwd=size_dir,
            env=environment,
            commands=commands,
        )
        require(generated, f"N={size} generation")
        preparation["opensbli_symbolic_generation_seconds"] = elapsed
        instrumented, elapsed = execute(
            [sys.executable, str(INSTRUMENTER), "opensbli.cpp", "opensbliblock00_kernels.h"],
            cwd=size_dir,
            env=environment,
            commands=commands,
        )
        require(instrumented, f"N={size} instrumentation")
        preparation["u4c_instrumentation_seconds"] = elapsed
        seq_build, elapsed = execute(
            ["make", "opensbli_seq", f"HDF5_LIB_SEQ={hdf5_link}"],
            cwd=size_dir,
            env=environment,
            commands=commands,
        )
        require(seq_build, f"N={size} sequential build")
        preparation["ops_translation_and_sequential_build_seconds"] = elapsed
        (raw / f"n{size}_seq_build.stdout").write_text(seq_build.stdout)
        (raw / f"n{size}_seq_build.stderr").write_text(seq_build.stderr)
        cuda_build, elapsed = execute(
            [
                "make",
                "opensbli_cuda",
                "NVCC_FLAG_SET=1",
                f"NVCCFLAGS={native_flags}",
                "CXXFLAGS=-O3 -fPIC -Wall -g -std=c++11 -DU4C_CUDA",
                f"HDF5_LIB_SEQ={hdf5_link}",
            ],
            cwd=size_dir,
            env=environment,
            commands=commands,
        )
        require(cuda_build, f"N={size} CUDA build")
        preparation["cuda_build_seconds"] = elapsed
        (raw / f"n{size}_cuda_build.stdout").write_text(cuda_build.stdout)
        (raw / f"n{size}_cuda_build.stderr").write_text(cuda_build.stderr)

        frozen_state, reference = canonical_problem(size)
        state_path = arrays / f"n{size}_state.bin"
        frozen_state.tofile(state_path)
        reference_path = arrays / f"n{size}_canonical.bin"
        reference.tofile(reference_path)
        qualification: dict[str, Any] = {
            "canonical": {
                "sha256": digest(reference_path),
                "state_sha256": digest(state_path),
                "finite": bool(np.all(np.isfinite(reference))),
                "conservation": conservation(reference),
            }
        }
        candidates: dict[str, np.ndarray] = {}
        for lane, binary in (("opensbli_cpu", "opensbli_seq"), ("opensbli_cuda", "opensbli_cuda")):
            path = arrays / f"n{size}_{lane}.bin"
            lane_env = environment | {
                "U4C_MODE": "qualify",
                "U4C_RHS_PATH": str(path),
                "U4C_STATE_PATH": str(state_path),
                "OPS_BLOCK_SIZE_X": "256",
            }
            completed, elapsed = execute([f"./{binary}"], cwd=size_dir, env=lane_env, commands=commands)
            require(completed, f"N={size} {lane} qualification")
            preparation[f"{lane}_qualification_process_seconds"] = elapsed
            (raw / f"n{size}_{lane}_qualification.stdout").write_text(completed.stdout)
            (raw / f"n{size}_{lane}_qualification.stderr").write_text(completed.stderr)
            candidates[lane] = np.fromfile(path, dtype=np.float64)

        grad_env = os.environ.copy()
        grad_env["PYTHONPATH"] = str(ROOT / "src")
        grad_env["TORCHINDUCTOR_CACHE_DIR"] = str(work / "torchinductor_cache")
        for lane, device in (("gradflow_cpu", "cpu"), ("gradflow_cuda", "cuda")):
            path = arrays / f"n{size}_{lane}.bin"
            completed, elapsed = execute(
                [sys.executable, str(GRADFLOW_WORKER), "--size", str(size), "--device", device, "--mode", "qualify", "--input", str(state_path), "--output", str(path)],
                cwd=ROOT,
                env=grad_env,
                commands=commands,
            )
            require(completed, f"N={size} {lane} qualification")
            payload = parse_json(completed.stdout)
            preparation[f"{lane}_qualification_process_seconds"] = elapsed
            preparation[f"{lane}_first_call_seconds"] = payload["first_call_seconds"]
            (raw / f"n{size}_{lane}_qualification.stdout").write_text(completed.stdout)
            (raw / f"n{size}_{lane}_qualification.stderr").write_text(completed.stderr)
            candidates[lane] = np.fromfile(path, dtype=np.float64)
            qualification.setdefault("worker_metadata", {})[lane] = payload

        for lane, candidate in candidates.items():
            if candidate.shape != reference.shape:
                raise RuntimeError(f"N={size} {lane} shape mismatch")
            path = arrays / f"n{size}_{lane}.bin"
            qualification[lane] = {**comparison(candidate, reference), "sha256": digest(path)}
        admitted = bool(
            qualification["canonical"]["finite"]
            and qualification["canonical"]["conservation"]["passed"]
            and all(qualification[lane]["passed"] for lane in candidates)
        )
        qualification["all_lanes_admitted"] = admitted
        if not admitted:
            result["sizes"][str(size)] = {"qualification": qualification, "preparation": preparation, "status": "correctness_excluded"}
            (evidence / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"U4-C N={size}: correctness excluded", flush=True)
            continue
        del candidates, reference

        timing: dict[str, Any] = {}
        for device in ("cpu", "cuda"):
            lane_records: dict[str, list[dict[str, Any]]] = {"opensbli": [], "gradflow": []}
            blocks = []
            for worker_index in range(WORKERS):
                lane_order = ["opensbli", "gradflow"]
                order_rng.shuffle(lane_order)
                block = {"worker": worker_index, "order": lane_order, "records": {}}
                for lane in lane_order:
                    if device == "cuda":
                        before = telemetry()
                        if before["temperature_c"] >= THERMAL_STOP_C:
                            raise RuntimeError("U4-C thermal stop before CUDA worker")
                    else:
                        before = None
                    if lane == "opensbli":
                        binary = "opensbli_cuda" if device == "cuda" else "opensbli_seq"
                        lane_env = environment | {
                            "U4C_MODE": "resident",
                            "U4C_WARMUPS": str(WARMUPS),
                            "U4C_SAMPLES": str(SAMPLES),
                            "U4C_STATE_PATH": str(state_path),
                            "OPS_BLOCK_SIZE_X": "256",
                        }
                        completed, wall = execute([f"./{binary}"], cwd=size_dir, env=lane_env, commands=commands)
                        require(completed, f"N={size} {device} OpenSBLI worker {worker_index}")
                        record = {
                            "samples_milliseconds": parse_external_samples(completed.stdout),
                            "process_wall_seconds": wall,
                            "peak_rss_kib": None,
                        }
                    else:
                        completed, wall = execute(
                            [sys.executable, str(GRADFLOW_WORKER), "--size", str(size), "--device", device, "--mode", "resident", "--input", str(state_path)],
                            cwd=ROOT,
                            env=grad_env,
                            commands=commands,
                        )
                        require(completed, f"N={size} {device} GradFlow worker {worker_index}")
                        record = parse_json(completed.stdout)
                        if record["graph"] != {"unique_graphs": 1, "graph_break_count": 0}:
                            raise RuntimeError(f"N={size} {device} GradFlow graph gate failed")
                        if len(record["samples_milliseconds"]) != SAMPLES:
                            raise RuntimeError("GradFlow worker sample count mismatch")
                        record["process_wall_seconds"] = wall
                    if device == "cuda":
                        after = telemetry()
                        if after["temperature_c"] >= THERMAL_STOP_C:
                            raise RuntimeError("U4-C thermal stop after CUDA worker")
                        record["telemetry_before"] = before
                        record["telemetry_after"] = after
                    stdout_path = raw / f"n{size}_{device}_w{worker_index}_{lane}.stdout"
                    stderr_path = raw / f"n{size}_{device}_w{worker_index}_{lane}.stderr"
                    stdout_path.write_text(completed.stdout)
                    stderr_path.write_text(completed.stderr)
                    lane_records[lane].append(record)
                    block["records"][lane] = {"stdout": str(stdout_path.relative_to(evidence)), "stderr": str(stderr_path.relative_to(evidence))}
                blocks.append(block)
            timing[device] = {
                "workers_per_lane": WORKERS,
                "warmups_per_worker": WARMUPS,
                "samples_per_worker": SAMPLES,
                "randomized_blocks": blocks,
                "worker_records": lane_records,
                "analysis": analyze_workers(lane_records, SEED + size + (1 if device == "cuda" else 0)),
            }

        result["sizes"][str(size)] = {
            "status": "complete",
            "qualification": qualification,
            "preparation": preparation,
            "timing": timing,
            "generated_source_sha256": {
                name: digest(size_dir / name)
                for name in ("opensbli.cpp", "opensbli_ops.cpp", "opensbliblock00_kernels.h")
            },
        }
        (evidence / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"U4-C N={size}: CPU {timing['cpu']['analysis']['decision']}; "
            f"CUDA {timing['cuda']['analysis']['decision']}",
            flush=True,
        )

    result["complete"] = True
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (evidence / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
    (evidence / "COMMANDS.txt").write_text("\n".join(commands) + "\n")
    write_checksums(evidence)


if __name__ == "__main__":
    main()
