#!/usr/bin/env python3
"""Run the frozen U4-E E2/E3 three-way timing campaign."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import random
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
U4C = ROOT / "experiments" / "academic_u4c"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(U4C))
sys.path.insert(0, str(HERE))

from run_performance import (  # noqa: E402
    BOOTSTRAPS,
    SAMPLES,
    THERMAL_STOP_C,
    WARMUPS,
    comparison,
    digest,
    execute,
    parse_json,
    quantile,
    require,
    stats,
    telemetry,
    write_checksums,
)
from run_qualification import parse_policy, verify_policy  # noqa: E402


SIZE = 8192
WORKERS = 6
SEED = 20260831
LANES = ("dveb", "opensbli", "gradflow")
PROTOCOL_COMMIT = "4788585cecd3a765e452406e08e4de5788ae7f0b"
E1_COMMIT = "a9e6e947ce79c9581138fcda0e20ea66191c528c"
QUALIFICATION = HERE / "evidence" / "u4e_e1_20260831" / "qualification.json"
U4D_CAMPAIGN = ROOT / "experiments" / "academic_u4d" / "evidence" / "u4d_campaign_20260830" / "campaign.json"
GRADFLOW_WORKER = U4C / "gradflow_worker.py"
AOT_WORKER = U4C / "aot_launch_worker.py"


def parse_samples(stdout: str, marker: str) -> list[float]:
    values = [
        float(line.split(marker, 1)[1].strip())
        for line in stdout.splitlines()
        if marker in line
    ]
    if len(values) != SAMPLES:
        raise RuntimeError(f"expected {SAMPLES} {marker.strip()} samples, found {len(values)}")
    return values


def pairwise_analysis(
    records: dict[str, list[dict[str, Any]]], seed: int
) -> dict[str, Any]:
    medians = {
        lane: [statistics.median(worker["samples_milliseconds"]) for worker in workers]
        for lane, workers in records.items()
    }
    pairs: dict[str, Any] = {}
    wins = {lane: set() for lane in LANES}
    for pair_index, (left, right) in enumerate(itertools.combinations(LANES, 2)):
        ratios = [a / b for a, b in zip(medians[left], medians[right])]
        generator = random.Random(seed + pair_index)
        bootstrapped = []
        for _ in range(BOOTSTRAPS):
            sample = [ratios[generator.randrange(len(ratios))] for _ in ratios]
            bootstrapped.append(statistics.median(sample))
        interval = [quantile(bootstrapped, 0.025), quantile(bootstrapped, 0.975)]
        median = statistics.median(ratios)
        if median < 0.95 and interval[1] < 1.0:
            decision = f"{left}_win"
            wins[left].add(right)
        elif median > 1.05 and interval[0] > 1.0:
            decision = f"{right}_win"
            wins[right].add(left)
        else:
            decision = "unresolved"
        pairs[f"{left}_over_{right}"] = {
            **stats(ratios),
            "bootstrap_median_95_ci": interval,
            "decision": decision,
        }
    overall = [lane for lane, beaten in wins.items() if len(beaten) == len(LANES) - 1]
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
        "paired_worker_median_ratios": pairs,
        "overall_winner": overall[0] if len(overall) == 1 else "unresolved",
    }


def descriptive_pairwise(analysis: dict[str, Any]) -> dict[str, float]:
    return {
        f"median_ratio_{left}_over_{right}": analysis[left]["median"] / analysis[right]["median"]
        for left, right in itertools.combinations(LANES, 2)
    }


def external_checksum(stdout: str, marker: str) -> float:
    for line in stdout.splitlines():
        if marker in line:
            if marker == "U4E_RESULT":
                fields = dict(item.split("=", 1) for item in line.split()[1:])
                if fields["finite"] != "1":
                    raise RuntimeError("DVEB reported a non-finite result")
                return float(fields["checksum"])
            return float(line.split(marker, 1)[1].strip())
    raise RuntimeError(f"missing checksum marker {marker}")


def check_temperature(label: str) -> dict[str, Any]:
    result = telemetry()
    throttle = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=clocks_throttle_reasons.active",
         "--format=csv,noheader"], text=True
    ).strip()
    result["active_throttle_reasons"] = throttle
    if result["temperature_c"] >= THERMAL_STOP_C:
        raise RuntimeError(f"thermal stop {label}: {result['temperature_c']} C")
    if throttle not in {"0x0000000000000000", "0x0000000000000000 "}:
        raise RuntimeError(f"active throttle reason at {label}: {throttle}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--hdf5-root", type=Path, required=True)
    parser.add_argument("--aot-package", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    cuda = args.cuda_root.resolve()
    hdf5 = args.hdf5_root.resolve()
    package = args.aot_package.resolve()
    evidence = args.evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=False)
    raw = evidence / "raw"
    arrays = evidence / "endpoint_arrays"
    raw.mkdir()
    arrays.mkdir()
    commands = [
        f"(cd {shlex.quote(str(Path.cwd()))} && "
        f"{shlex.join([sys.executable, *sys.argv])})"
    ]

    qualification = json.loads(QUALIFICATION.read_text())
    if qualification["decision"] != "all_six_lanes_qualified":
        raise RuntimeError("U4-E E1 did not admit all six lanes")
    artifacts = qualification["artifacts"]
    dveb_executable = Path(artifacts["dveb_executable"])
    opensbli_root = Path(artifacts["opensbli_root"])
    opensbli_executables = {
        "cpu": Path(artifacts["opensbli_cpu_executable"]),
        "cuda": Path(artifacts["opensbli_cuda_executable"]),
    }
    expected_hashes = {
        dveb_executable: artifacts["dveb_executable_sha256"],
        opensbli_executables["cpu"]: artifacts["opensbli_cpu_sha256"],
        opensbli_executables["cuda"]: artifacts["opensbli_cuda_sha256"],
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"qualified artifact missing or changed: {path}")

    u4d = json.loads(U4D_CAMPAIGN.read_text())
    if not package.is_file() or digest(package) != u4d["artifacts"]["aot_package_sha256"]:
        raise RuntimeError("qualified U4-D AOT package missing or changed")

    c2_arrays = U4C / "evidence" / "u4c_c2_20260830" / "qualification_arrays"
    state_path = c2_arrays / f"n{SIZE}_state.bin"
    canonical_path = c2_arrays / f"n{SIZE}_canonical.bin"
    if digest(state_path) != qualification["canonical"]["state_sha256"]:
        raise RuntimeError("frozen state changed after E1")
    if digest(canonical_path) != qualification["canonical"]["rhs_sha256"]:
        raise RuntimeError("canonical RHS changed after E1")
    canonical = np.fromfile(canonical_path, dtype=np.float64)

    native_env = os.environ.copy()
    native_env.update({
        "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
        "LD_LIBRARY_PATH": os.pathsep.join(
            [str(cuda / "lib64"), str(hdf5 / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
        ),
    })
    grad_env = os.environ.copy()
    grad_env["PYTHONPATH"] = str(ROOT / "src")
    grad_env["TORCHINDUCTOR_CACHE_DIR"] = artifacts["torchinductor_cache"]

    def run_lane(
        lane: str, device: str, mode: str, output: Path | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], float, dict[str, Any]]:
        if lane == "dveb":
            command = [
                str(dveb_executable), "--size", str(SIZE), "--backend", device,
                "--mode", mode, "--input", str(state_path), "--warmups", str(WARMUPS),
                "--samples", str(SAMPLES),
            ]
            if output is not None:
                command.extend(["--output", str(output)])
            completed, wall = execute(
                command, cwd=dveb_executable.parent, env=native_env, commands=commands
            )
            require(completed, f"U4-E {lane} {device} {mode}")
            query = parse_policy(completed.stdout, "QUERY")
            run = parse_policy(completed.stdout, "RUN")
            verify_policy(query, run, device)
            record = {
                "samples_milliseconds": (
                    parse_samples(completed.stdout, "U4E_SAMPLE ")
                    if mode in {"resident", "transfer"} else []
                ),
                "query": query, "run": run, "process_wall_seconds": wall,
            }
        elif lane == "opensbli":
            lane_env = native_env | {
                "U4C_MODE": mode, "U4C_WARMUPS": str(WARMUPS),
                "U4C_SAMPLES": str(SAMPLES), "U4C_STATE_PATH": str(state_path),
                "OPS_BLOCK_SIZE_X": "256",
            }
            if output is not None:
                lane_env["U4C_RHS_PATH"] = str(output)
            completed, wall = execute(
                [str(opensbli_executables[device])], cwd=opensbli_root,
                env=lane_env, commands=commands,
            )
            require(completed, f"U4-E {lane} {device} {mode}")
            record = {
                "samples_milliseconds": (
                    parse_samples(completed.stdout, "U4C_SAMPLE ")
                    if mode in {"resident", "transfer"} else []
                ),
                "process_wall_seconds": wall,
            }
        else:
            command = [
                sys.executable, str(GRADFLOW_WORKER), "--size", str(SIZE),
                "--device", device, "--mode", mode, "--input", str(state_path),
            ]
            if output is not None:
                command.extend(["--output", str(output)])
            completed, wall = execute(command, cwd=ROOT, env=grad_env, commands=commands)
            require(completed, f"U4-E {lane} {device} {mode}")
            record = parse_json(completed.stdout)
            if record["graph"] != {"unique_graphs": 1, "graph_break_count": 0}:
                raise RuntimeError(f"U4-E GradFlow {device} graph gate failed")
            if mode in {"resident", "transfer"} and len(record["samples_milliseconds"]) != SAMPLES:
                raise RuntimeError("U4-E GradFlow sample count mismatch")
            record["process_wall_seconds"] = wall
        return completed, wall, record

    resident: dict[str, Any] = {}
    order_rng = random.Random(SEED)
    for device in ("cpu", "cuda"):
        records: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
        blocks = []
        for worker_index in range(WORKERS):
            order = list(LANES)
            order_rng.shuffle(order)
            block = {"worker": worker_index, "order": order, "records": {}}
            for lane in order:
                before = check_temperature(f"before {device} worker") if device == "cuda" else None
                completed, _, record = run_lane(lane, device, "resident")
                if device == "cuda":
                    record["telemetry_before"] = before
                    record["telemetry_after"] = check_temperature(f"after {device} worker")
                stdout = raw / f"resident_{device}_w{worker_index}_{lane}.stdout"
                stderr = raw / f"resident_{device}_w{worker_index}_{lane}.stderr"
                stdout.write_text(completed.stdout)
                stderr.write_text(completed.stderr)
                record["stdout"] = str(stdout.relative_to(evidence))
                record["stderr"] = str(stderr.relative_to(evidence))
                records[lane].append(record)
                block["records"][lane] = {"stdout": record["stdout"], "stderr": record["stderr"]}
            blocks.append(block)
        resident[device] = {
            "workers_per_lane": WORKERS, "warmups_per_worker": WARMUPS,
            "samples_per_worker": SAMPLES, "randomized_blocks": blocks,
            "worker_records": records,
            "analysis": pairwise_analysis(records, SEED + (0 if device == "cpu" else 10_000)),
        }

    transfer_records: dict[str, Any] = {}
    transfer_order = list(LANES)
    random.Random(SEED + 20_000).shuffle(transfer_order)
    for lane in transfer_order:
        before = check_temperature("before transfer worker")
        output = arrays / f"transfer_{lane}_rhs.bin"
        completed, _, record = run_lane(lane, "cuda", "transfer", output)
        record["telemetry_before"] = before
        record["telemetry_after"] = check_temperature("after transfer worker")
        candidate = np.fromfile(output, dtype=np.float64)
        record["correctness"] = comparison(candidate, canonical)
        if not record["correctness"]["passed"]:
            raise RuntimeError(f"U4-E {lane} transfer correctness failed")
        record["rhs_sha256"] = digest(output)
        stdout = raw / f"transfer_{lane}.stdout"
        stderr = raw / f"transfer_{lane}.stderr"
        stdout.write_text(completed.stdout)
        stderr.write_text(completed.stderr)
        record["stdout"] = str(stdout.relative_to(evidence))
        record["stderr"] = str(stderr.relative_to(evidence))
        transfer_records[lane] = record
    transfer_analysis: dict[str, Any] = {
        lane: stats(record["samples_milliseconds"]) for lane, record in transfer_records.items()
    }
    transfer_analysis.update(descriptive_pairwise(transfer_analysis))

    aot_output = arrays / "aot_qualification_rhs.bin"
    aot_qualified, aot_qualification_seconds = execute(
        [sys.executable, str(AOT_WORKER), "--size", str(SIZE), "--input", str(state_path),
         "--package", str(package), "--output", str(aot_output)],
        cwd=ROOT, env=grad_env, commands=commands,
    )
    (raw / "aot_qualification.stdout").write_text(aot_qualified.stdout)
    (raw / "aot_qualification.stderr").write_text(aot_qualified.stderr)
    require(aot_qualified, "U4-E inherited AOT qualification")
    aot_comparison = comparison(np.fromfile(aot_output, dtype=np.float64), canonical)
    if not aot_comparison["passed"]:
        raise RuntimeError("U4-E inherited AOT correctness gate failed")
    aot_admission = {
        "status": "qualified", "process_seconds": aot_qualification_seconds,
        "comparison": aot_comparison, "rhs_sha256": digest(aot_output),
        "worker": parse_json(aot_qualified.stdout),
    }

    launch_lanes = ("dveb", "opensbli", "gradflow_aot")
    launch_records: dict[str, list[dict[str, Any]]] = {lane: [] for lane in launch_lanes}
    launch_blocks = []
    launch_rng = random.Random(SEED + 30_000)
    for repetition in range(3):
        order = list(launch_lanes)
        launch_rng.shuffle(order)
        block = {"repetition": repetition, "order": order}
        for lane in order:
            before = check_temperature("before launch worker")
            if lane in {"dveb", "opensbli"}:
                completed, elapsed, record_from_run = run_lane(lane, "cuda", "launch")
                checksum = external_checksum(
                    completed.stdout, "U4E_RESULT" if lane == "dveb" else "U4C_CHECKSUM "
                )
                record = {
                    "repetition": repetition, "parent_launch_to_answer_seconds": elapsed,
                    "finite_checksum": bool(np.isfinite(checksum)),
                    "checksum_float64": checksum,
                }
                if lane == "dveb":
                    record["query"] = record_from_run["query"]
                    record["run"] = record_from_run["run"]
            else:
                completed, elapsed = execute(
                    [sys.executable, str(AOT_WORKER), "--size", str(SIZE),
                     "--input", str(state_path), "--package", str(package)],
                    cwd=ROOT, env=grad_env, commands=commands,
                )
                require(completed, f"U4-E AOT launch {repetition}")
                worker = parse_json(completed.stdout)
                record = {
                    "repetition": repetition, "parent_launch_to_answer_seconds": elapsed,
                    "finite_checksum": worker["finite"],
                    "checksum_float64": worker["checksum_float64"], "worker": worker,
                }
            record["telemetry_before"] = before
            record["telemetry_after"] = check_temperature("after launch worker")
            stdout = raw / f"launch_r{repetition}_{lane}.stdout"
            stderr = raw / f"launch_r{repetition}_{lane}.stderr"
            stdout.write_text(completed.stdout)
            stderr.write_text(completed.stderr)
            record["stdout"] = str(stdout.relative_to(evidence))
            record["stderr"] = str(stderr.relative_to(evidence))
            if not record["finite_checksum"]:
                raise RuntimeError(f"U4-E {lane} launch returned non-finite data")
            launch_records[lane].append(record)
        launch_blocks.append(block)
    launch_analysis: dict[str, Any] = {
        lane: stats([record["parent_launch_to_answer_seconds"] for record in records])
        for lane, records in launch_records.items()
    }
    for left, right in itertools.combinations(launch_lanes, 2):
        launch_analysis[f"median_ratio_{left}_over_{right}"] = (
            launch_analysis[left]["median"] / launch_analysis[right]["median"]
        )

    u4d_resident = {
        device: {
            lane: u4d["resident"][device]["analysis"]["lanes"][lane]["worker_medians"]["median"]
            for lane in LANES
        }
        for device in ("cpu", "cuda")
    }
    result = {
        "schema": "gradflow.academic_u4e.campaign.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "size": SIZE,
        "reason_single_size": "N=8192 was the sole U4-C correctness-admitted external size",
        "protocol_commit": PROTOCOL_COMMIT, "e1_commit": E1_COMMIT,
        "qualification_record_sha256": digest(QUALIFICATION),
        "environment": {
            "host": platform.node(), "platform": platform.platform(), "python": sys.version,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "cpu_threads": 1,
        },
        "resident": resident,
        "transfer": {
            "order": transfer_order, "warmups": WARMUPS, "samples": SAMPLES,
            "records": transfer_records, "analysis": transfer_analysis,
            "statistical_winner_prohibited": True,
        },
        "preparation": {
            "e1_observations": qualification["preparation"],
            "gradflow_aot_inherited_from_u4d": u4d["preparation"]["gradflow_aot_build"],
        },
        "aot_package_retained_outside_repository": str(package),
        "aot_admission": aot_admission,
        "prepared_launch": {
            "repetitions_per_lane": 3, "randomized_blocks": launch_blocks,
            "records": launch_records, "analysis": launch_analysis,
            "statistical_winner_prohibited": True,
        },
        "u4d_historical_resident_medians_milliseconds": u4d_resident,
        "u4d_to_u4e_cross_campaign_comparison_is_descriptive": True,
        "artifacts": {
            "dveb_executable_sha256": digest(dveb_executable),
            "dveb_library_sha256": artifacts["dveb_library_sha256"],
            "opensbli_cpu_executable_sha256": digest(opensbli_executables["cpu"]),
            "opensbli_cuda_executable_sha256": digest(opensbli_executables["cuda"]),
            "aot_package_sha256": digest(package),
            "frozen_state_sha256": digest(state_path),
            "canonical_rhs_sha256": digest(canonical_path),
            "campaign_harness_sha256": digest(Path(__file__)),
        },
    }
    (evidence / "campaign.json").write_text(json.dumps(result, indent=2) + "\n")
    (evidence / "COMMANDS.txt").write_text("\n".join(commands) + "\n")
    write_checksums(evidence)
    print(json.dumps({
        "resident": {device: value["analysis"] for device, value in resident.items()},
        "transfer": transfer_analysis, "prepared_launch": launch_analysis,
    }, indent=2))


if __name__ == "__main__":
    main()
