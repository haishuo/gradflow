#!/usr/bin/env python3
"""Orchestrate the frozen A2 S1/S2/E1 worker matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("benchmark_worker.py")


def configurations() -> list[dict[str, Any]]:
    records: dict[tuple[Any, ...], dict[str, Any]] = {}

    def add(subject: str, order: int, dtype: str, dimensions: int, n: int, role: str) -> None:
        key = (subject, order, dtype, dimensions, n)
        if key in records:
            records[key]["roles"].append(role)
        else:
            records[key] = {
                "subject": subject,
                "order": order,
                "dtype": dtype,
                "dimensions": dimensions,
                "n": n,
                "roles": [role],
            }

    for order in (5, 7, 9, 11, 13, 15):
        for dtype in ("float32", "float64"):
            add("scalar", order, dtype, 1, 8192, "S1")
            add("scalar", order, dtype, 3, 64, "S1")
    for order in (5, 15):
        for n in (128, 512, 2048, 8192, 32768):
            add("scalar", order, "float32", 1, n, "S2")
        for n in (16, 32, 64, 96):
            add("scalar", order, "float32", 3, n, "S2")
    for order in (5, 11, 15):
        for dtype in ("float32", "float64"):
            add("characteristic", order, dtype, 3, 32, "E1")
    for order in (5, 15):
        add("characteristic", order, "float32", 3, 64, "E1_scale")
    return list(records.values())


def key(configuration: dict[str, Any], device: str) -> str:
    return "_".join(
        (
            configuration["subject"],
            f"o{configuration['order']}",
            configuration["dtype"],
            f"d{configuration['dimensions']}",
            f"n{configuration['n']}",
            device,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.exists() and not arguments.resume:
        raise SystemExit(f"refusing existing output: {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    raw = arguments.output.parent / "raw"
    if arguments.resume:
        if not arguments.output.exists():
            raise SystemExit("cannot resume a missing campaign")
        document = json.loads(arguments.output.read_text())
        if document["schema"] != "gradflow-academic-a2-campaign-v1":
            raise SystemExit("refusing to resume an unknown campaign schema")
        if document["configurations"] != configurations():
            raise SystemExit("frozen configuration matrix changed")
        raw.mkdir(exist_ok=True)
        document["complete"] = False
        document.pop("completed_utc", None)
    else:
        raw.mkdir(exist_ok=False)
        document: dict[str, Any] = {
            "schema": "gradflow-academic-a2-campaign-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "complete": False,
            "protocol_commit": "6464c50",
            "configurations": configurations(),
            "workers": {},
            "environment": {
                "platform": platform.platform(),
                "python": sys.version,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "canonical_source_changed": False,
        }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    excluded_unregistered = []
    for identifier, worker in document["workers"].items():
        if "E1_scale" in worker["configuration"]["roles"] and worker["device"] == "cpu":
            worker["protocol_eligible"] = False
            worker["exclusion_reason"] = (
                "unregistered CPU dispatch for a CUDA-only E1_scale configuration"
            )
            excluded_unregistered.append(identifier)
        else:
            worker["protocol_eligible"] = True
    document["excluded_unregistered_workers"] = excluded_unregistered
    for configuration in document["configurations"]:
        devices = ("cuda",) if "E1_scale" in configuration["roles"] else ("cpu", "cuda")
        for device in devices:
            identifier = key(configuration, device)
            if identifier in document["workers"]:
                print(f"{identifier}: already recorded", flush=True)
                continue
            command = (
                sys.executable,
                str(WORKER),
                "--subject",
                configuration["subject"],
                "--order",
                str(configuration["order"]),
                "--dtype",
                configuration["dtype"],
                "--dimensions",
                str(configuration["dimensions"]),
                "--size",
                str(configuration["n"]),
                "--device",
                device,
            )
            started = datetime.now(timezone.utc).isoformat()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            stdout_path = raw / f"{identifier}.stdout"
            stderr_path = raw / f"{identifier}.stderr"
            stdout_path.write_text(completed.stdout)
            stderr_path.write_text(completed.stderr)
            try:
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                payload = None
            document["workers"][identifier] = {
                "configuration": configuration,
                "device": device,
                "started_utc": started,
                "returncode": completed.returncode,
                "record": payload,
                "stdout": str(stdout_path.relative_to(arguments.output.parent)),
                "stderr": str(stderr_path.relative_to(arguments.output.parent)),
                "protocol_eligible": True,
            }
            arguments.output.write_text(json.dumps(document, indent=2) + "\n")
            status = payload.get("status") if isinstance(payload, dict) else "no-record"
            print(f"{identifier}: rc={completed.returncode} status={status}", flush=True)
    document["complete"] = True
    document["completed_utc"] = datetime.now(timezone.utc).isoformat()
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
