#!/usr/bin/env python3
"""Run the frozen A2 C1 fresh-process deployment slice."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REPETITIONS = 3


def core_identifier(order: int, dimensions: int, n: int, device: str) -> str:
    return f"scalar_o{order}_float32_d{dimensions}_n{n}_{device}"


def admitted(
    core: dict[str, Any], order: int, dimensions: int, n: int, lane: str
) -> bool:
    device = "cpu" if lane == "cpu_compiled" else "cuda"
    worker = core["workers"][core_identifier(order, dimensions, n, device)]
    if lane == "cpu_compiled":
        return bool(worker["record"]["cpu"]["6"]["correctness"]["compiled"]["admitted"])
    return bool(worker["record"]["correctness"]["compiled"]["admitted"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--aot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing existing output: {arguments.output}")
    core = json.loads(arguments.core.read_text())
    aot = json.loads(arguments.aot.read_text())
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(HERE)))
    document: dict[str, Any] = {
        "schema": "gradflow-academic-a2-deployment-campaign-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "repetitions": REPETITIONS,
        "configurations": [],
    }
    for order in (5, 15):
        for dimensions, n in ((1, 8192), (3, 64)):
            for lane in ("cpu_compiled", "cuda_compiled", "cuda_aot"):
                eligibility = admitted(core, order, dimensions, n, lane)
                package = None
                reason = None
                if lane == "cuda_aot":
                    if dimensions != 3:
                        eligibility = False
                        reason = "P1 did not build a 1-D package"
                    else:
                        build = aot["orders"][str(order)]["build_record"]
                        qualification = aot["orders"][str(order)]["qualification"]
                        eligibility = (
                            eligibility
                            and build["status"] == "complete"
                            and qualification is not None
                        )
                        package = aot["orders"][str(order)][
                            "package_retained_outside_repository"
                        ]
                        if not eligibility:
                            reason = "AOT build/qualification or JIT reference was not eligible"
                elif not eligibility:
                    reason = "corresponding core compiled lane failed correctness"
                configuration: dict[str, Any] = {
                    "order": order,
                    "dimensions": dimensions,
                    "n": n,
                    "lane": lane,
                    "eligible": eligibility,
                    "ineligible_reason": reason,
                    "records": [],
                }
                if eligibility:
                    for repetition in range(REPETITIONS):
                        command = [
                            sys.executable,
                            str(HERE / "deployment_worker.py"),
                            "--order",
                            str(order),
                            "--dimensions",
                            str(dimensions),
                            "--size",
                            str(n),
                            "--lane",
                            lane,
                        ]
                        if package is not None:
                            command.extend(("--package", package))
                        started = time.perf_counter()
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            env=environment,
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=3600,
                        )
                        parent_seconds = time.perf_counter() - started
                        try:
                            payload = json.loads(
                                completed.stdout.strip().splitlines()[-1]
                            )
                        except (IndexError, json.JSONDecodeError):
                            payload = None
                        configuration["records"].append(
                            {
                                "repetition": repetition,
                                "returncode": completed.returncode,
                                "parent_start_to_finish_seconds": parent_seconds,
                                "worker": payload,
                                "stderr": completed.stderr,
                            }
                        )
                    successful = [
                        item
                        for item in configuration["records"]
                        if item["returncode"] == 0
                        and item["worker"] is not None
                        and item["worker"]["finite"]
                    ]
                    if len(successful) == REPETITIONS:
                        values = [
                            item["parent_start_to_finish_seconds"]
                            for item in successful
                        ]
                        configuration["median_parent_start_to_finish_seconds"] = (
                            statistics.median(values)
                        )
                        configuration["all_checksums_identical"] = (
                            len(
                                {
                                    item["worker"]["checksum_float64"]
                                    for item in successful
                                }
                            )
                            == 1
                        )
                    else:
                        configuration["failure"] = (
                            "not all fresh processes returned a finite record"
                        )
                document["configurations"].append(configuration)
                arguments.output.parent.mkdir(parents=True, exist_ok=True)
                arguments.output.write_text(json.dumps(document, indent=2) + "\n")
                print(
                    f"deployment o{order} d{dimensions} {lane}: eligible={eligibility}",
                    flush=True,
                )
    document["complete"] = True
    document["completed_utc"] = datetime.now(timezone.utc).isoformat()
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
