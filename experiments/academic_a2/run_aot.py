#!/usr/bin/env python3
"""Build, qualify, and time the frozen A2 prepared-AOT slice."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise SystemExit(f"refusing existing output: {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.package_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(HERE)))
    document = {
        "schema": "gradflow-academic-a2-aot-campaign-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "orders": {},
    }
    for order in (5, 11, 15):
        package = arguments.package_dir / f"scalar_o{order}_f32_d3_n64.pt2"
        build_record = arguments.output.parent / f"aot_build_o{order}.json"
        build = subprocess.run(
            (
                sys.executable,
                str(HERE / "build_aot.py"),
                "--order",
                str(order),
                "--output",
                str(package),
                "--record",
                str(build_record),
            ),
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        qualification = None
        worker_stdout = arguments.output.parent / f"aot_worker_o{order}.stdout"
        worker_stderr = arguments.output.parent / f"aot_worker_o{order}.stderr"
        run = None
        if build.returncode == 0:
            run = subprocess.run(
                (
                    sys.executable,
                    str(HERE / "aot_worker.py"),
                    "--order",
                    str(order),
                    "--package",
                    str(package),
                ),
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            worker_stdout.write_text(run.stdout)
            worker_stderr.write_text(run.stderr)
            try:
                qualification = json.loads(run.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                qualification = None
        document["orders"][str(order)] = {
            "build_returncode": build.returncode,
            "build_record": json.loads(build_record.read_text()),
            "worker_returncode": None if run is None else run.returncode,
            "qualification": qualification,
            "package_retained_outside_repository": str(package),
        }
        arguments.output.write_text(json.dumps(document, indent=2) + "\n")
        print(
            f"AOT order {order}: build={build.returncode} "
            f"worker={None if run is None else run.returncode}",
            flush=True,
        )
    document["complete"] = True
    document["completed_utc"] = datetime.now(timezone.utc).isoformat()
    arguments.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
