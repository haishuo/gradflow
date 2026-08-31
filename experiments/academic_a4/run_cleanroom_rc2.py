#!/usr/bin/env python3
"""Reproduce rc2 sentinels from an isolated no-hardlink local clone."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path, environment: dict[str, str]) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=cwd, env=environment, check=False, capture_output=True, text=True
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gradflow-a4-rc2-cleanroom-") as temporary:
        clone = Path(temporary) / "gradflow"
        subprocess.run(
            ("git", "clone", "--quiet", "--no-hardlinks", "--local", str(ROOT), str(clone)),
            check=True,
        )
        subprocess.run(("git", "checkout", "--quiet", arguments.ref), cwd=clone, check=True)
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=clone, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty_before = subprocess.run(
            ("git", "status", "--porcelain"), cwd=clone, check=True, capture_output=True, text=True
        ).stdout
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join((str(clone), str(clone / "src")))
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        commands = [
            [sys.executable, "-m", "pytest", "-q"],
            [sys.executable, "experiments/academic_a1/verify_a1.py", "experiments/academic_a1/evidence/a1_20260830"],
            [sys.executable, "experiments/academic_a2/verify_a2.py", "experiments/academic_a2/evidence/a2_20260830"],
            [sys.executable, "experiments/academic_a3/verify_a3.py", "experiments/academic_a3/evidence/a3_20260830"],
            [sys.executable, "experiments/academic_a4/verify_a4.py", "experiments/academic_a4/evidence/a4_20260830", "--ref", "academic-v0.1.0-rc1"],
            [sys.executable, "experiments/academic_u4a/verify_u4a.py"],
            [sys.executable, "experiments/academic_u4b/verify_u4b.py"],
            [sys.executable, "experiments/academic_u4c/verify_cuda_qualification.py"],
            [sys.executable, "experiments/academic_u4c/verify_endpoints.py"],
            [sys.executable, "experiments/academic_u4c/verify_performance.py"],
            [sys.executable, "experiments/academic_u4d/verify_campaign.py"],
            [sys.executable, "experiments/academic_u4e/verify_campaign.py"],
            [sys.executable, "experiments/academic_u4f/verify_campaign.py"],
            [sys.executable, "experiments/academic_u5/verify_u5.py", "experiments/academic_u5/evidence/u5_20260831"],
            [sys.executable, "experiments/academic_a4/verify_a4_rc2.py", "experiments/academic_a4/evidence/a4_rc2_20260831", "--ref", arguments.ref],
        ]
        records = [run(command, clone, environment) for command in commands]
        dirty_after = subprocess.run(
            ("git", "status", "--porcelain"), cwd=clone, check=True, capture_output=True, text=True
        ).stdout
    stdout_path = output / "cleanroom_stdout.log"
    stderr_path = output / "cleanroom_stderr.log"
    stdout_path.write_text("\n\n".join(item["stdout"] for item in records))
    stderr_path.write_text("\n\n".join(item["stderr"] for item in records))
    payload = {
        "schema": "gradflow-academic-a4-cleanroom-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_ref": arguments.ref,
        "tested_commit": commit,
        "clone_mode": "local_no_hardlinks",
        "network_used": False,
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "source_tree_clean_before": dirty_before == "",
        "source_tree_clean_after": dirty_after == "",
        "commands": [
            {key: item[key] for key in ("command", "returncode", "wall_seconds")}
            for item in records
        ],
        "all_passed": all(item["returncode"] == 0 for item in records),
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    }
    (output / "cleanroom.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["all_passed"] or dirty_before or dirty_after:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

