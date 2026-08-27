#!/usr/bin/env python3
"""Orchestrate the frozen Phase-D mixed-precision CUDA benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("benchmark_worker.py")
TIER1B = ROOT / "experiments/mixed_precision/results/phase_d_tier1b_20260827/search.json"
ORDERS = (5, 11, 15)
POLICIES = (
    "all_f64",
    "indicators_f32",
    "weight_formation_f32",
    "indicators_and_weight_formation_f32",
    "all_internal_f32",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_text(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def numerical_classes() -> dict[tuple[int, int], str]:
    payload = json.loads(TIER1B.read_text())
    return {
        (record["order"], record["mask"]): record["classification"]
        for record in payload["records"]
    }


def verdict(speedup: float) -> str:
    if speedup > 1.05:
        return "performance_positive"
    if speedup < 0.95:
        return "performance_negative"
    return "unresolved_within_5_percent"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output directory: {args.output}")
    source_commit = git_text("rev-parse", "HEAD")
    source_dirty = bool(git_text("status", "--porcelain"))
    args.output.mkdir(parents=True)
    raw_dir = args.output / "raw"
    raw_dir.mkdir()
    classes = numerical_classes()
    records: list[dict[str, Any]] = []

    for order in ORDERS:
        for policy in POLICIES:
            output = raw_dir / f"order{order}_{policy}.json"
            with tempfile.TemporaryDirectory(
                prefix=f"gradflow-inductor-o{order}-{policy}-"
            ) as cache:
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(ROOT / "src")
                environment["TORCHINDUCTOR_CACHE_DIR"] = cache
                completed = subprocess.run(
                    (
                        sys.executable,
                        str(WORKER),
                        "--order",
                        str(order),
                        "--policy",
                        policy,
                        "--output",
                        str(output),
                    ),
                    cwd=ROOT,
                    env=environment,
                    text=True,
                    capture_output=True,
                )
            if not output.exists():
                output.write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "order": order,
                            "policy": policy,
                            "error_type": "WorkerProcessFailure",
                            "error": f"return code {completed.returncode}",
                            "stdout": completed.stdout,
                            "stderr": completed.stderr,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            record = json.loads(output.read_text())
            mask = record.get("mask")
            record["numerical_classification"] = classes.get(
                (order, mask), "not_recorded"
            )
            records.append(record)
            print(order, policy, record["status"], flush=True)

    for order in ORDERS:
        selected = [record for record in records if record["order"] == order]
        baseline = next(record for record in selected if record["policy"] == "all_f64")
        if baseline["status"] != "completed":
            continue
        for record in selected:
            if record["status"] != "completed":
                continue
            for mode in ("eager", "compiled"):
                speedup = (
                    baseline[mode]["median_ms"] / record[mode]["median_ms"]
                )
                record[mode]["speedup_vs_all_f64"] = speedup
                record[mode]["verdict_vs_all_f64"] = verdict(speedup)

    result = {
        "schema_version": 1,
        "phase": "D-scalar-mixed-precision-performance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/MIXED_PRECISION_PHASE_D_PERFORMANCE_PROTOCOL.md",
        "source_commit": source_commit,
        "source_dirty": source_dirty,
        "command": " ".join(sys.argv),
        "tier1b_result_sha256": sha256(TIER1B),
        "orders": ORDERS,
        "policies": POLICIES,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "records": records,
    }
    result_path = args.output / "benchmark.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sums = [f"{sha256(result_path)}  {result_path.name}"]
    sums.extend(
        f"{sha256(path)}  raw/{path.name}" for path in sorted(raw_dir.glob("*.json"))
    )
    (args.output / "SHA256SUMS").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
