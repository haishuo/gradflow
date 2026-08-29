#!/usr/bin/env python3
"""Build and qualify the frozen Phase-6E AOT candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LANE_A = ROOT / "experiments/fd_fv_euler/results/phase_6e_20260829"
LANE_A_VERIFY = ROOT / "experiments/fd_fv_euler/verify_phase6e_repro.py"
BUILDER = Path(__file__).with_name("build_phase6e_aot.py")
WORKER = Path(__file__).with_name("phase6e_aot_worker.py")
PROTOCOL = ROOT / "docs/FD_FV_PHASE_6E_PROTOCOL.md"
PROTOCOL_COMMIT = "af90466"
PROBLEMS = ("sod", "shu_osher")
METHODS = ("fd", "fv")
LANES = ("host", "device")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()


def environment(cache: str) -> dict[str, str]:
    result = os.environ.copy()
    result["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    result["TORCHINDUCTOR_CACHE_DIR"] = cache
    return result


def run(
    command: tuple[str, ...], *, cache_prefix: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    with tempfile.TemporaryDirectory(prefix=cache_prefix) as cache:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(cache),
            capture_output=True,
            text=True,
            check=False,
        )
        cache_files = sorted(
            str(path.relative_to(cache))
            for path in Path(cache).rglob("*")
            if path.is_file()
        )
    return completed, cache_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    artifacts = arguments.artifact_dir.resolve()
    if output.exists() or artifacts.exists():
        raise FileExistsError("refusing existing Phase 6E AOT output")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6E AOT qualification requires a clean tree")
    lane_a = subprocess.run(
        (sys.executable, str(LANE_A_VERIFY)), cwd=ROOT, capture_output=True, text=True
    )
    if lane_a.returncode:
        raise RuntimeError("Phase 6E Lane A verification failed")
    source_commit = git("rev-parse", "HEAD")
    build_records = output / "build_records"
    qualification_records = output / "qualification_records"
    arrays = output / "arrays"
    build_records.mkdir(parents=True)
    qualification_records.mkdir()
    arrays.mkdir()
    artifacts.mkdir(parents=True)
    builds: list[dict[str, Any]] = []
    qualifications: list[dict[str, Any]] = []

    for lane in LANES:
        for problem in PROBLEMS:
            for method in METHODS:
                stem = f"{lane}_{problem}_{method}"
                package = artifacts / f"{stem}.pt2"
                record_path = build_records / f"{stem}.json"
                completed, cache_files = run(
                    (
                        sys.executable,
                        str(BUILDER),
                        "--lane",
                        lane,
                        "--problem",
                        problem,
                        "--method",
                        method,
                        "--output",
                        str(package),
                        "--record",
                        str(record_path),
                    ),
                    cache_prefix=f"gradflow-phase6e-build-{stem}-",
                )
                record = json.loads(record_path.read_text()) if record_path.exists() else {
                    "status": "failed",
                    "error_type": "BuilderProcessFailure",
                    "error": "builder produced no record",
                }
                record.update(
                    {
                        "worker_returncode": completed.returncode,
                        "worker_stdout": completed.stdout,
                        "worker_stderr": completed.stderr,
                        "build_cache_files": cache_files,
                    }
                )
                record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                builds.append(record)
                print(f"build {stem}: {record.get('status')}", flush=True)

                if record.get("status") != "completed":
                    qualifications.append(
                        {
                            "status": "not_run_build_failed",
                            "lane": lane,
                            "problem": problem,
                            "method": method,
                            "eligible": False,
                        }
                    )
                    continue
                authority_stem = f"{problem}_{method}_cpu_eager_r0"
                qualify_path = qualification_records / f"{stem}.json"
                array_path = arrays / f"{stem}.npy"
                completed, cache_files = run(
                    (
                        sys.executable,
                        str(WORKER),
                        "--lane",
                        lane,
                        "--problem",
                        problem,
                        "--method",
                        method,
                        "--package",
                        str(package),
                        "--authority-array",
                        str(LANE_A / "arrays" / f"{authority_stem}.npy"),
                        "--authority-record",
                        str(LANE_A / "raw" / f"{authority_stem}.json"),
                        "--output",
                        str(qualify_path),
                        "--array-output",
                        str(array_path),
                    ),
                    cache_prefix=f"gradflow-phase6e-qualify-{stem}-",
                )
                qualification = json.loads(qualify_path.read_text()) if qualify_path.exists() else {
                    "status": "failed",
                    "error_type": "QualificationProcessFailure",
                    "error": "qualification worker produced no record",
                    "eligible": False,
                }
                qualification.update(
                    {
                        "worker_returncode": completed.returncode,
                        "worker_stdout": completed.stdout,
                        "worker_stderr": completed.stderr,
                        "qualification_cache_files": cache_files,
                    }
                )
                qualify_path.write_text(
                    json.dumps(qualification, indent=2, sort_keys=True) + "\n"
                )
                qualifications.append(qualification)
                print(
                    f"qualify {stem}: {qualification.get('status')} "
                    f"eligible={qualification.get('eligible')}",
                    flush=True,
                )

    lane_status = {}
    for lane in LANES:
        selected_builds = [item for item in builds if item.get("lane") == lane]
        selected_qualifications = [
            item for item in qualifications if item.get("lane") == lane
        ]
        passed = (
            len(selected_builds) == len(selected_qualifications) == 4
            and all(item.get("status") == "completed" for item in selected_builds)
            and all(item.get("eligible") for item in selected_qualifications)
        )
        lane_status[lane] = {
            "builds_completed": sum(
                item.get("status") == "completed" for item in selected_builds
            ),
            "qualifications_eligible": sum(
                bool(item.get("eligible")) for item in selected_qualifications
            ),
            "passed": passed,
            "performance_admitted": passed,
        }

    payload = {
        "schema_version": 1,
        "phase": "fd_fv_euler_phase_6e_aot_qualification",
        "measurement_date": "2026-08-29",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": source_commit,
        "source_dirty": False,
        "lane_a_verification_stdout": lane_a.stdout.strip(),
        "lane_a_verification_passed": True,
        "artifact_directory": str(artifacts),
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PROTOCOL, BUILDER, WORKER, Path(__file__), Path(__file__).with_name("phase6e_aot_model.py"))
        },
        "build_records": builds,
        "qualification_records": qualifications,
        "lane_status": lane_status,
        "performance_measurements_collected": False,
        "production_sources_modified": False,
        "dveb_modified": False,
        "publication_claim": False,
    }
    aggregate = output / "qualification.json"
    aggregate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    files = [
        aggregate,
        *sorted(build_records.glob("*.json")),
        *sorted(qualification_records.glob("*.json")),
        *sorted(arrays.glob("*.npy")),
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )
    print(f"Phase 6E AOT lane status: {json.dumps(lane_status, sort_keys=True)}")


if __name__ == "__main__":
    main()
