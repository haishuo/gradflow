#!/usr/bin/env python3
"""Run the one authorized non-aliasing Phase-6E device-loop requalification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LANE_A = ROOT / "experiments/fd_fv_euler/results/phase_6e_20260829"
LANE_A_VERIFY = ROOT / "experiments/fd_fv_euler/verify_phase6e_repro.py"
INITIAL_AOT_VERIFY = ROOT / "experiments/fd_fv_euler/verify_phase6e_aot.py"
BUILDER = Path(__file__).with_name("build_phase6e_aot.py")
WORKER = Path(__file__).with_name("phase6e_aot_worker.py")
MODEL = Path(__file__).with_name("phase6e_aot_model.py")
PROTOCOL = ROOT / "docs/FD_FV_PHASE_6E_PROTOCOL.md"
AMENDMENT = ROOT / "docs/FD_FV_PHASE_6E_PROTOCOL_AMENDMENT.md"
PROTOCOL_COMMIT = "af90466"
AMENDMENT_COMMIT = "94e0fe4"
PROBLEMS = ("sod", "shu_osher")
METHODS = ("fd", "fv")


sys.path.insert(0, str(ROOT))
from experiments.fd_fv_euler.qualify_phase6e_aot import run


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()


def verify(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        (sys.executable, str(path)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    artifacts = arguments.artifact_dir.resolve()
    if output.exists() or artifacts.exists():
        raise FileExistsError("refusing existing Phase 6E device-r1 output")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6E device-r1 qualification requires a clean tree")
    prerequisites = [verify(LANE_A_VERIFY), verify(INITIAL_AOT_VERIFY)]
    if not all(item["passed"] for item in prerequisites):
        raise RuntimeError("Phase 6E device-r1 prerequisite verification failed")
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

    for problem in PROBLEMS:
        for method in METHODS:
            stem = f"device_{problem}_{method}"
            package = artifacts / f"{stem}.pt2"
            build_path = build_records / f"{stem}.json"
            completed, cache_files = run(
                (
                    sys.executable,
                    str(BUILDER),
                    "--lane",
                    "device",
                    "--problem",
                    problem,
                    "--method",
                    method,
                    "--output",
                    str(package),
                    "--record",
                    str(build_path),
                ),
                cache_prefix=f"gradflow-phase6e-device-r1-build-{stem}-",
            )
            build = json.loads(build_path.read_text()) if build_path.exists() else {
                "status": "failed",
                "error_type": "BuilderProcessFailure",
                "error": "builder produced no record",
            }
            build.update(
                {
                    "worker_returncode": completed.returncode,
                    "worker_stdout": completed.stdout,
                    "worker_stderr": completed.stderr,
                    "build_cache_files": cache_files,
                    "series": "phase_6e_device_r1_20260829",
                }
            )
            build_path.write_text(json.dumps(build, indent=2, sort_keys=True) + "\n")
            builds.append(build)
            print(f"build r1 {stem}: {build.get('status')}", flush=True)
            if build.get("status") != "completed":
                qualifications.append(
                    {
                        "status": "not_run_build_failed",
                        "lane": "device",
                        "problem": problem,
                        "method": method,
                        "series": "phase_6e_device_r1_20260829",
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
                    "device",
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
                cache_prefix=f"gradflow-phase6e-device-r1-qualify-{stem}-",
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
                    "series": "phase_6e_device_r1_20260829",
                }
            )
            qualify_path.write_text(
                json.dumps(qualification, indent=2, sort_keys=True) + "\n"
            )
            qualifications.append(qualification)
            print(
                f"qualify r1 {stem}: {qualification.get('status')} "
                f"eligible={qualification.get('eligible')}",
                flush=True,
            )

    passed = (
        len(builds) == len(qualifications) == 4
        and all(item.get("status") == "completed" for item in builds)
        and all(item.get("eligible") for item in qualifications)
    )
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_euler_phase_6e_device_r1_qualification",
        "measurement_date": "2026-08-29",
        "series": "phase_6e_device_r1_20260829",
        "protocol_commit": PROTOCOL_COMMIT,
        "amendment_commit": AMENDMENT_COMMIT,
        "source_commit": source_commit,
        "source_dirty": False,
        "prerequisites": prerequisites,
        "artifact_directory": str(artifacts),
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PROTOCOL, AMENDMENT, BUILDER, WORKER, MODEL, Path(__file__))
        },
        "build_records": builds,
        "qualification_records": qualifications,
        "lane_status": {
            "builds_completed": sum(item.get("status") == "completed" for item in builds),
            "qualifications_eligible": sum(bool(item.get("eligible")) for item in qualifications),
            "passed": passed,
            "performance_admitted": passed,
        },
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
    print(f"Phase 6E device-r1 passed={passed}", flush=True)


if __name__ == "__main__":
    main()
