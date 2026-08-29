#!/usr/bin/env python3
"""Construct and qualify the frozen Phase-6F prepared runtime image."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch


PROTOCOL = ROOT / "docs/FD_FV_PHASE_6F_PROTOCOL.md"
PROTOCOL_COMMIT = "c3a19eb"
WORKER = Path(__file__).with_name("phase6f_worker.py")
PHASE6E = ROOT / "experiments/fd_fv_euler/results"
LANE_A = PHASE6E / "phase_6e_20260829"
HOST_RESULTS = PHASE6E / "phase_6e_aot_20260829"
TENSOR_RESULTS = PHASE6E / "phase_6e_device_r1_20260829"
PHASE6E_VERIFIERS = (
    Path(__file__).with_name("verify_phase6e_repro.py"),
    Path(__file__).with_name("verify_phase6e_aot.py"),
    Path(__file__).with_name("verify_phase6e_device_r1.py"),
)
PROBLEMS = ("sod", "shu_osher")
METHODS = ("fd", "fv")
COMPILER_NAMES = {
    "c++",
    "cc",
    "clang",
    "clang++",
    "gcc",
    "g++",
    "nvcc",
    "ld",
    "ld.lld",
    "as",
    "cmake",
    "make",
    "ninja",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()


def verify(path: Path) -> dict[str, Any]:
    verifier_environment = os.environ.copy()
    verifier_environment["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    completed = subprocess.run(
        (sys.executable, str(path)),
        cwd=ROOT,
        env=verifier_environment,
        capture_output=True,
        text=True,
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "passed": completed.returncode == 0,
    }


def file_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "mode": oct(path.stat().st_mode & 0o777),
            "sha256": sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def environment(cache: Path) -> dict[str, str]:
    result = os.environ.copy()
    result["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    result["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    return result


def compiler_processes(trace: Path) -> list[str]:
    result = []
    expression = re.compile(r'execve\("([^"]+)"')
    for line in trace.read_text(errors="replace").splitlines():
        match = expression.search(line)
        if match and Path(match.group(1)).name in COMPILER_NAMES:
            result.append(line)
    return result


def traced_worker(
    arguments: tuple[str, ...], cache: Path, output: Path, trace: Path
) -> tuple[subprocess.CompletedProcess[str], float]:
    command = (
        "/usr/bin/strace",
        "-f",
        "-e",
        "trace=process",
        "-o",
        str(trace),
        sys.executable,
        str(WORKER),
        *arguments,
        "--output",
        str(output),
    )
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment(cache),
        capture_output=True,
        text=True,
        check=False,
    )
    return completed, (time.perf_counter_ns() - started) * 1.0e-9


def deterministic_tar(source: Path, target: Path) -> None:
    with target.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(source.rglob("*")):
                    relative = path.relative_to(source)
                    info = archive.gettarinfo(str(path), arcname=str(relative))
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    if path.is_file():
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
                    else:
                        archive.addfile(info)


def package_record(lane: str, problem: str, method: str) -> dict[str, Any]:
    base = HOST_RESULTS if lane == "host" else TENSOR_RESULTS
    stem = f"{'host' if lane == 'host' else 'device'}_{problem}_{method}"
    return json.loads((base / "build_records" / f"{stem}.json").read_text())


def authority_paths(problem: str, method: str) -> tuple[Path, Path]:
    stem = f"{problem}_{method}_cpu_eager_r0"
    return LANE_A / "arrays" / f"{stem}.npy", LANE_A / "raw" / f"{stem}.json"


def source_evidence(package: Path) -> dict[str, Any]:
    installed = Path(torch.__file__).resolve().parent / "_inductor/codegen/wrapper.py"
    source_lines = installed.read_text().splitlines()
    source_matches = [
        {"line": index, "text": line.strip()}
        for index, line in enumerate(source_lines, 1)
        if "while_loop is codegened as a host side while_loop" in line
    ]
    with zipfile.ZipFile(package) as archive:
        wrappers = sorted(name for name in archive.namelist() if name.endswith("wrapper.cpp"))
        if len(wrappers) != 1:
            raise RuntimeError("expected exactly one generated wrapper")
        wrapper_name = wrappers[0]
        wrapper_bytes = archive.read(wrapper_name)
    wrapper_lines = wrapper_bytes.decode(errors="replace").splitlines()
    wrapper_matches = [
        {"line": index, "text": line.strip()}
        for index, line in enumerate(wrapper_lines, 1)
        if "while (1)" in line or "aoti_torch_item_bool" in line
    ]
    return {
        "installed_inductor_source": str(installed),
        "installed_inductor_source_sha256": sha256(installed),
        "installed_source_matches": source_matches,
        "package": str(package),
        "package_sha256": sha256(package),
        "wrapper_member": wrapper_name,
        "wrapper_sha256": hashlib.sha256(wrapper_bytes).hexdigest(),
        "wrapper_matches": wrapper_matches,
        "host_side_documentation_found": bool(source_matches),
        "host_while_found": any("while (1)" in item["text"] for item in wrapper_matches),
        "item_bool_found": any("aoti_torch_item_bool" in item["text"] for item in wrapper_matches),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    artifacts = arguments.artifact_dir.resolve()
    if output.exists() or artifacts.exists():
        raise FileExistsError("refusing existing Phase 6F qualification output")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6F qualification requires a clean tree")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"), cwd=ROOT
    ).returncode:
        raise RuntimeError("frozen Phase 6F protocol is not an ancestor")
    prerequisites = [verify(path) for path in PHASE6E_VERIFIERS]
    if not all(item["passed"] for item in prerequisites):
        raise RuntimeError("Phase 6E prerequisite verification failed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to Phase 6F orchestrator")

    output.mkdir(parents=True)
    (output / "records").mkdir()
    (output / "arrays").mkdir()
    (output / "traces").mkdir()
    (output / "profiles").mkdir()
    artifacts.mkdir(parents=True)
    prepared_cache = artifacts / "prepared_runtime_cache"
    prepared_cache.mkdir()

    prep_package_record = package_record("host", "sod", "fd")
    prep_package = Path(prep_package_record["package_path"])
    prep_record_path = output / "cache_preparation.json"
    prep_trace = output / "traces/cache_preparation.strace"
    prep_completed, prep_wall = traced_worker(
        (
            "--action", "prepare", "--endpoint", "aot_host",
            "--problem", "sod", "--method", "fd", "--package", str(prep_package),
        ),
        prepared_cache,
        prep_record_path,
        prep_trace,
    )
    preparation = json.loads(prep_record_path.read_text())
    preparation.update(
        {
            "process_launch_to_exit_seconds": prep_wall,
            "worker_returncode": prep_completed.returncode,
            "worker_stdout": prep_completed.stdout,
            "worker_stderr": prep_completed.stderr,
            "compiler_processes": compiler_processes(prep_trace),
        }
    )
    prep_record_path.write_text(json.dumps(preparation, indent=2, sort_keys=True) + "\n")
    if preparation.get("status") != "completed":
        raise RuntimeError("prepared cache construction failed")

    prepared_manifest = file_manifest(prepared_cache)
    cache_manifest_path = output / "prepared_cache_manifest.json"
    cache_manifest_path.write_text(
        json.dumps(prepared_manifest, indent=2, sort_keys=True) + "\n"
    )
    cache_archive = artifacts / "prepared_runtime_cache.tar.gz"
    deterministic_tar(prepared_cache, cache_archive)

    qualifications = []
    for endpoint, lane in (("aot_host", "host"), ("aot_tensor", "tensor")):
        for problem in PROBLEMS:
            for method in METHODS:
                package_data = package_record(lane, problem, method)
                package = Path(package_data["package_path"])
                authority_array, authority_record = authority_paths(problem, method)
                stem = f"{endpoint}_{problem}_{method}"
                record_path = output / "records" / f"{stem}.json"
                array_path = output / "arrays" / f"{stem}.npy"
                trace_path = output / "traces" / f"{stem}.strace"
                with tempfile.TemporaryDirectory(prefix=f"gradflow-phase6f-{stem}-") as temp:
                    cache = Path(temp) / "cache"
                    shutil.copytree(prepared_cache, cache)
                    cache_before = file_manifest(cache)
                    completed, wall = traced_worker(
                        (
                            "--action", "solve", "--endpoint", endpoint,
                            "--problem", problem, "--method", method,
                            "--package", str(package),
                            "--authority-array", str(authority_array),
                            "--authority-record", str(authority_record),
                            "--array-output", str(array_path),
                        ),
                        cache,
                        record_path,
                        trace_path,
                    )
                    cache_after = file_manifest(cache)
                record = json.loads(record_path.read_text())
                compilers = compiler_processes(trace_path)
                cache_unchanged = cache_before == cache_after
                record.update(
                    {
                        "process_launch_to_exit_seconds_qualification": wall,
                        "worker_returncode": completed.returncode,
                        "worker_stdout": completed.stdout,
                        "worker_stderr": completed.stderr,
                        "runtime_compiler_processes": compilers,
                        "runtime_compiler_process_count": len(compilers),
                        "cache_before": cache_before,
                        "cache_after": cache_after,
                        "cache_unchanged": cache_unchanged,
                        "prepared_cache_manifest_sha256": sha256(cache_manifest_path),
                    }
                )
                record["eligible"] = bool(
                    record.get("eligible")
                    and completed.returncode == 0
                    and not compilers
                    and cache_unchanged
                )
                record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                qualifications.append(record)
                print(f"qualify {stem}: eligible={record['eligible']}", flush=True)

    profiles = []
    for problem in PROBLEMS:
        for method in METHODS:
            package_data = package_record("tensor", problem, method)
            package = Path(package_data["package_path"])
            stem = f"aot_tensor_{problem}_{method}"
            record_path = output / "profiles" / f"{stem}.json"
            with tempfile.TemporaryDirectory(prefix=f"gradflow-phase6f-profile-{stem}-") as temp:
                cache = Path(temp) / "cache"
                shutil.copytree(prepared_cache, cache)
                completed = subprocess.run(
                    (
                        sys.executable, str(WORKER), "--action", "profile",
                        "--endpoint", "aot_tensor", "--problem", problem,
                        "--method", method, "--package", str(package),
                        "--output", str(record_path),
                    ),
                    cwd=ROOT,
                    env=environment(cache),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            record = json.loads(record_path.read_text())
            record.update(
                {
                    "worker_returncode": completed.returncode,
                    "worker_stdout": completed.stdout,
                    "worker_stderr": completed.stderr,
                }
            )
            record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            profiles.append(record)

    tensor_example = Path(package_record("tensor", "sod", "fd")["package_path"])
    lowering = source_evidence(tensor_example)
    lowering_path = output / "tensor_loop_lowering.json"
    lowering_path.write_text(json.dumps(lowering, indent=2, sort_keys=True) + "\n")
    all_qualified = len(qualifications) == 8 and all(
        item["eligible"] for item in qualifications
    )
    characterization_passed = bool(
        lowering["host_side_documentation_found"]
        and lowering["host_while_found"]
        and lowering["item_bool_found"]
        and len(profiles) == 4
        and all(item.get("host_synchronization_observed") for item in profiles)
    )
    environment_record = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
    }
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_euler_phase_6f_qualification",
        "measurement_date": "2026-08-29",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": False,
        "prerequisites": prerequisites,
        "environment": environment_record,
        "preparation": preparation,
        "prepared_cache": {
            "directory": str(prepared_cache),
            "file_count": len(prepared_manifest),
            "manifest": str(cache_manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(cache_manifest_path),
            "archive": str(cache_archive),
            "archive_size": cache_archive.stat().st_size,
            "archive_sha256": sha256(cache_archive),
            "restore_command": f"mkdir CACHE && tar -xzf {cache_archive} -C CACHE",
        },
        "qualification_records": qualifications,
        "tensor_loop_profiles": profiles,
        "tensor_loop_lowering": lowering,
        "lane_status": {
            "prepared_runtime_cache": preparation["status"] == "completed",
            "prepared_package_runtime": all_qualified,
            "tensor_loop_host_synchronization_characterized": characterization_passed,
            "performance_admitted": all_qualified,
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PROTOCOL, WORKER, Path(__file__))
        },
        "production_sources_modified": False,
        "dveb_modified": False,
        "performance_measurements_collected": False,
        "publication_claim": False,
    }
    aggregate = output / "qualification.json"
    aggregate.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    included = [
        aggregate,
        prep_record_path,
        cache_manifest_path,
        lowering_path,
        *sorted((output / "records").glob("*.json")),
        *sorted((output / "arrays").glob("*.npy")),
        *sorted((output / "traces").glob("*.strace")),
        *sorted((output / "profiles").glob("*.json")),
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in included)
    )
    print(
        f"Phase 6F qualification passed={all_qualified}; "
        f"host synchronization characterized={characterization_passed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
