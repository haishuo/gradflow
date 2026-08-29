#!/usr/bin/env python3
"""Qualify the frozen Phase-6G compiler-independent internal AOTI loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "src", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch


PROTOCOL = ROOT / "docs/FD_FV_PHASE_6G_PROTOCOL.md"
PROTOCOL_COMMIT = "0efdccb"
WORKER = Path(__file__).with_name("phase6g_worker.py")
PHASE6F_VERIFY = Path(__file__).with_name("verify_phase6f_qualification.py")
PHASE6F = ROOT / "experiments/fd_fv_euler/results/phase_6f_qualification_20260829"
PHASE6E = ROOT / "experiments/fd_fv_euler/results"
LANE_A = PHASE6E / "phase_6e_20260829"
HOST_RESULTS = PHASE6E / "phase_6e_aot_20260829"
TENSOR_RESULTS = PHASE6E / "phase_6e_device_r1_20260829"
PROBLEMS = ("sod", "shu_osher")
METHODS = ("fd", "fv")
TOOL_NAMES = {
    "c++", "cc", "clang", "clang++", "gcc", "g++", "nvcc", "ld", "ld.lld",
    "as", "cmake", "make", "ninja",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()


def environment(cache: Path) -> dict[str, str]:
    result = os.environ.copy()
    result["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
    result["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    return result


def verify(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        (sys.executable, str(path)),
        cwd=ROOT,
        env=environment(Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp"))),
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


def exec_lines(trace: Path) -> list[dict[str, str]]:
    expression = re.compile(r'execve\("([^"]+)"')
    result = []
    for line in trace.read_text(errors="replace").splitlines():
        match = expression.search(line)
        if match:
            result.append(
                {"path": match.group(1), "basename": Path(match.group(1)).name, "line": line}
            )
    return result


def process_classification(trace: Path) -> dict[str, Any]:
    executions = exec_lines(trace)
    tool_attempts = [item for item in executions if item["basename"] in TOOL_NAMES]
    python_processes = [
        item for item in executions if item["basename"].startswith("python")
    ]
    return {
        "execve_count": len(executions),
        "tool_attempts": tool_attempts,
        "tool_attempt_count": len(tool_attempts),
        "python_execve": python_processes,
        "python_child_count": max(0, len(python_processes) - 1),
        "compiler_free": not tool_attempts,
        "helper_process_free": len(python_processes) == 1,
    }


def package_record(endpoint: str, problem: str, method: str) -> dict[str, Any]:
    base = HOST_RESULTS if endpoint == "aot_host_internal" else TENSOR_RESULTS
    prefix = "host" if endpoint == "aot_host_internal" else "device"
    return json.loads(
        (base / "build_records" / f"{prefix}_{problem}_{method}.json").read_text()
    )


def authority_paths(problem: str, method: str) -> tuple[Path, Path]:
    stem = f"{problem}_{method}_cpu_eager_r0"
    return LANE_A / "arrays" / f"{stem}.npy", LANE_A / "raw" / f"{stem}.json"


def public_control() -> dict[str, Any]:
    records = []
    for endpoint in ("aot_host", "aot_tensor"):
        for problem in PROBLEMS:
            for method in METHODS:
                stem = f"{endpoint}_{problem}_{method}"
                trace = PHASE6F / "traces" / f"{stem}.strace"
                classification = process_classification(trace)
                successful_gxx = [
                    item
                    for item in classification["tool_attempts"]
                    if item["path"] == "/usr/bin/g++"
                ]
                compile_commands = [
                    item
                    for item in successful_gxx
                    if not any(flag in item["line"] for flag in ('"--version"', '"-v"'))
                ]
                phase6f_record = json.loads((PHASE6F / "records" / f"{stem}.json").read_text())
                records.append(
                    {
                        "endpoint": endpoint,
                        "problem": problem,
                        "method": method,
                        "trace": str(trace.relative_to(ROOT)),
                        "trace_sha256": sha256(trace),
                        "tool_attempt_count": classification["tool_attempt_count"],
                        "successful_usr_bin_gxx_count": len(successful_gxx),
                        "runtime_compile_command_count": len(compile_commands),
                        "cache_unchanged": phase6f_record["cache_unchanged"],
                    }
                )
    return {
        "records": records,
        "all_three_successful_metadata_queries": all(
            item["successful_usr_bin_gxx_count"] == 3 for item in records
        ),
        "all_zero_runtime_compile_commands": all(
            item["runtime_compile_command_count"] == 0 for item in records
        ),
        "all_caches_unchanged": all(item["cache_unchanged"] for item in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    if output.exists():
        raise FileExistsError("refusing existing Phase 6G qualification output")
    if git("status", "--porcelain"):
        raise RuntimeError("Phase 6G qualification requires a clean tree")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"), cwd=ROOT
    ).returncode:
        raise RuntimeError("frozen Phase 6G protocol is not an ancestor")
    prerequisite = verify(PHASE6F_VERIFY)
    if not prerequisite["passed"]:
        raise RuntimeError("Phase 6F verification failed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible to Phase 6G orchestrator")

    phase6f = json.loads((PHASE6F / "qualification.json").read_text())
    prepared_cache = Path(phase6f["prepared_cache"]["directory"])
    recorded_cache = json.loads((PHASE6F / "prepared_cache_manifest.json").read_text())
    if file_manifest(prepared_cache) != recorded_cache:
        raise RuntimeError("Phase 6F prepared cache identity changed")

    output.mkdir(parents=True)
    (output / "records").mkdir()
    (output / "arrays").mkdir()
    (output / "traces").mkdir()
    qualifications = []
    for endpoint in ("aot_host_internal", "aot_tensor_internal"):
        for problem in PROBLEMS:
            for method in METHODS:
                package_data = package_record(endpoint, problem, method)
                package = Path(package_data["package_path"])
                if sha256(package) != package_data["package_sha256"]:
                    raise RuntimeError(f"package identity changed: {package}")
                authority_array, authority_record = authority_paths(problem, method)
                stem = f"{endpoint}_{problem}_{method}"
                record_path = output / "records" / f"{stem}.json"
                array_path = output / "arrays" / f"{stem}.npy"
                trace_path = output / "traces" / f"{stem}.strace"
                with tempfile.TemporaryDirectory(prefix=f"gradflow-phase6g-{stem}-") as temp:
                    cache = Path(temp) / "cache"
                    shutil.copytree(prepared_cache, cache)
                    before = file_manifest(cache)
                    command = (
                        "/usr/bin/strace", "-f", "-e", "trace=process", "-o", str(trace_path),
                        sys.executable, str(WORKER), "--endpoint", endpoint,
                        "--problem", problem, "--method", method,
                        "--package", str(package),
                        "--authority-array", str(authority_array),
                        "--authority-record", str(authority_record),
                        "--array-output", str(array_path), "--output", str(record_path),
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
                    wall = (time.perf_counter_ns() - started) * 1.0e-9
                    after = file_manifest(cache)
                record = json.loads(record_path.read_text())
                processes = process_classification(trace_path)
                cache_unchanged = before == after == recorded_cache
                record.update(
                    {
                        "process_launch_to_exit_seconds_qualification": wall,
                        "worker_returncode": completed.returncode,
                        "worker_stdout": completed.stdout,
                        "worker_stderr": completed.stderr,
                        "process_trace": processes,
                        "cache_before": before,
                        "cache_after": after,
                        "cache_unchanged": cache_unchanged,
                    }
                )
                record["eligible"] = bool(
                    record.get("eligible")
                    and completed.returncode == 0
                    and processes["compiler_free"]
                    and processes["helper_process_free"]
                    and cache_unchanged
                )
                record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                qualifications.append(record)
                print(f"qualify {stem}: eligible={record['eligible']}", flush=True)

    control = public_control()
    torch_c = Path(torch._C.__file__).resolve()
    internal_source = Path(torch.__file__).resolve().parent / "_C/_aoti.pyi"
    passed = len(qualifications) == 8 and all(item["eligible"] for item in qualifications)
    payload = {
        "schema_version": 1,
        "phase": "fd_fv_euler_phase_6g_qualification",
        "measurement_date": "2026-08-29",
        "protocol_commit": PROTOCOL_COMMIT,
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": False,
        "phase6f_prerequisite": prerequisite,
        "phase6f_qualification_sha256": sha256(PHASE6F / "qualification.json"),
        "prepared_cache_manifest_sha256": sha256(PHASE6F / "prepared_cache_manifest.json"),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "torch_c_extension": str(torch_c),
            "torch_c_extension_sha256": sha256(torch_c),
            "aoti_stub": str(internal_source),
            "aoti_stub_sha256": sha256(internal_source),
        },
        "public_loader_control": control,
        "qualification_records": qualifications,
        "lane_status": {
            "compiler_free_internal_aoti": passed,
            "performance_admitted": passed,
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PROTOCOL, WORKER, Path(__file__), Path(__file__).with_name("phase6f_worker.py"))
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
        *sorted((output / "records").glob("*.json")),
        *sorted((output / "arrays").glob("*.npy")),
        *sorted((output / "traces").glob("*.strace")),
    ]
    (output / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in files)
    )
    print(f"Phase 6G qualification passed={passed}", flush=True)


if __name__ == "__main__":
    main()
