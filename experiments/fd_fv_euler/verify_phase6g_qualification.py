#!/usr/bin/env python3
"""Independently verify immutable Phase-6G internal-loader records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6g_qualification_20260829"
PHASE6F = ROOT / "experiments/fd_fv_euler/results/phase_6f_qualification_20260829"
TOOLS = {
    "c++", "cc", "clang", "clang++", "gcc", "g++", "nvcc", "ld", "ld.lld",
    "as", "cmake", "make", "ninja",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def classification(trace: Path) -> dict:
    executions = exec_lines(trace)
    tools = [item for item in executions if item["basename"] in TOOLS]
    pythons = [item for item in executions if item["basename"].startswith("python")]
    return {
        "execve_count": len(executions),
        "tool_attempts": tools,
        "tool_attempt_count": len(tools),
        "python_execve": pythons,
        "python_child_count": max(0, len(pythons) - 1),
        "compiler_free": not tools,
        "helper_process_free": len(pythons) == 1,
    }


def verify_checksums() -> None:
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = RESULTS / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative


def main() -> None:
    verify_checksums()
    payload = json.loads((RESULTS / "qualification.json").read_text())
    assert payload["protocol_commit"] == "0efdccb"
    assert payload["phase6f_prerequisite"]["passed"]
    assert sha256(PHASE6F / "qualification.json") == payload[
        "phase6f_qualification_sha256"
    ]
    assert sha256(PHASE6F / "prepared_cache_manifest.json") == payload[
        "prepared_cache_manifest_sha256"
    ]
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected, relative
    environment = payload["environment"]
    assert sha256(Path(environment["torch_c_extension"])) == environment[
        "torch_c_extension_sha256"
    ]
    assert sha256(Path(environment["aoti_stub"])) == environment["aoti_stub_sha256"]

    records = [
        json.loads(path.read_text())
        for path in sorted((RESULTS / "records").glob("*.json"))
    ]
    assert len(records) == 8
    eligibility = []
    for record in records:
        stem = f"{record['endpoint']}_{record['problem']}_{record['method']}"
        trace = RESULTS / "traces" / f"{stem}.strace"
        recomputed = classification(trace)
        assert recomputed == record["process_trace"]
        assert record["cache_before"] == record["cache_after"]
        assert record["cache_unchanged"]
        assert record["authority_parity"]["passed"]
        assert record["oracle"]["passed"]
        assert record["diagnostics"]["completed"]
        assert record["terminal_state_cuda_before_materialization"]
        array = RESULTS / "arrays" / f"{stem}.npy"
        assert sha256(array) == record["array_file_sha256"]
        eligible = bool(
            record["worker_returncode"] == 0
            and recomputed["compiler_free"]
            and recomputed["helper_process_free"]
            and record["cache_unchanged"]
            and record["authority_parity"]["passed"]
            and record["oracle"]["passed"]
            and record["diagnostics"]["completed"]
        )
        assert record["eligible"] == eligible
        eligibility.append(eligible)

    control = payload["public_loader_control"]
    assert len(control["records"]) == 8
    for item in control["records"]:
        trace = ROOT / item["trace"]
        assert sha256(trace) == item["trace_sha256"]
        public = classification(trace)
        successful = [
            entry for entry in public["tool_attempts"] if entry["path"] == "/usr/bin/g++"
        ]
        compile_commands = [
            entry
            for entry in successful
            if not any(flag in entry["line"] for flag in ('"--version"', '"-v"'))
        ]
        assert public["tool_attempt_count"] == item["tool_attempt_count"]
        assert len(successful) == item["successful_usr_bin_gxx_count"] == 3
        assert len(compile_commands) == item["runtime_compile_command_count"] == 0
        assert item["cache_unchanged"]
    assert control["all_three_successful_metadata_queries"]
    assert control["all_zero_runtime_compile_commands"]
    assert control["all_caches_unchanged"]

    passed = len(eligibility) == 8 and all(eligibility)
    assert payload["lane_status"]["compiler_free_internal_aoti"] == passed
    assert payload["lane_status"]["performance_admitted"] == passed
    assert not payload["performance_measurements_collected"]
    assert not payload["production_sources_modified"]
    assert not payload["dveb_modified"]
    assert not payload["publication_claim"]
    print(
        "Phase 6G qualification records verified "
        f"(8/8 numerical; performance admitted={passed})."
    )


if __name__ == "__main__":
    main()
