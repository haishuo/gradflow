#!/usr/bin/env python3
"""Independently verify immutable Phase-6F qualification records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/fd_fv_euler/results/phase_6f_qualification_20260829"
AGGREGATE = RESULTS / "qualification.json"
COMPILERS = {
    "c++", "cc", "clang", "clang++", "gcc", "g++", "nvcc", "ld", "ld.lld",
    "as", "cmake", "make", "ninja",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_compilers(path: Path) -> list[str]:
    expression = re.compile(r'execve\("([^"]+)"')
    result = []
    for line in path.read_text(errors="replace").splitlines():
        match = expression.search(line)
        if match and Path(match.group(1)).name in COMPILERS:
            result.append(line)
    return result


def verify_checksums() -> None:
    for line in (RESULTS / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = RESULTS / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative


def main() -> None:
    verify_checksums()
    payload = json.loads(AGGREGATE.read_text())
    assert payload["protocol_commit"] == "c3a19eb"
    assert all(item["passed"] for item in payload["prerequisites"])
    for relative, expected in payload["source_hashes"].items():
        assert sha256(ROOT / relative) == expected, relative

    cache = payload["prepared_cache"]
    archive = Path(cache["archive"])
    assert archive.is_file()
    assert archive.stat().st_size == cache["archive_size"]
    assert sha256(archive) == cache["archive_sha256"]
    recorded_manifest = json.loads((ROOT / cache["manifest"]).read_text())
    assert sha256(ROOT / cache["manifest"]) == cache["manifest_sha256"]
    assert len(recorded_manifest) == cache["file_count"]
    archived_manifest = []
    with tarfile.open(archive, "r:gz") as source:
        for member in sorted(source.getmembers(), key=lambda item: item.name):
            if not member.isfile():
                continue
            extracted = source.extractfile(member)
            assert extracted is not None
            contents = extracted.read()
            archived_manifest.append(
                {
                    "path": member.name,
                    "size": member.size,
                    "mode": oct(member.mode),
                    "sha256": hashlib.sha256(contents).hexdigest(),
                }
            )
    assert archived_manifest == recorded_manifest

    prep_trace = RESULTS / "traces/cache_preparation.strace"
    preparation = json.loads((RESULTS / "cache_preparation.json").read_text())
    assert preparation["status"] == "completed"
    assert trace_compilers(prep_trace) == preparation["compiler_processes"]
    assert preparation["compiler_processes"], "preparation did not record helper compilation"

    records = [
        json.loads(path.read_text())
        for path in sorted((RESULTS / "records").glob("*.json"))
    ]
    assert len(records) == 8
    recomputed_eligibility = []
    for record in records:
        stem = f"{record['endpoint']}_{record['problem']}_{record['method']}"
        trace = RESULTS / "traces" / f"{stem}.strace"
        assert trace_compilers(trace) == record["runtime_compiler_processes"]
        assert record["runtime_compiler_process_count"] == len(
            record["runtime_compiler_processes"]
        )
        assert record["cache_before"] == recorded_manifest
        assert record["cache_after"] == recorded_manifest
        assert record["cache_unchanged"]
        assert record["authority_parity"]["passed"]
        assert record["oracle"]["passed"]
        assert record["diagnostics"]["completed"]
        array = RESULTS / "arrays" / f"{stem}.npy"
        assert sha256(array) == record["array_file_sha256"]
        eligible = bool(
            record["authority_parity"]["passed"]
            and record["oracle"]["passed"]
            and record["diagnostics"]["completed"]
            and record["cache_unchanged"]
            and record["runtime_compiler_process_count"] == 0
            and record["worker_returncode"] == 0
        )
        assert record["eligible"] == eligible
        recomputed_eligibility.append(eligible)

    profiles = [
        json.loads(path.read_text())
        for path in sorted((RESULTS / "profiles").glob("*.json"))
    ]
    assert len(profiles) == 4
    for record in profiles:
        assert record["eligible"]
        assert record["host_synchronization_observed"]
        assert record["selected_events"]

    lowering = json.loads((RESULTS / "tensor_loop_lowering.json").read_text())
    assert lowering["host_side_documentation_found"]
    assert lowering["host_while_found"]
    assert lowering["item_bool_found"]
    assert sha256(Path(lowering["installed_inductor_source"])) == lowering[
        "installed_inductor_source_sha256"
    ]
    assert payload["lane_status"]["prepared_runtime_cache"]
    all_qualified = len(recomputed_eligibility) == 8 and all(recomputed_eligibility)
    assert payload["lane_status"]["prepared_package_runtime"] == all_qualified
    assert payload["lane_status"]["tensor_loop_host_synchronization_characterized"]
    assert payload["lane_status"]["performance_admitted"] == all_qualified
    assert not payload["performance_measurements_collected"]
    assert not payload["production_sources_modified"]
    assert not payload["dveb_modified"]
    assert not payload["publication_claim"]
    assert subprocess.check_output(
        ("git", "rev-parse", "c3a19eb^{commit}"), cwd=ROOT, text=True
    ).strip()
    print(
        "Phase 6F qualification records verified "
        f"(8/8 numerical; performance admitted={all_qualified})."
    )


if __name__ == "__main__":
    main()
