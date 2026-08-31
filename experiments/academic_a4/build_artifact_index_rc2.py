#!/usr/bin/env python3
"""Build the deterministic Academic A4 rc2 release-payload index."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
TAG = "academic-v0.1.0-rc2"
EXCLUDED = {
    "docs/ACADEMIC_A4_RC2_RESULTS.md",
    "experiments/academic_a4/evidence/a4_rc2_20260831/SHA256SUMS",
    "experiments/academic_a4/evidence/a4_rc2_20260831/artifact_index.json",
    "experiments/academic_a4/evidence/a4_rc2_20260831/cleanroom.json",
    "experiments/academic_a4/evidence/a4_rc2_20260831/cleanroom_stderr.log",
    "experiments/academic_a4/evidence/a4_rc2_20260831/cleanroom_stdout.log",
    "experiments/academic_a4/evidence/a4_rc2_20260831/CLEANROOM_SHA256SUMS",
}
REFERENCE_BLOCKERS = (
    "references/gottlieb_matlab/",
    "references/jiang_shu_fortran/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if git("status", "--porcelain"):
        raise RuntimeError("refusing to index a dirty source tree")
    tracked = sorted(filter(None, git("ls-files").splitlines()))
    indexed = [path for path in tracked if path not in EXCLUDED]
    entries = []
    totals: dict[str, dict[str, int]] = {}
    for relative in indexed:
        path = ROOT / relative
        group = relative.split("/", 1)[0] if "/" in relative else "project"
        entry = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "category": group,
        }
        entries.append(entry)
        summary = totals.setdefault(group, {"files": 0, "bytes": 0})
        summary["files"] += 1
        summary["bytes"] += entry["bytes"]
    payload = {
        "schema": "gradflow-academic-a4-artifact-index-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree_clean": True,
        "release_candidate_tag": TAG,
        "previous_release_candidate_tag": "academic-v0.1.0-rc1",
        "hash_algorithm": "sha256",
        "self_excluded_paths": sorted(EXCLUDED),
        "redistribution": {
            "status": "unresolved_public_release_blockers",
            "policy_document": "docs/ACADEMIC_A4_RIGHTS_STATUS.md",
            "reference_prefixes_requiring_permission_or_exclusion": list(REFERENCE_BLOCKERS),
            "top_level_project_license_present": any(
                (ROOT / name).exists() for name in ("LICENSE", "LICENSE.txt", "LICENSE.md")
            ),
        },
        "primary_toolchain_evidence": {
            "document": "docs/ACADEMIC_U5_RESULTS.md",
            "torch": "2.13.0+cu130",
            "evidence": "experiments/academic_u5/evidence/u5_20260831",
        },
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "category_totals": totals,
        "files": entries,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"indexed {len(entries)} files ({payload['total_bytes']} bytes)")


if __name__ == "__main__":
    main()

