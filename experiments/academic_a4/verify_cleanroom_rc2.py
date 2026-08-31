#!/usr/bin/env python3
"""Verify the post-tag Academic A4 rc2 clean-room record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
TAG = "academic-v0.1.0-rc2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    arguments = parser.parse_args()
    evidence = arguments.evidence.resolve()
    payload = json.loads((evidence / "cleanroom.json").read_text())
    assert payload["schema"] == "gradflow-academic-a4-cleanroom-v2"
    assert payload["requested_ref"] == TAG
    tagged = subprocess.run(
        ("git", "rev-list", "-n", "1", TAG),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert payload["tested_commit"] == tagged
    assert payload["clone_mode"] == "local_no_hardlinks"
    assert payload["network_used"] is False
    assert payload["source_tree_clean_before"] and payload["source_tree_clean_after"]
    assert payload["all_passed"] and len(payload["commands"]) == 15
    assert all(item["returncode"] == 0 for item in payload["commands"])
    assert payload["stdout_sha256"] == sha256(evidence / "cleanroom_stdout.log")
    assert payload["stderr_sha256"] == sha256(evidence / "cleanroom_stderr.log")
    for line in (evidence / "CLEANROOM_SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert sha256(evidence / relative) == expected, relative
    print("Academic A4 rc2 clean-room audit record verifies.")


if __name__ == "__main__":
    main()

