#!/usr/bin/env python3
"""Freeze a compact, self-verifying result set after the counted campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    copied_manifest = args.results_dir / "PREPARATION_MANIFEST.json"
    shutil.copyfile(args.prepared_manifest, copied_manifest)

    files = sorted(
        path for path in [*args.results_dir.glob("*.json"), args.report]
        if path.name != "RESULT_MANIFEST.json"
    )
    manifest = {
        "schema": "gradflow-dveb-abi-bakeoff-result-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git_output("branch", "--show-current"),
        "head_before_result_commit": git_output("rev-parse", "HEAD"),
        "prepared_manifest_source": str(args.prepared_manifest.resolve()),
        "prepared_manifest_sha256": sha256(args.prepared_manifest),
        "files": {
            str(path.resolve().relative_to(Path.cwd().resolve())): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
        "verification": "From the repository root, run: sha256sum -c experiments/shu_torch_ablation/results/dveb_abi_bakeoff_20260826/SHA256SUMS",
    }
    manifest_path = args.results_dir / "RESULT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    checksum_path = args.results_dir / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{record['sha256']}  {name}\n"
            for name, record in sorted(manifest["files"].items())
        )
    )
    print(json.dumps({"files": len(files), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
