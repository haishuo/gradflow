#!/usr/bin/env python3
"""Freeze the final committed DVEB WENO artifact and native ceiling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


DVEB_EXPECTED = "2b087aed48f999ae2ab0e81dd5fb48a40c289d3ef5292f7dceb491f330a970f8"
CEILING_EXPECTED = "873a9227196664398012e7d42a27e29ec9cd3610c45a4c61ab40a0688aed3caa"
PROGRAM_HASH = "c6e5bd916f951ff412eac99863a74f8c98e5e14b044097a7ad59fe26f704c381"
MODULE_HASH = "555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260"
DEFAULT_DVEB = Path("/mnt/projects/dveb/build/shu_euler3d_portable/shu_euler3d_portable")
DEFAULT_CEILING = Path("/mnt/projects/dveb/tools/shu_euler3d_ceiling/build/shu_ceiling")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def freeze(source: Path, destination: Path, expected: str) -> dict[str, str]:
    source = source.resolve()
    if not source.is_file() or sha256(source) != expected:
        observed = sha256(source) if source.is_file() else "missing"
        raise SystemExit(f"refusing {source}: expected {expected}, got {observed}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o755)
    if sha256(destination) != expected:
        raise RuntimeError(f"copy verification failed: {destination}")
    return {"source": str(source), "frozen_copy": str(destination), "sha256": expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dveb", type=Path, default=DEFAULT_DVEB)
    parser.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite {manifest_path}")

    gradflow = Path(__file__).resolve().parents[2]
    dveb = Path("/mnt/projects/dveb")
    if git_value(dveb, "status", "--porcelain"):
        raise SystemExit("DVEB worktree is not clean")
    dveb_commit = git_value(dveb, "rev-parse", "HEAD")
    if dveb_commit != "2f1f3ab2c98349cc3fa1eab2b76d3f20b2eefedc":
        raise SystemExit(f"unexpected DVEB commit: {dveb_commit}")

    manifest = {
        "schema_version": 1,
        "purpose": "final committed DVEB WENO requalification",
        "protocol": "DVEB_FINAL_REQUALIFICATION_PROTOCOL.md",
        "gradflow_commit": git_value(gradflow, "rev-parse", "HEAD"),
        "dveb_commit": dveb_commit,
        "program_sha256": PROGRAM_HASH,
        "module_sha256": MODULE_HASH,
        "native": {
            "dveb": freeze(arguments.dveb, output / "bin" / "dveb_shu3d", DVEB_EXPECTED),
            "ceiling": freeze(
                arguments.ceiling, output / "bin" / "shu3d_ceiling", CEILING_EXPECTED
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
