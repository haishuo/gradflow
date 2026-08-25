#!/usr/bin/env python3
"""Prepare hash-locked native binaries and fixed-shape AOT packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


EXPERIMENT = Path(__file__).resolve().parent
DVEB_EXPECTED_SHA256 = (
    "884d874308dc7b1fd12f56491ae9addd85d1872ffcea4c2f26a0157c9c55c03c"
)
CEILING_EXPECTED_SHA256 = (
    "873a9227196664398012e7d42a27e29ec9cd3610c45a4c61ab40a0688aed3caa"
)
DEFAULT_DVEB = Path(
    "/mnt/projects/dveb/build/shu_euler3d_portable/shu_euler3d_portable"
)
DEFAULT_CEILING = Path(
    "/mnt/projects/dveb/tools/shu_euler3d_ceiling/build/shu_ceiling"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_json(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"command emitted no JSON object:\n{text}")


def run_json(command: list[str], *, env: dict[str, str] | None = None) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=EXPERIMENT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    record = last_json(completed.stdout)
    record["external_seconds"] = time.perf_counter() - started
    if completed.stderr.strip():
        record["stderr"] = completed.stderr.strip()
    return record


def checked_copy(source: Path, destination: Path, expected: str) -> dict[str, object]:
    source = source.resolve()
    if not source.is_file():
        raise SystemExit(f"missing native input: {source}")
    observed = sha256(source)
    if observed != expected:
        raise SystemExit(
            f"refusing changed native input {source}: expected {expected}, got {observed}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o755)
    copied = sha256(destination)
    if copied != expected:
        raise RuntimeError(f"copied hash mismatch for {destination}")
    return {
        "source": str(source),
        "frozen_copy": str(destination.resolve()),
        "sha256": copied,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--dveb", type=Path, default=DEFAULT_DVEB)
    parser.add_argument("--ceiling", type=Path, default=DEFAULT_CEILING)
    arguments = parser.parse_args()
    output = arguments.output_dir.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite existing preparation: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)

    native = {
        "dveb": checked_copy(
            arguments.dveb, output / "bin" / "dveb_shu3d", DVEB_EXPECTED_SHA256
        ),
        "ceiling": checked_copy(
            arguments.ceiling,
            output / "bin" / "shu3d_ceiling",
            CEILING_EXPECTED_SHA256,
        ),
    }

    packages: dict[str, object] = {}
    for size in arguments.sizes:
        package = output / "aot" / f"shu3d_n{size}.pt2"
        package.parent.mkdir(parents=True, exist_ok=True)
        build_environment = os.environ.copy()
        build_environment["TORCHINDUCTOR_CACHE_DIR"] = str(
            output / "aot_build_cache" / f"n{size}"
        )
        build_record = run_json(
            [
                sys.executable,
                str(EXPERIMENT / "build_aot_package.py"),
                "--size",
                str(size),
                "--output",
                str(package),
            ],
            env=build_environment,
        )
        runtime_cache = output / "aot_runtime_cache" / f"n{size}"
        runtime_environment = os.environ.copy()
        runtime_environment["TORCHINDUCTOR_CACHE_DIR"] = str(runtime_cache)
        preparation_record = run_json(
            [
                sys.executable,
                str(EXPERIMENT / "bakeoff_worker.py"),
                "--lane",
                "aot",
                "--size",
                str(size),
                "--steps",
                "1",
                "--package",
                str(package),
            ],
            env=runtime_environment,
        )
        packages[str(size)] = {
            "path": str(package),
            "sha256": sha256(package),
            "bytes": package.stat().st_size,
            "runtime_cache": str(runtime_cache),
            "build": build_record,
            "extraction_cache_preparation": preparation_record,
        }
        print(json.dumps({"prepared_size": size, **packages[str(size)]}), flush=True)

    manifest = {
        "schema_version": 1,
        "purpose": "GradFlow DVEB-inclusive bakeoff preparation",
        "sizes": arguments.sizes,
        "python": sys.executable,
        "native": native,
        "aot_packages": packages,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "sizes": arguments.sizes}))


if __name__ == "__main__":
    main()
