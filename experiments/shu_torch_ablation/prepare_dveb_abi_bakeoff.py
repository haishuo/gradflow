#!/usr/bin/env python3
"""Prepare immutable artifacts for the frozen forced-target ABI bakeoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
DVEB_BUILD = Path("/mnt/projects/dveb/build/shu_euler3d_portable")
CEILING = Path("/mnt/projects/dveb/tools/shu_euler3d_ceiling/build/shu_ceiling")
FORTRAN_SOURCE = EXPERIMENT / "fortran" / "shu_euler_3d.f90"
EXPECTED = {
    "library": "cfa939a5b492ed5711a432391d604ceda65ed55c6df7a4a77b6bfabdd7bd1b1c",
    "header": "c14731d87423f95f9b19f216ddb7d4d2719e7196b6bd0d19205598ab23015c2a",
    "program": "c6e5bd916f951ff412eac99863a74f8c98e5e14b044097a7ad59fe26f704c381",
    "module": "555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260",
    "ceiling": "873a9227196664398012e7d42a27e29ec9cd3610c45a4c61ab40a0688aed3caa",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.strip()


def last_json(text: str) -> dict[str, object]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"command emitted no JSON object:\n{text}")


def run_json(
    command: list[str], *, environment: dict[str, str] | None = None
) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=EXPERIMENT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"preparation command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    record = last_json(completed.stdout)
    record["external_seconds"] = elapsed
    record["command"] = command
    if completed.stderr.strip():
        record["stderr"] = completed.stderr.strip()
    return record


def verified_copy(source: Path, destination: Path, expected: str | None = None) -> dict[str, object]:
    source = source.resolve()
    if not source.is_file():
        raise SystemExit(f"missing preparation input: {source}")
    observed = sha256(source)
    if expected is not None and observed != expected:
        raise SystemExit(f"hash mismatch for {source}: expected {expected}, got {observed}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied = sha256(destination)
    if copied != observed:
        raise RuntimeError(f"copy verification failed for {destination}")
    return {
        "source": str(source),
        "frozen_copy": str(destination.resolve()),
        "sha256": copied,
        "bytes": destination.stat().st_size,
    }


def hash_record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--sizes", nargs="+", type=int, default=[8, 16, 32, 64, 96, 128, 160])
    args = parser.parse_args()
    counted_sizes = list(dict.fromkeys(args.sizes))
    prepared_sizes = sorted(set(counted_sizes) | {6})
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing preparation directory: {output}")
    output.mkdir(parents=True)

    source_artifact = json.loads((DVEB_BUILD / "artifact.json").read_text())
    if source_artifact.get("schema") != "dveb-artifact-v2":
        raise SystemExit("unexpected DVEB artifact schema")
    if source_artifact.get("program_sha256") != EXPECTED["program"]:
        raise SystemExit("DVEB program identity changed")
    if source_artifact.get("module_sha256") != EXPECTED["module"]:
        raise SystemExit("DVEB module identity changed")

    dveb_dir = output / "dveb"
    library = verified_copy(
        DVEB_BUILD / source_artifact["abi"]["library"],
        dveb_dir / source_artifact["abi"]["library"], EXPECTED["library"],
    )
    header = verified_copy(
        DVEB_BUILD / source_artifact["abi"]["header"],
        dveb_dir / source_artifact["abi"]["header"], EXPECTED["header"],
    )
    artifact_copy = verified_copy(DVEB_BUILD / "artifact.json", dveb_dir / "artifact.json")
    program = verified_copy(
        Path(source_artifact["source"]), dveb_dir / "source" / Path(source_artifact["source"]).name,
        EXPECTED["program"],
    )
    module = verified_copy(
        Path(source_artifact["module"]), dveb_dir / "source" / Path(source_artifact["module"]).name,
        EXPECTED["module"],
    )

    ceiling = verified_copy(CEILING, output / "bin" / "shu3d_ceiling", EXPECTED["ceiling"])
    fortran_copy = verified_copy(FORTRAN_SOURCE, output / "source" / FORTRAN_SOURCE.name)
    fortran_binary = output / "bin" / "shu_euler_3d"
    fortran_binary.parent.mkdir(parents=True, exist_ok=True)
    fortran_command = [
        "gfortran", "-O3", "-march=native", "-std=f2008",
        str(fortran_copy["frozen_copy"]), "-o", str(fortran_binary),
    ]
    build_started = time.perf_counter()
    subprocess.run(fortran_command, check=True, cwd=output / "source")
    fortran_build_seconds = time.perf_counter() - build_started
    fortran = hash_record(fortran_binary)
    fortran["source"] = fortran_copy
    fortran["build_command"] = fortran_command
    fortran["build_seconds"] = fortran_build_seconds

    # Preserve the environment entry point. Resolving a venv symlink can turn
    # it into the system interpreter and silently discard that environment's
    # site-packages.
    python = str(args.python.absolute())
    packages: dict[str, object] = {}
    compile_caches: dict[str, object] = {}
    for size in prepared_sizes:
        key = str(size)
        package = output / "aot" / f"shu3d_n{size}.pt2"
        package.parent.mkdir(parents=True, exist_ok=True)
        aot_build_cache = output / "cache" / "aot-build" / f"n{size}"
        build_environment = os.environ.copy()
        build_environment["TORCHINDUCTOR_CACHE_DIR"] = str(aot_build_cache)
        build = run_json([
            python, str(EXPERIMENT / "build_abi_aot_package.py"),
            "--size", key, "--output", str(package),
        ], environment=build_environment)

        aot_runtime_cache = output / "cache" / "aot-runtime" / f"n{size}"
        aot_environment = os.environ.copy()
        aot_environment["TORCHINDUCTOR_CACHE_DIR"] = str(aot_runtime_cache)
        aot_prepare = run_json([
            python, str(EXPERIMENT / "abi_bakeoff_worker.py"),
            "--lane", "aot-inductor", "--endpoint", "prepare",
            "--size", key, "--steps", "1", "--package", str(package),
        ], environment=aot_environment)
        packages[key] = {
            **hash_record(package),
            "build_cache": str(aot_build_cache),
            "runtime_cache": str(aot_runtime_cache),
            "build": build,
            "extraction_cache_preparation": aot_prepare,
        }

        compile_cache = output / "cache" / "torch-compile" / f"n{size}"
        compile_environment = os.environ.copy()
        compile_environment["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache)
        compile_prepare = run_json([
            python, str(EXPERIMENT / "abi_bakeoff_worker.py"),
            "--lane", "persistent-compile", "--endpoint", "prepare",
            "--size", key, "--steps", "1",
        ], environment=compile_environment)
        compile_caches[key] = {
            "path": str(compile_cache),
            "preparation": compile_prepare,
        }
        print(json.dumps({"prepared_size": size, "package": packages[key]}, sort_keys=True), flush=True)

    inputs = {}
    for name in (
        "DVEB_ABI_BAKEOFF_PROTOCOL.md",
        "abi_bakeoff_worker.py",
        "analyze_dveb_abi_bakeoff.py",
        "build_abi_aot_package.py",
        "check_dveb_abi_bakeoff.py",
        "prepare_dveb_abi_bakeoff.py",
        "run_dveb_abi_bakeoff.py",
    ):
        inputs[name] = hash_record(EXPERIMENT / name)
    for path in (ROOT / "src" / "gradflow").glob("*.py"):
        inputs[f"src/gradflow/{path.name}"] = hash_record(path)

    manifest = {
        "schema": "gradflow-dveb-abi-bakeoff-preparation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "DVEB_ABI_BAKEOFF_PROTOCOL.md",
        "gradflow_commit_before_harness": "0f37bb463cf14dc690701071adcec7ad339e6295",
        "gradflow_protocol_commit": "e5df176",
        "gradflow_worktree_commit_at_preparation": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "dveb_commit": "f71d86717c065841c002b41287ff943e9f0a7898",
        "counted_and_capacity_sizes": counted_sizes,
        "prepared_sizes": prepared_sizes,
        "python": python,
        "dveb": {
            "manifest": artifact_copy,
            "library": library,
            "header": header,
            "program": program,
            "module": module,
        },
        "native": {"fortran": fortran, "ceiling": ceiling},
        "aot_packages": packages,
        "compile_caches": compile_caches,
        "inputs": inputs,
        "environment": {
            "platform": platform.platform(),
            "python_version": command_output([python, "--version"]),
            "torch": command_output([python, "-c", "import torch; print(torch.__version__, torch.version.cuda)"]),
            "gfortran": command_output(["gfortran", "--version"]),
            "nvcc": command_output(["nvcc", "--version"]),
            "kernel": platform.release(),
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
