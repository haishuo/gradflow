#!/usr/bin/env python3
"""Freeze ABI v2 artifacts and bind them to the existing AOT preparation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DVEB_BUILD = Path("/mnt/projects/dveb/build/shu_euler3d_portable")
EXPECTED = {
    "program": "c6e5bd916f951ff412eac99863a74f8c98e5e14b044097a7ad59fe26f704c381",
    "module": "555c6cd2d7947160ce25182a860bab8288727d251d546c22232da27b59aa6260",
    "v1_header": "c14731d87423f95f9b19f216ddb7d4d2719e7196b6bd0d19205598ab23015c2a",
    "v1_library": "fb41b855e31e2ca2a8a989798be838b20d8220848d92189afcf5d94dc18f6663",
    "v2_header": "ad920101e3aa7ed4a41bf8ac86625e7c9149c58cd5ee218beb607750181ee2a4",
    "v2_library": "4541677eb21c6d93a7f0c6694ff78006c707b1f6b79c5752c7b497a841ff199c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def copy(source: Path, destination: Path, expected: str) -> dict[str, object]:
    if sha256(source) != expected:
        raise SystemExit(f"unexpected identity for {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(destination) != expected:
        raise RuntimeError(f"copy verification failed: {destination}")
    return record(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing directory: {output}")
    output.mkdir(parents=True)
    base = args.base_manifest.resolve()
    base_data = json.loads(base.read_text())
    if base_data.get("schema") != "gradflow-dveb-abi-bakeoff-preparation-v1":
        raise SystemExit("unexpected base bakeoff manifest")

    source = json.loads((DVEB_BUILD / "artifact.json").read_text())
    if source.get("schema") != "dveb-artifact-v3":
        raise SystemExit("expected DVEB artifact schema v3")
    if source["program_sha256"] != EXPECTED["program"] or source["module_sha256"] != EXPECTED["module"]:
        raise SystemExit("DVEB mathematical identity changed")
    frozen = output / "dveb"
    copies = {
        "v1_library": copy(DVEB_BUILD / source["abi"]["library"], frozen / source["abi"]["library"], EXPECTED["v1_library"]),
        "v1_header": copy(DVEB_BUILD / source["abi"]["header"], frozen / source["abi"]["header"], EXPECTED["v1_header"]),
        "v2_library": copy(DVEB_BUILD / source["device_abi"]["library"], frozen / source["device_abi"]["library"], EXPECTED["v2_library"]),
        "v2_header": copy(DVEB_BUILD / source["device_abi"]["header"], frozen / source["device_abi"]["header"], EXPECTED["v2_header"]),
    }
    shutil.copy2(DVEB_BUILD / "artifact.json", frozen / "artifact.json")
    copies["artifact_manifest"] = record(frozen / "artifact.json")

    inputs = {}
    for name in (
        "DVEB_DEVICE_ABI_E4_PROTOCOL.md",
        "device_abi_e4_worker.py",
        "prepare_dveb_device_e4.py",
        "run_dveb_device_e4.py",
    ):
        inputs[name] = record(HERE / name)
    for path in sorted((ROOT / "src" / "gradflow").glob("*.py")):
        inputs[f"src/gradflow/{path.name}"] = record(path)

    manifest = {
        "schema": "gradflow-dveb-device-e4-preparation-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "DVEB_DEVICE_ABI_E4_PROTOCOL.md",
        "seed": 20260827,
        "dveb_commit": "0d12788",
        "gradflow_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
            text=True, capture_output=True,
        ).stdout.strip(),
        "device_artifact_manifest": str((frozen / "artifact.json").resolve()),
        "device_artifacts": copies,
        "base_manifest": record(base),
        "inputs": inputs,
        "environment": {"platform": platform.platform()},
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(path), "sha256": sha256(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
