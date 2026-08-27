#!/usr/bin/env python3
"""Verify Phase-A checksums, schemas, and numerical array invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_RECORD = Path(__file__).resolve().parent / "results" / "phase_a_20260827"
EXPECTED_ARRAYS = {
    "sod_exact_t0p2_n8192.npz": 8192,
    "sod_fv_wenoz_hllc_t0p2_n1600.npz": 1600,
    "shu_osher_fv_wenoz_hllc_t1p8_n12800.npz": 12800,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(record: Path) -> dict[str, object]:
    checksum_path = record / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    checked_hashes: dict[str, str] = {}
    for line in checksum_path.read_text().splitlines():
        expected, name = line.split("  ", maxsplit=1)
        artifact = record / name
        actual = _sha256(artifact)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {name}")
        checked_hashes[name] = actual

    manifest = json.loads((record / "manifest.json").read_text())
    thresholds = json.loads((record / "thresholds.json").read_text())
    if manifest["schema"] != "gradflow.euler_boundary_shock.phase_a.v1":
        raise RuntimeError("unexpected Phase-A schema")
    if not manifest["source_worktree_clean"]:
        raise RuntimeError("Phase-A record did not originate from a clean source tree")
    if any(manifest["claim_boundary"].values()):
        raise RuntimeError("Phase-A claim boundary contains an unexpected true value")
    if not thresholds["derivation"][
        "selected_without_gradflow_boundary_implementation"
    ]:
        raise RuntimeError("thresholds were not marked as preimplementation")

    arrays: dict[str, object] = {}
    for name, size in EXPECTED_ARRAYS.items():
        with np.load(record / name) as archive:
            x = archive["x"]
            primitive = archive["primitive"]
            conserved = archive["conserved"]
        if x.shape != (size,):
            raise RuntimeError(f"unexpected coordinate shape in {name}")
        if primitive.shape != (3, size) or conserved.shape != (3, size):
            raise RuntimeError(f"unexpected state shape in {name}")
        if not np.all(np.isfinite(primitive)) or not np.all(np.isfinite(conserved)):
            raise RuntimeError(f"nonfinite state in {name}")
        if np.min(primitive[0]) <= 0.0 or np.min(primitive[2]) <= 0.0:
            raise RuntimeError(f"nonpositive density or pressure in {name}")
        arrays[name] = {
            "points": size,
            "minimum_density": float(np.min(primitive[0])),
            "minimum_pressure": float(np.min(primitive[2])),
        }
    return {
        "passed": True,
        "record": str(record),
        "source_commit": manifest["source_commit"],
        "checked_hashes": checked_hashes,
        "arrays": arrays,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path, nargs="?", default=DEFAULT_RECORD)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.record.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
