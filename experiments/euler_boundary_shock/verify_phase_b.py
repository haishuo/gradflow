#!/usr/bin/env python3
"""Verify the frozen Phase-B checksums and qualification invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.euler_boundary_shock.verify_phase_a import (
    DEFAULT_RECORD as PHASE_A_RECORD,
    verify as verify_phase_a,
)


DEFAULT_RECORD = Path(__file__).resolve().parent / "results" / "phase_b_20260827"
REPRESENTATIVE_ORDERS = (5, 11, 15)
PILOT_ORDERS = (7, 9, 13)
EXPECTED_ARRAYS = {
    f"{problem}_order{order}_n800.npz"
    for problem in ("sod", "shu_osher")
    for order in REPRESENTATIVE_ORDERS
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
    if set(checked_hashes) != EXPECTED_ARRAYS | {"qualification.json"}:
        raise RuntimeError("unexpected Phase-B artifact inventory")

    manifest = json.loads((record / "qualification.json").read_text())
    if manifest["schema"] != "gradflow.euler_boundary_shock.phase_b.v1":
        raise RuntimeError("unexpected Phase-B schema")
    if not manifest["source_worktree_clean"]:
        raise RuntimeError("Phase-B record did not originate from a clean source tree")
    if manifest["decision"] != "PASS" or not manifest["shock_study"]["passed"]:
        raise RuntimeError("Phase-B qualification decision is not PASS")
    if any(manifest["claim_boundary"].values()):
        raise RuntimeError("Phase-B claim boundary contains an unexpected true value")
    if not manifest["static_inspection"]["passed"]:
        raise RuntimeError("Phase-B static transfer inspection failed")

    phase_a = verify_phase_a(PHASE_A_RECORD)
    recorded_phase_a = manifest["phase_a"]
    if recorded_phase_a["source_commit"] != phase_a["source_commit"]:
        raise RuntimeError("Phase-A source identity mismatch")
    for name, key in (
        ("manifest.json", "manifest_sha256"),
        ("thresholds.json", "thresholds_sha256"),
        (
            "shu_osher_fv_wenoz_hllc_t1p8_n12800.npz",
            "shu_osher_reference_sha256",
        ),
    ):
        if recorded_phase_a[key] != _sha256(PHASE_A_RECORD / name):
            raise RuntimeError(f"Phase-A dependency mismatch for {name}")

    decisions = manifest["shock_study"]["representative_order_decisions"]
    if any(not decisions[str(order)]["passed"] for order in REPRESENTATIVE_ORDERS):
        raise RuntimeError("a representative-order shock gate failed")
    pilots = manifest["shock_study"]["pilot_order_admissibility"]
    if any(
        not pilots[str(order)][problem]
        for order in PILOT_ORDERS
        for problem in ("sod", "shu_osher")
    ):
        raise RuntimeError("a pilot-order shock run was not admissible")

    arrays: dict[str, object] = {}
    for name in sorted(EXPECTED_ARRAYS):
        with np.load(record / name) as archive:
            x = archive["x"]
            primitive = archive["primitive"]
            conserved = archive["conserved"]
        if x.shape != (800,):
            raise RuntimeError(f"unexpected coordinate shape in {name}")
        if primitive.shape != (3, 800) or conserved.shape != (3, 800):
            raise RuntimeError(f"unexpected state shape in {name}")
        if not np.all(np.isfinite(primitive)) or not np.all(np.isfinite(conserved)):
            raise RuntimeError(f"nonfinite state in {name}")
        if np.min(primitive[0]) <= 0.0 or np.min(primitive[2]) <= 0.0:
            raise RuntimeError(f"nonpositive density or pressure in {name}")
        arrays[name] = {
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
