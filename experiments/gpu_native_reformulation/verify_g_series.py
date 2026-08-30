#!/usr/bin/env python3
"""Verify the closed G0--G6 reckless-to-correct evidence chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/gpu_native_reformulation"
EVIDENCE = HERE / "evidence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(directory: Path, manifest_name: str = "SHA256SUMS") -> None:
    for line in (directory / manifest_name).read_text().splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        actual = sha256(directory / relative)
        if actual != expected:
            raise RuntimeError(
                f"checksum mismatch in {directory.name}/{manifest_name}: "
                f"{relative}: {actual}"
            )


def run_verifier(script: str, evidence: str) -> None:
    completed = subprocess.run(
        (sys.executable, str(HERE / script), str(EVIDENCE / evidence)),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{script} failed:\n{completed.stdout}\n{completed.stderr}"
        )


def main() -> None:
    g1 = EVIDENCE / "g1_u0_20260829"
    g3_recovery = EVIDENCE / "g3_recovery_20260829"
    verify_manifest(g1)
    verify_manifest(g1, "G2_SHA256SUMS")
    verify_manifest(g3_recovery)

    g2 = json.loads((g1 / "g2_damage.json").read_text())
    u0_error = g2["steps_1"]["forward_euler_same_first_dt"]["error"]
    assert u0_error["maximum_absolute_error"] == 0.0003268401329465931
    assert u0_error["rms_error_over_oracle_update_rms"] > 0.83
    assert u0_error["u0_oracle_update_cosine"] < 0.60

    r6 = json.loads((g3_recovery / "r6_damage.json").read_text())
    r6_error = r6["qualified_ssp_rk3"]["error"]
    assert r6_error["maximum_absolute_error"] < 4.0e-7
    assert r6_error["rms_error_over_oracle_update_rms"] < 0.0015
    assert r6_error["u0_oracle_update_cosine"] > 0.999998
    r6_timing = json.loads((g3_recovery / "r6_n128_s1.json").read_text())
    assert 4.0 < r6_timing["median_device_ms"] < 5.5

    for script, evidence in (
        ("verify_g3_qualification.py", "g3_qualification_20260829"),
        ("verify_g4.py", "g4_performance_20260829"),
        ("verify_g5.py", "g5_shared_pencil_20260829"),
        ("verify_g6.py", "g6_occupancy_20260830"),
    ):
        run_verifier(script, evidence)

    g4 = json.loads(
        (EVIDENCE / "g4_performance_20260829/campaign.json").read_text()
    )
    assert g4["primary_decision"]["schedule_hypothesis_supported"]
    assert g4["primary_decision"]["backend_qualification_implication"] is False
    g5 = json.loads(
        (EVIDENCE / "g5_shared_pencil_20260829/campaign.json").read_text()
    )
    assert not g5["primary_decision"]["successful_memory_recovery_pareto_result"]
    g6 = json.loads(
        (EVIDENCE / "g6_occupancy_20260830/campaign.json").read_text()
    )
    assert not g6["primary_decision"]["any_meaningful_occupancy_improvement"]

    synthesis = (HERE / "G_SERIES_SYNTHESIS.md").read_text()
    assert "G0--G6 complete; experimental program closed" in synthesis
    assert "No G7" in synthesis
    roadmap = (ROOT / "docs/ACADEMIC_COMPLETION_ROADMAP.md").read_text()
    assert "A2. Run the core arbitrary-order performance matrix" in roadmap
    assert "G7" in roadmap and "explicitly deferred" in roadmap.lower()
    print("Closed G0--G6 reckless-to-correct evidence chain verifies.")


if __name__ == "__main__":
    main()
