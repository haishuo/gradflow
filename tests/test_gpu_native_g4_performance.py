from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "experiments"
    / "gpu_native_reformulation"
    / "evidence"
    / "g4_performance_20260829"
)


def test_g4_performance_record_and_claim_boundary_verify() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(
                ROOT
                / "experiments"
                / "gpu_native_reformulation"
                / "verify_g4.py"
            ),
            str(EVIDENCE),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
