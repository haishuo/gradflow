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
    / "g6_occupancy_20260830"
)


def test_g6_occupancy_record_and_negative_conclusion_verify() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(
                ROOT
                / "experiments"
                / "gpu_native_reformulation"
                / "verify_g6.py"
            ),
            str(EVIDENCE),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
