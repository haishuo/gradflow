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
    / "g5_shared_pencil_20260829"
)


def test_g5_shared_pencil_record_and_negative_conclusion_verify() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(
                ROOT
                / "experiments"
                / "gpu_native_reformulation"
                / "verify_g5.py"
            ),
            str(EVIDENCE),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
