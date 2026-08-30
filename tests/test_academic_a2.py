from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments" / "academic_a2" / "evidence" / "a2_20260830"


def test_academic_a2_evidence_and_performance_freeze_verify() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(ROOT / "experiments" / "academic_a2" / "verify_a2.py"),
            str(EVIDENCE),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
