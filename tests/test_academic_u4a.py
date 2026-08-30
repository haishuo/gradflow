from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "experiments" / "academic_u4a" / "evidence" / "u4a_20260830"


def test_academic_u4a_compatibility_audit_verifies() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(ROOT / "experiments" / "academic_u4a" / "verify_u4a.py"),
            str(EVIDENCE),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
