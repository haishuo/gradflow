from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "experiments"
    / "face_ownership_screen"
    / "evidence"
    / "face_ownership_20260830"
)


def test_face_ownership_screen_and_bounded_conclusion_verify() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(
                ROOT
                / "experiments"
                / "face_ownership_screen"
                / "verify_screen.py"
            ),
            str(EVIDENCE),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
