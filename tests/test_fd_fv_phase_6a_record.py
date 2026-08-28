from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_fd_fv_euler_phase_6a_record_verifies() -> None:
    result = subprocess.run(
        (
            sys.executable,
            str(ROOT / "experiments/fd_fv_euler/verify_phase6a.py"),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "FD/FV Euler Phase 6A verified" in result.stdout
