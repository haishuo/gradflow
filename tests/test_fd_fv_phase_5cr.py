from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_phase_5cr_record_verifies() -> None:
    result = subprocess.run(
        (
            sys.executable,
            str(
                ROOT
                / "experiments/fd_fv_nonlinear/verify_phase5cr.py"
            ),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Phase 5CR verified" in result.stdout
