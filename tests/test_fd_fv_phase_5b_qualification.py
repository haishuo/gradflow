from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_phase_5b_record_verifies() -> None:
    script = ROOT / "experiments/fd_fv_nonlinear/verify_phase_5b.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "FD/FV nonlinear Phase 5B verified" in result.stdout
