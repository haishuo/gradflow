from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_closed_gpu_native_g_series_verifies() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            str(
                ROOT
                / "experiments"
                / "gpu_native_reformulation"
                / "verify_g_series.py"
            ),
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
