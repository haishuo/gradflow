#!/usr/bin/env python3
"""Check complete-step float32 parity against both Fortran comparators."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from shu_euler_torch import cfl_timestep, periodic_vortex, ssp_rk3_step


EXPERIMENT_DIR = Path(__file__).resolve().parent
FORTRAN_2D = EXPERIMENT_DIR.parent / "fortran_scaling" / "build" / "weno_dynamic"
FORTRAN_3D = EXPERIMENT_DIR / "build" / "shu_euler_3d"


def _torch_step(intervals: tuple[int, ...]) -> np.ndarray:
    state, spacing = periodic_vortex(intervals, dtype=torch.float32)
    dt = torch.minimum(
        cfl_timestep(state, spacing, 0.1),
        torch.tensor(0.001, dtype=torch.float32),
    )
    return ssp_rk3_step(state, spacing, dt).numpy()


def _fortran_2d(scratch: Path, size: int) -> np.ndarray:
    environment = dict(os.environ, WENO_WRITE_SOLUTION="1")
    subprocess.run(
        [str(FORTRAN_2D)],
        input=f"3\n{size} {size}\n0.1\n1\n0.001\n1\n0\n",
        text=True,
        cwd=scratch,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    raw = (scratch / "restart").read_bytes()
    record_bytes = int.from_bytes(raw[:4], byteorder="little")
    payload = np.frombuffer(raw[4 : 4 + record_bytes], dtype="<f4")
    return payload[1:].reshape(4, size + 1, size + 1)


def _fortran_3d(scratch: Path, size: int) -> np.ndarray:
    output = scratch / "state.bin"
    environment = dict(os.environ, WENO_WRITE_STATE=str(output))
    subprocess.run(
        [str(FORTRAN_3D)],
        input=f"{size} {size} {size}\n0.1\n1\n0.001\n",
        text=True,
        cwd=scratch,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = np.frombuffer(output.read_bytes(), dtype="<f4")
    return payload[1:].reshape(5, size + 1, size + 1, size + 1)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        scratch = Path(temporary_directory)
        fortran_2d = _fortran_2d(scratch, 10)
        torch_2d = _torch_step((10, 10))
        error_2d = float(np.max(np.abs(fortran_2d - torch_2d)))

        fortran_3d = _fortran_3d(scratch, 6)
        torch_3d = _torch_step((6, 6, 6))
        error_3d = float(np.max(np.abs(fortran_3d - torch_3d)))

    print(f"2-D complete-step Linf: {error_2d:.9e}")
    print(f"3-D complete-step Linf: {error_3d:.9e}")
    if error_2d > 1.0e-6 or error_3d > 1.0e-6:
        raise SystemExit("FAIL: Fortran/PyTorch parity tolerance exceeded")
    print("PASS: both complete-step comparisons are within float32 tolerance")


if __name__ == "__main__":
    main()
