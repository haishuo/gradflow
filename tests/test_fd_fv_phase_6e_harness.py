from __future__ import annotations

import numpy as np
from pathlib import Path
import subprocess
import sys
import torch

from experiments.fd_fv_euler.run_phase6e_repro import comparison
from experiments.fd_fv_euler.phase6c_problem import shock_initial
from experiments.fd_fv_euler.phase6e_aot_model import HostControlledAdvance


def test_phase6e_roundoff_envelope_accepts_exact_arrays() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    result = comparison(
        values,
        values.copy(),
        steps=7000,
        reference_name="reference.npy",
        actual_name="actual.npy",
    )
    assert result["exact_equal"]
    assert result["passed"]


def test_phase6e_roundoff_envelope_rejects_material_difference() -> None:
    reference = np.ones((3, 4), dtype=np.float64)
    actual = reference.copy()
    actual[1, 2] += 1.0e-4
    result = comparison(
        reference,
        actual,
        steps=7000,
        reference_name="reference.npy",
        actual_name="actual.npy",
    )
    assert result["maximum_location"] == [1, 2]
    assert not result["passed"]


def test_committed_phase6e_lane_a_record_verifies() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        (
            sys.executable,
            str(root / "experiments/fd_fv_euler/verify_phase6e_repro.py"),
        ),
        cwd=root,
        check=False,
    )
    assert completed.returncode == 0


def test_phase6e_host_advance_matches_frozen_stage_algebra() -> None:
    state = shock_initial("fd", "sod", 800)
    remaining = state.new_tensor(0.2)
    module = HostControlledAdvance("fd", "sod")
    next_state, dt, density, pressure, finite = module(state, remaining)
    assert next_state.shape == state.shape
    assert next_state.dtype == state.dtype == torch.float64
    assert dt.ndim == 0 and 0.0 < float(dt) <= 0.2
    assert density.shape == pressure.shape == finite.shape == (3,)
    assert bool(torch.all(density > 0.0))
    assert bool(torch.all(pressure > 0.0))
    assert bool(torch.all(finite))
