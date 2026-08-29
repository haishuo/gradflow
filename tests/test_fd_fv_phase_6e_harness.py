from __future__ import annotations

import numpy as np

from experiments.fd_fv_euler.run_phase6e_repro import comparison


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
