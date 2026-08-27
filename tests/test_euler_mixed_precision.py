import torch

from gradflow import (
    PRECISION_BLOCKS,
    WENOJSPrecisionPolicy,
    euler1d_rhs,
    euler_ssp_rk3_step,
    euler_weno_rhs,
    periodic_vortex,
)


def _all_float64() -> WENOJSPrecisionPolicy:
    return WENOJSPrecisionPolicy(
        **{block: torch.float64 for block in PRECISION_BLOCKS}
    )


def _combined() -> WENOJSPrecisionPolicy:
    return WENOJSPrecisionPolicy(
        indicators=torch.float32,
        weight_formation=torch.float32,
    )


def test_explicit_float64_policy_preserves_periodic_euler_default() -> None:
    state, spacing = periodic_vortex((7, 7, 7), dtype=torch.float64)
    expected = euler_weno_rhs(state, spacing, order=7)
    actual = euler_weno_rhs(
        state, spacing, order=7, precision=_all_float64()
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    dt = torch.tensor(1.0e-4, dtype=state.dtype)
    expected_step = euler_ssp_rk3_step(state, spacing, dt, order=7)
    actual_step = euler_ssp_rk3_step(
        state, spacing, dt, order=7, precision=_all_float64()
    )
    torch.testing.assert_close(actual_step, expected_step, rtol=0.0, atol=0.0)


def test_mixed_policy_preserves_euler_state_dtype_and_device() -> None:
    state, spacing = periodic_vortex((7, 7, 7), dtype=torch.float64)
    rhs = euler_weno_rhs(state, spacing, order=7, precision=_combined())
    assert rhs.dtype is state.dtype
    assert rhs.device == state.device
    assert torch.isfinite(rhs).all()


def test_bounded_euler_accepts_same_explicit_policy() -> None:
    points = 31
    density = torch.linspace(0.9, 1.1, points, dtype=torch.float64)
    velocity = torch.full_like(density, 0.2)
    pressure = torch.full_like(density, 1.0)
    state = torch.stack(
        (
            density,
            density * velocity,
            pressure / 0.4 + 0.5 * density * velocity.square(),
        )
    )
    expected = euler1d_rhs(
        state, 1.0 / points, order=11, boundary="transmissive"
    )
    actual = euler1d_rhs(
        state,
        1.0 / points,
        order=11,
        boundary="transmissive",
        precision=_combined(),
    )
    assert actual.dtype is torch.float64
    assert torch.isfinite(actual).all()
    assert torch.max(torch.abs(actual - expected)) < 1.0e-4
