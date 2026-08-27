from __future__ import annotations

import inspect
import math

import pytest
import torch

import gradflow.euler1d as implementation
import gradflow.euler3d as shared_implementation
from gradflow import (
    EULER_GAMMA,
    QUALIFIED_EULER_WENO_ORDERS,
    euler1d_cfl_timestep,
    euler1d_rhs,
    euler1d_rhs_with_boundary_fluxes,
    euler1d_ssp_rk3_step,
)


def primitive_state(
    density: torch.Tensor,
    velocity: torch.Tensor | float,
    pressure: torch.Tensor | float,
) -> torch.Tensor:
    velocity_tensor = torch.as_tensor(
        velocity, dtype=density.dtype, device=density.device
    ).expand_as(density)
    pressure_tensor = torch.as_tensor(
        pressure, dtype=density.dtype, device=density.device
    ).expand_as(density)
    energy = pressure_tensor / (EULER_GAMMA - 1.0) + (
        0.5 * density * velocity_tensor.square()
    )
    return torch.stack((density, density * velocity_tensor, energy))


def entropy_wave(
    points: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    x = (torch.arange(points, dtype=dtype, device=device) + 0.5) / points
    density = 1.0 + 0.1 * torch.sin(2.0 * math.pi * x)
    density_derivative = 0.2 * math.pi * torch.cos(2.0 * math.pi * x)
    velocity = 0.7
    state = primitive_state(density, velocity, 1.0)
    density_rhs = -velocity * density_derivative
    exact = torch.stack(
        (
            density_rhs,
            velocity * density_rhs,
            0.5 * velocity**2 * density_rhs,
        )
    )
    return state, exact


@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
@pytest.mark.parametrize("boundary", ("periodic", "transmissive"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_uniform_state(order: int, boundary: str, dtype: torch.dtype) -> None:
    density = torch.full((max(order, 19),), 1.2, dtype=dtype)
    state = primitive_state(density, 0.3, 0.9)
    rhs = euler1d_rhs(state, 0.05, order=order, boundary=boundary)
    tolerance = 2.0e-5 if dtype is torch.float32 else 2.0e-12
    assert float(torch.max(torch.abs(rhs))) <= tolerance


@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
def test_periodic_overlap_with_qualified_line_algebra(order: int) -> None:
    points = max(order + 4, 21)
    state, _ = entropy_wave(points)
    duplicated = torch.cat((state, state[:, :1]), dim=-1)
    scheme = shared_implementation._EULER_WENO_SCHEMES[order]
    expected = shared_implementation._generated_line_rhs(
        duplicated, float(points), scheme
    )[:, :-1]
    actual = euler1d_rhs(
        state, 1.0 / points, order=order, boundary="periodic"
    )
    torch.testing.assert_close(actual, expected, rtol=2.0e-13, atol=2.0e-13)


@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
def test_entropy_wave_spatial_convergence(order: int) -> None:
    sizes = (24, 36, 54, 81)
    roundoff_floor = 1.0e-11
    errors = []
    for points in sizes:
        state, exact = entropy_wave(points)
        actual = euler1d_rhs(
            state, 1.0 / points, order=order, boundary="periodic"
        )
        errors.append(float(torch.sqrt(torch.mean((actual - exact).square()))))
    assert all(
        fine < coarse
        for coarse, fine in zip(errors, errors[1:])
        if coarse > roundoff_floor and fine > roundoff_floor
    )
    rates = [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors, errors[1:], sizes, sizes[1:]
        )
    ]
    observable_rates = [
        rate
        for rate, coarse, fine in zip(rates, errors, errors[1:])
        if coarse > roundoff_floor and fine > roundoff_floor
    ]
    if observable_rates:
        assert max(observable_rates) >= order - 2
    else:
        assert errors[0] <= roundoff_floor


@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
@pytest.mark.parametrize("boundary", ("periodic", "transmissive"))
def test_boundary_flux_conservation(order: int, boundary: str) -> None:
    points = 43
    x = (torch.arange(points, dtype=torch.float64) + 0.5) / points
    density = 1.1 + 0.07 * torch.sin(2.0 * math.pi * x)
    velocity = 0.25 + 0.03 * torch.cos(2.0 * math.pi * x)
    pressure = 0.9 + 0.04 * torch.sin(4.0 * math.pi * x)
    state = primitive_state(density, velocity, pressure)
    dx = 1.0 / points
    rhs, fluxes = euler1d_rhs_with_boundary_fluxes(
        state, dx, order=order, boundary=boundary
    )
    residual = torch.abs(dx * torch.sum(rhs, dim=-1) + fluxes[:, 1] - fluxes[:, 0])
    scale = torch.finfo(state.dtype).eps * torch.clamp_min(
        dx * torch.sum(torch.abs(rhs), dim=-1)
        + torch.abs(fluxes[:, 0])
        + torch.abs(fluxes[:, 1]),
        1.0,
    )
    assert float(torch.max(residual / scale)) <= 64.0


@pytest.mark.parametrize("order", (5, 11, 15))
def test_boundary_sensitive_directional_derivative(order: int) -> None:
    points = 19
    x = (torch.arange(points, dtype=torch.float64) + 0.5) / points
    base = primitive_state(
        1.1 + 0.05 * torch.sin(1.3 * math.pi * x),
        0.2 + 0.02 * x,
        1.0 + 0.03 * torch.cos(0.7 * math.pi * x),
    )
    direction = torch.sin(
        torch.arange(base.numel(), dtype=base.dtype).reshape_as(base) + 0.3
    )
    direction = direction / torch.linalg.vector_norm(direction)
    weights = torch.linspace(0.3, 1.7, points, dtype=base.dtype)
    component_weights = torch.tensor([1.0, -0.2, 0.1], dtype=base.dtype)[:, None]

    def objective(values: torch.Tensor) -> torch.Tensor:
        advanced = euler1d_ssp_rk3_step(
            values,
            1.0 / points,
            2.0e-4,
            order=order,
            boundary="transmissive",
        )
        return torch.sum(advanced * component_weights * weights)

    state = base.detach().requires_grad_(True)
    gradient = torch.autograd.grad(objective(state), state)[0]
    actual = torch.sum(gradient * direction)
    epsilon = 1.0e-6
    expected = (
        objective(base + epsilon * direction)
        - objective(base - epsilon * direction)
    ) / (2.0 * epsilon)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-7)


@pytest.mark.parametrize("order", (5, 11, 15))
@pytest.mark.parametrize("boundary", ("periodic", "transmissive"))
def test_compile_fullgraph(order: int, boundary: str) -> None:
    state, _ = entropy_wave(max(order + 4, 21))

    def rhs(values: torch.Tensor) -> torch.Tensor:
        return euler1d_rhs(
            values,
            1.0 / values.shape[-1],
            order=order,
            boundary=boundary,
        )

    expected = rhs(state)
    compiled = torch.compile(rhs, fullgraph=True)
    actual = compiled(state)
    torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=2.0e-12)
    explanation = torch._dynamo.explain(rhs)(state)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("order", QUALIFIED_EULER_WENO_ORDERS)
@pytest.mark.parametrize("boundary", ("periodic", "transmissive"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_cpu_cuda_agreement(
    order: int, boundary: str, dtype: torch.dtype
) -> None:
    state, _ = entropy_wave(37, dtype=dtype)
    expected = euler1d_rhs(
        state, 1.0 / 37.0, order=order, boundary=boundary
    )
    actual = euler1d_rhs(
        state.cuda(), 1.0 / 37.0, order=order, boundary=boundary
    ).cpu()
    tolerance = 3.0e-4 if dtype is torch.float32 else 5.0e-11
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=tolerance)


def test_cfl_remains_on_device_and_positive() -> None:
    state, _ = entropy_wave(23)
    timestep = euler1d_cfl_timestep(state, 1.0 / 23.0)
    assert timestep.device == state.device
    assert timestep.ndim == 0
    assert float(timestep) > 0.0


def test_validation_rejects_unsupported_contracts() -> None:
    state, _ = entropy_wave(19)
    with pytest.raises(ValueError, match="order"):
        euler1d_rhs(state, 1.0 / 19.0, order=3)
    with pytest.raises(ValueError, match="boundary"):
        euler1d_rhs(state, 1.0 / 19.0, boundary="wall")
    with pytest.raises(ValueError, match="shape"):
        euler1d_rhs(state[:2], 1.0 / 19.0)
    with pytest.raises(ValueError, match="positive"):
        euler1d_rhs(state, 0.0)


def test_numerical_loop_has_no_forbidden_transfer_or_custom_code() -> None:
    source = inspect.getsource(implementation)
    shared_source = inspect.getsource(
        shared_implementation._generated_bounded_line_rhs
    )
    for forbidden in (
        ".cpu(",
        ".cuda(",
        ".to(",
        ".item(",
        ".numpy(",
        "torch.library",
        "triton",
    ):
        assert forbidden not in source
        assert forbidden not in shared_source
