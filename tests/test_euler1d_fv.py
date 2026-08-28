from __future__ import annotations

import inspect
import math

import pytest
import torch

import gradflow.euler1d_fv as implementation
from experiments.fd_fv_euler.phase6a_oracle import entropy_average
from gradflow import (
    EULER1D_FV_FORMULATION_ID,
    EULER_GAMMA,
    euler1d_fv_cfl_timestep,
    euler1d_fv_rhs,
    euler1d_fv_rhs_with_boundary_fluxes,
    euler1d_fv_ssp_rk3_step,
)


def smooth_state(cells: int, *, device: str = "cpu") -> torch.Tensor:
    state, _ = entropy_average(cells, 0.0)
    return torch.from_numpy(state).to(device=device)


def primitive_state(
    density: torch.Tensor,
    velocity: torch.Tensor,
    pressure: torch.Tensor,
) -> torch.Tensor:
    energy = pressure / (EULER_GAMMA - 1.0) + 0.5 * density * velocity.square()
    return torch.stack((density, density * velocity, energy))


def test_formulation_identity_is_frozen() -> None:
    assert EULER1D_FV_FORMULATION_ID == (
        "fv_dimensional_characteristic_js5_global_matrix_lf_euler1d_v1"
    )


@pytest.mark.parametrize("boundary", ("periodic", "transmissive"))
def test_uniform_state_and_boundary_flux_conservation(boundary: str) -> None:
    density = torch.full((19,), 1.2, dtype=torch.float64)
    state = primitive_state(
        density,
        torch.full_like(density, 0.3),
        torch.full_like(density, 0.9),
    )
    rhs, fluxes = euler1d_fv_rhs_with_boundary_fluxes(
        state, 0.05, boundary=boundary
    )
    assert float(torch.max(torch.abs(rhs))) <= 2.0e-12
    residual = torch.abs(0.05 * torch.sum(rhs, dim=-1) + fluxes[:, 1] - fluxes[:, 0])
    assert float(torch.max(residual)) <= 2.0e-14


def test_entropy_wave_spatial_convergence() -> None:
    sizes = (24, 36, 54, 81)
    errors = []
    for cells in sizes:
        state, exact = entropy_average(cells, 0.0)
        actual = euler1d_fv_rhs(torch.from_numpy(state), 1.0 / cells)
        difference = actual - torch.from_numpy(exact)
        errors.append(float(torch.sqrt(torch.mean(difference.square()))))
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    rates = [
        math.log(coarse / fine) / math.log(fine_n / coarse_n)
        for coarse, fine, coarse_n, fine_n in zip(
            errors, errors[1:], sizes, sizes[1:]
        )
    ]
    assert max(rates) >= 4.0


def test_cfl_and_step_preserve_state_contract() -> None:
    state = smooth_state(23)
    dt = euler1d_fv_cfl_timestep(state, 1.0 / 23.0)
    assert dt.ndim == 0
    assert dt.device == state.device
    assert float(dt) > 0.0
    advanced = euler1d_fv_ssp_rk3_step(state, 1.0 / 23.0, 1.0e-4)
    assert advanced.shape == state.shape
    assert advanced.dtype == state.dtype
    assert advanced.device == state.device
    assert bool(torch.isfinite(advanced).all())


def test_directional_derivative_matches_centered_difference() -> None:
    cells = 19
    base = smooth_state(cells)
    direction = torch.sin(
        torch.arange(base.numel(), dtype=base.dtype).reshape_as(base) + 0.2
    )
    direction = direction / torch.linalg.vector_norm(direction)
    weights = torch.linspace(0.3, 1.7, cells, dtype=base.dtype)

    def objective(values: torch.Tensor) -> torch.Tensor:
        advanced = euler1d_fv_ssp_rk3_step(
            values, 1.0 / cells, 2.0e-4, boundary="transmissive"
        )
        return torch.sum(advanced * weights)

    state = base.detach().requires_grad_(True)
    gradient = torch.autograd.grad(objective(state), state)[0]
    actual = torch.sum(gradient * direction)
    epsilon = 1.0e-6
    expected = (
        objective(base + epsilon * direction)
        - objective(base - epsilon * direction)
    ) / (2.0 * epsilon)
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-7)


@pytest.mark.parametrize("boundary", ("periodic", "transmissive"))
def test_compile_fullgraph(boundary: str) -> None:
    state = smooth_state(21)

    def rhs(values: torch.Tensor) -> torch.Tensor:
        return euler1d_fv_rhs(values, 1.0 / 21.0, boundary=boundary)

    expected = rhs(state)
    torch._dynamo.reset()
    compiled = torch.compile(rhs, fullgraph=True, dynamic=False)
    actual = compiled(state)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=5.0e-11)
    explanation = torch._dynamo.explain(rhs)(state)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("boundary", ("periodic", "transmissive"))
def test_cpu_cuda_agreement(boundary: str) -> None:
    state = smooth_state(37)
    expected = euler1d_fv_rhs(state, 1.0 / 37.0, boundary=boundary)
    actual = euler1d_fv_rhs(
        state.cuda(), 1.0 / 37.0, boundary=boundary
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=5.0e-11)


def test_validation_refuses_unsupported_contracts() -> None:
    state = smooth_state(19)
    with pytest.raises(ValueError, match="boundary"):
        euler1d_fv_rhs(state, 1.0 / 19.0, boundary="wall")
    with pytest.raises(ValueError, match="shape"):
        euler1d_fv_rhs(state[:2], 1.0 / 19.0)
    with pytest.raises(ValueError, match="positive"):
        euler1d_fv_rhs(state, 0.0)


def test_numerical_loop_has_no_forbidden_transfer_or_custom_code() -> None:
    source = inspect.getsource(implementation)
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
    assert "numpy" not in source
