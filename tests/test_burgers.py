from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from gradflow import (
    BURGERS_FD_WENO5_FORMULATION_ID,
    BURGERS_FV_WENO5_FORMULATION_ID,
    burgers_fd_weno5_rhs,
    burgers_flux,
    burgers_fv_weno5_rhs,
    ssp_rk3_step,
)
from experiments.fd_fv_nonlinear.burgers_oracle import (
    LF_ALPHA,
    exact_point,
    exact_spatial_derivative,
    projected_state,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formulation_identities_are_frozen() -> None:
    assert BURGERS_FD_WENO5_FORMULATION_ID == (
        "fd_classical_js5_burgers_global_lf_periodic_v1"
    )
    assert BURGERS_FV_WENO5_FORMULATION_ID == (
        "fv_dimensional_js5_burgers_global_lf_periodic_v1"
    )


def test_burgers_flux_preserves_shape_dtype_device_and_gradient() -> None:
    state = torch.tensor([-0.5, 0.0, 0.75], dtype=torch.float64, requires_grad=True)
    flux = burgers_flux(state)
    torch.testing.assert_close(flux, 0.5 * state.square())
    assert flux.shape == state.shape
    assert flux.dtype == state.dtype
    assert flux.device == state.device
    flux.sum().backward()
    torch.testing.assert_close(state.grad, state.detach())


@pytest.mark.parametrize(
    "rhs", (burgers_fd_weno5_rhs, burgers_fv_weno5_rhs)
)
def test_constant_state_is_stationary_and_conservative(rhs) -> None:
    state = torch.full((3, 37), 0.4, dtype=torch.float64)
    actual = rhs(state, 1.0 / 37.0, LF_ALPHA)
    assert torch.max(torch.abs(actual)).item() <= 5.0e-13
    step = ssp_rk3_step(
        state,
        1.0e-3,
        lambda values: rhs(values, 1.0 / 37.0, LF_ALPHA),
    )
    torch.testing.assert_close(step, state, rtol=0.0, atol=5.0e-13)

    generator = torch.Generator().manual_seed(20260828)
    nonconstant = 0.5 + 0.1 * torch.randn(
        (3, 37), generator=generator, dtype=torch.float64
    )
    residual = torch.abs(torch.sum(rhs(nonconstant, 1.0 / 37.0, LF_ALPHA), dim=-1))
    bound = 64.0 * torch.finfo(torch.float64).eps * torch.sum(
        torch.abs(rhs(nonconstant, 1.0 / 37.0, LF_ALPHA)), dim=-1
    )
    assert torch.all(residual <= bound)


def test_fd_and_fv_use_their_own_exact_state_semantics() -> None:
    cells = 37
    dx = 1.0 / cells
    fd = torch.tensor(projected_state("fd", cells, 0.0), dtype=torch.float64)
    fv = torch.tensor(projected_state("fv", cells, 0.0), dtype=torch.float64)
    assert torch.max(torch.abs(fd - fv)).item() > 1.0e-4

    fd_exact = torch.tensor(
        [
            -exact_point(index * dx, 0.0)
            * exact_spatial_derivative(index * dx, 0.0)
            for index in range(cells)
        ],
        dtype=torch.float64,
    )
    faces = torch.arange(cells + 1, dtype=torch.float64) * dx
    face_values = torch.tensor(
        [exact_point(float(x), 0.0) for x in faces], dtype=torch.float64
    )
    fv_exact = -(
        burgers_flux(face_values[1:]) - burgers_flux(face_values[:-1])
    ) / dx

    fd_error = torch.mean(torch.abs(burgers_fd_weno5_rhs(fd, dx, LF_ALPHA) - fd_exact))
    fv_error = torch.mean(torch.abs(burgers_fv_weno5_rhs(fv, dx, LF_ALPHA) - fv_exact))
    assert torch.isfinite(fd_error)
    assert torch.isfinite(fv_error)
    assert fd_error.item() < 1.0e-3
    assert fv_error.item() < 1.0e-3


@pytest.mark.parametrize(
    "rhs", (burgers_fd_weno5_rhs, burgers_fv_weno5_rhs)
)
def test_refusal_and_axis_contract(rhs) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        rhs(torch.ones(8, dtype=torch.int64), 1.0, 1.0)
    with pytest.raises(ValueError, match="spatial dimension"):
        rhs(torch.tensor(1.0, dtype=torch.float64), 1.0, 1.0)
    with pytest.raises(ValueError, match="at least"):
        rhs(torch.ones(4, dtype=torch.float64), 1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        rhs(torch.ones(8, dtype=torch.float64), 0.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        rhs(torch.ones(8, dtype=torch.float64), 1.0, -1.0)
    with pytest.raises(ValueError, match="scalar"):
        rhs(
            torch.ones(8, dtype=torch.float64),
            1.0,
            torch.ones(2, dtype=torch.float64),
        )

    state = torch.linspace(0.3, 0.7, 22, dtype=torch.float64).reshape(2, 11)
    actual = rhs(state, 1.0 / 11.0, LF_ALPHA, axis=1)
    expected = torch.stack(
        [rhs(row, 1.0 / 11.0, LF_ALPHA) for row in state], dim=0
    )
    torch.testing.assert_close(actual, expected)


def test_burgers_source_has_no_transfer_or_scalar_extraction() -> None:
    source = (ROOT / "src/gradflow/burgers.py").read_text()
    for forbidden in (
        ".cpu(",
        ".cuda(",
        ".to(",
        ".item(",
        ".numpy(",
        "numpy",
        "triton",
    ):
        assert forbidden not in source.lower()


@pytest.mark.parametrize(
    "rhs", (burgers_fd_weno5_rhs, burgers_fv_weno5_rhs)
)
def test_fullgraph_compile_cpu(rhs) -> None:
    cells = 19
    coordinates = torch.arange(cells, dtype=torch.float64) / cells
    state = 0.5 + 0.1 * torch.sin(2.0 * math.pi * coordinates)

    def call(values: torch.Tensor) -> torch.Tensor:
        return rhs(values, 1.0 / cells, LF_ALPHA)

    expected = call(state)
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(call)(state)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0
    torch._dynamo.reset()
    compiled = torch.compile(call, fullgraph=True, dynamic=False)
    torch.testing.assert_close(compiled(state), expected, rtol=0.0, atol=2.0e-11)
