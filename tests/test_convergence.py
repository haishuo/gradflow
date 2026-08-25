import math

import pytest
import torch

from gradflow import weno5_rhs


@pytest.mark.parametrize("speed", [1.0, -1.0])
def test_fifth_order_spatial_convergence(speed: float) -> None:
    errors = []
    for n in (40, 80, 160, 320):
        x = torch.arange(n, dtype=torch.float64) / n
        u = torch.sin(2.0 * math.pi * x)
        actual = weno5_rhs(u, 1.0 / n, lambda q: speed * q, alpha=abs(speed))
        exact = -speed * 2.0 * math.pi * torch.cos(2.0 * math.pi * x)
        errors.append(torch.sqrt(torch.mean((actual - exact) ** 2)).item())

    rates = [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:])]
    assert min(rates) > 4.8, f"errors={errors}, rates={rates}"


def test_periodic_conservation_to_roundoff() -> None:
    generator = torch.Generator().manual_seed(20260825)
    u = torch.randn(3, 257, generator=generator, dtype=torch.float64)
    rhs = weno5_rhs(u, 1.0 / 257, lambda q: 0.5 * q * q, lambda q: q)

    residual = torch.abs(torch.sum(rhs, dim=-1))
    roundoff_bound = (
        8.0
        * torch.finfo(torch.float64).eps
        * torch.sum(torch.abs(rhs), dim=-1)
    )
    assert torch.all(residual <= roundoff_bound)


@pytest.mark.parametrize("flux_name", ["negative_linear", "burgers"])
def test_unique_grid_matches_gottlieb_adapter_with_active_negative_split(
    flux_name: str,
) -> None:
    """Cover the split-family sign outside right-moving linear advection."""
    from gradflow import weno5_rhs_gottlieb_periodic

    n = 256
    x = torch.arange(n, dtype=torch.float64) / n
    u = 0.6 + torch.sin(2.0 * math.pi * x) + 0.1 * torch.cos(6.0 * math.pi * x)
    duplicated = torch.cat((u, u[:1]))

    if flux_name == "negative_linear":
        flux = lambda q: -q
        flux_derivative = lambda q: -torch.ones_like(q)
        alpha = 1.0
    else:
        flux = lambda q: 0.5 * q * q
        flux_derivative = lambda q: q
        alpha = torch.max(torch.abs(duplicated))

    unique_rhs = weno5_rhs(
        u, 1.0 / n, flux, flux_derivative, alpha=alpha
    )
    duplicated_rhs = weno5_rhs_gottlieb_periodic(
        duplicated, 1.0 / n, flux, flux_derivative, alpha=alpha
    )

    torch.testing.assert_close(
        unique_rhs, duplicated_rhs[:-1], rtol=0.0, atol=2.0e-13
    )


def test_autograd_smoke() -> None:
    u = torch.linspace(-0.8, 0.9, 64, dtype=torch.float64, requires_grad=True)
    rhs = weno5_rhs(u, 1.0 / 64, lambda q: 0.5 * q * q, lambda q: q)
    torch.mean(rhs * rhs).backward()

    assert u.grad is not None
    assert u.grad.dtype == torch.float64
    assert torch.all(torch.isfinite(u.grad))
