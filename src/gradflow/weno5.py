"""Direct PyTorch seed for scalar finite-difference WENO-5.

This module follows Sigal Gottlieb's correction form of Jiang--Shu WENO-5:
a fourth-order central interface flux plus nonlinear corrections applied to
the positive and negative Lax--Friedrichs split flux differences.

The numerical path is ordinary tensor indexing, periodic shifts, and
elementwise arithmetic. It contains no convolution, custom operator,
handwritten CUDA, or handwritten Triton.
"""

from collections.abc import Callable
from typing import TypeAlias

import torch

TensorFunction: TypeAlias = Callable[[torch.Tensor], torch.Tensor]

DEFAULT_EPSILON = 1.0e-29


def _validate_state(u: torch.Tensor, minimum_points: int) -> None:
    """Enforce the deliberately narrow seed contract without moving data."""
    if not isinstance(u, torch.Tensor):
        raise TypeError("u must already be a torch.Tensor")
    if u.dtype != torch.float64:
        raise TypeError("the WENO-5 seed requires torch.float64 input")
    if u.ndim < 1:
        raise ValueError("u must have a spatial dimension")
    if u.shape[-1] < minimum_points:
        raise ValueError(f"WENO-5 requires at least {minimum_points} points")


def _global_lax_friedrichs_speed(
    u: torch.Tensor,
    flux_derivative: TensorFunction | None,
    alpha: float | torch.Tensor | None,
) -> float | torch.Tensor:
    """Return alpha without scalar extraction or an implicit device copy."""
    if alpha is not None:
        return alpha
    if flux_derivative is None:
        raise ValueError("provide flux_derivative when alpha is not explicit")
    return torch.amax(torch.abs(flux_derivative(u)))


def _shift(v: torch.Tensor, offset: int) -> torch.Tensor:
    """Return ``v[..., j + offset]`` with periodic wrap."""
    return torch.roll(v, shifts=-offset, dims=-1)


def _weno_correction(
    h1: torch.Tensor,
    h2: torch.Tensor,
    h3: torch.Tensor,
    h4: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Evaluate one directional correction in Gottlieb's flux form."""
    t1 = h1 - h2
    t2 = h2 - h3
    t3 = h3 - h4

    beta1 = 13.0 * t1 * t1 + 3.0 * (h1 - 3.0 * h2) ** 2
    beta2 = 13.0 * t2 * t2 + 3.0 * (h2 + h3) ** 2
    beta3 = 13.0 * t3 * t3 + 3.0 * (3.0 * h3 - h4) ** 2

    q1 = (epsilon + beta1) ** 2
    q2 = (epsilon + beta2) ** 2
    q3 = (epsilon + beta3) ** 2

    s1 = q2 * q3
    s2 = 6.0 * q1 * q3
    s3 = 3.0 * q1 * q2
    inverse_sum = 1.0 / (s1 + s2 + s3)
    weight1 = s1 * inverse_sum
    weight3 = s3 * inverse_sum

    return (
        weight1 * (t2 - t1) + (0.5 * weight3 - 0.25) * (t3 - t2)
    ) / 3.0


def weno5_rhs(
    u: torch.Tensor,
    dx: float | torch.Tensor,
    flux: TensorFunction,
    flux_derivative: TensorFunction | None = None,
    *,
    alpha: float | torch.Tensor | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Compute the scalar WENO-5 semidiscrete RHS on unique periodic nodes.

    The last dimension is space; any leading dimensions are batches. The grid
    represents ``x_j = x_0 + j*dx`` for ``j = 0, ..., n-1`` and does not repeat
    the periodic endpoint. The returned tensor has the same shape, dtype, and
    device as ``u``.

    ``alpha`` is the global Lax--Friedrichs speed. If it is omitted, it is
    computed on-device as ``max(abs(flux_derivative(u)))``. GradFlow never
    calls ``.item()``, ``.cpu()``, ``.cuda()``, or ``.to()`` in this numerical
    path. An explicit tensor ``alpha`` must therefore already share the input
    device. The current validated seed policy is IEEE float64 only.
    """
    _validate_state(u, minimum_points=5)
    alpha_value = _global_lax_friedrichs_speed(u, flux_derivative, alpha)

    physical_flux = flux(u)
    if physical_flux.shape != u.shape:
        raise ValueError("flux(u) must have the same shape as u")

    delta_u = _shift(u, 1) - u
    delta_f = _shift(physical_flux, 1) - physical_flux
    delta_plus = 0.5 * (delta_f + alpha_value * delta_u)
    delta_minus = 0.5 * (delta_f - alpha_value * delta_u)

    central = (
        -_shift(physical_flux, -1)
        + 7.0 * (physical_flux + _shift(physical_flux, 1))
        - _shift(physical_flux, 2)
    ) / 12.0

    positive = _weno_correction(
        _shift(delta_plus, -2),
        _shift(delta_plus, -1),
        delta_plus,
        _shift(delta_plus, 1),
        epsilon,
    )
    negative = _weno_correction(
        -_shift(delta_minus, 2),
        -_shift(delta_minus, 1),
        -delta_minus,
        -_shift(delta_minus, -1),
        epsilon,
    )

    interface_flux = central + positive + negative
    return (_shift(interface_flux, -1) - interface_flux) / dx


def weno5_rhs_gottlieb_periodic(
    u: torch.Tensor,
    dx: float | torch.Tensor,
    flux: TensorFunction,
    flux_derivative: TensorFunction | None = None,
    *,
    alpha: float | torch.Tensor | None = None,
    epsilon: float = DEFAULT_EPSILON,
) -> torch.Tensor:
    """Compute the RHS using Gottlieb's duplicated-endpoint convention.

    This adapter exists to reproduce the committed MATLAB oracle exactly. Its
    input contains both endpoints of the periodic interval. The endpoints may
    carry distinct traces at a discontinuity, as in the preserved ``sign(x)``
    example. New periodic calculations should normally use :func:`weno5_rhs`
    on unique nodes.
    """
    _validate_state(u, minimum_points=6)
    alpha_value = _global_lax_friedrichs_speed(u, flux_derivative, alpha)

    # MATLAB: [u(i-md:end-1), u, u(2:md+2)] with md=4.
    extended = torch.cat((u[..., -5:-1], u, u[..., 1:6]), dim=-1)
    physical_flux = flux(extended)
    if physical_flux.shape != extended.shape:
        raise ValueError("flux(u) must have the same shape as u")

    delta_u = extended[..., 1:] - extended[..., :-1]
    delta_f = physical_flux[..., 1:] - physical_flux[..., :-1]
    delta_plus = 0.5 * (delta_f + alpha_value * delta_u)
    delta_minus = 0.5 * (delta_f - alpha_value * delta_u)

    central = (
        -physical_flux[..., 2:-5]
        + 7.0 * (physical_flux[..., 3:-4] + physical_flux[..., 4:-3])
        - physical_flux[..., 5:-2]
    ) / 12.0
    positive = _weno_correction(
        delta_plus[..., 1:-5],
        delta_plus[..., 2:-4],
        delta_plus[..., 3:-3],
        delta_plus[..., 4:-2],
        epsilon,
    )
    negative = _weno_correction(
        -delta_minus[..., 5:-1],
        -delta_minus[..., 4:-2],
        -delta_minus[..., 3:-3],
        -delta_minus[..., 2:-4],
        epsilon,
    )

    interface_flux = central + positive + negative
    return (interface_flux[..., :-2] - interface_flux[..., 1:-1]) / dx


def ssp_rk3_step(
    u: torch.Tensor,
    dt: float | torch.Tensor,
    rhs: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Advance one Shu--Osher SSP-RK3 step."""
    stage1 = u + dt * rhs(u)
    stage2 = 0.75 * u + 0.25 * (stage1 + dt * rhs(stage1))
    return (u + 2.0 * (stage2 + dt * rhs(stage2))) / 3.0
