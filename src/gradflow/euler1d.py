"""Bounded one-dimensional characteristic finite-difference WENO-JS Euler.

Public states contain physical point samples only. Ghost construction is an
internal boundary operation, and the numerical path reuses the generated
characteristic algebra qualified by the existing periodic Euler system.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor

from .euler3d import (
    EULER_GAMMA,
    QUALIFIED_EULER_WENO_ORDERS,
    _euler_weno_scheme,
    _generated_bounded_line_rhs,
)
from .weno_js import WENOJS, WENOJSPrecisionPolicy

EULER1D_BOUNDARIES = ("periodic", "transmissive")


def _validate_spacing(dx: float) -> float:
    if isinstance(dx, bool) or not isinstance(dx, Real):
        raise TypeError("dx must be a positive finite real scalar")
    value = float(dx)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("dx must be a positive finite real scalar")
    return value


def _validate_state(state: Tensor, order: int | None = None) -> None:
    if not isinstance(state, Tensor):
        raise TypeError("state must already be a torch.Tensor")
    if state.dtype not in (torch.float32, torch.float64):
        raise TypeError("one-dimensional Euler requires float32 or float64")
    if state.ndim != 2 or state.shape[0] != 3:
        raise ValueError("one-dimensional Euler state shape must be (3, points)")
    if state.layout is not torch.strided:
        raise ValueError("one-dimensional Euler state must use strided layout")
    if order is not None and state.shape[1] < order:
        raise ValueError(
            f"Euler WENO-JS order {order} requires at least {order} points"
        )


def _normalize_order(order: int) -> int:
    if order not in QUALIFIED_EULER_WENO_ORDERS:
        raise ValueError(
            f"Euler WENO-JS order must be one of {QUALIFIED_EULER_WENO_ORDERS}"
        )
    return order


def _normalize_boundary(boundary: str) -> str:
    if not isinstance(boundary, str):
        raise TypeError("boundary must be a string")
    normalized = boundary.lower().replace("-", "_")
    if normalized not in EULER1D_BOUNDARIES:
        raise ValueError(f"boundary must be one of {EULER1D_BOUNDARIES}")
    return normalized


def _ghost_state(state: Tensor, width: int, boundary: str) -> Tensor:
    if boundary == "periodic":
        left = state[..., -width:]
        right = state[..., :width]
    else:
        left = state[..., :1].expand(*state.shape[:-1], width)
        right = state[..., -1:].expand(*state.shape[:-1], width)
    return torch.cat((left, state, right), dim=-1)


def euler1d_rhs_with_boundary_fluxes(
    state: Tensor,
    dx: float,
    *,
    order: int = 5,
    boundary: str = "periodic",
    precision: WENOJSPrecisionPolicy | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the physical RHS and left/right numerical boundary fluxes.

    ``state`` has shape ``(3, points)`` in conservative order
    ``(rho, rho*u, E)``. The boundary-flux result has shape ``(3, 2)`` with
    left then right domain flux.
    """
    normalized_order = _normalize_order(order)
    _validate_state(state, normalized_order)
    spacing = _validate_spacing(dx)
    normalized_boundary = _normalize_boundary(boundary)
    scheme = _euler_weno_scheme(normalized_order, precision)
    return _euler1d_rhs_with_scheme(
        state, spacing, normalized_boundary, scheme
    )


def _euler1d_rhs_with_scheme(
    state: Tensor,
    spacing: float,
    boundary: str,
    scheme: WENOJS,
) -> tuple[Tensor, Tensor]:
    """Execute bounded Euler with an already-resolved immutable scheme."""
    ghosted = _ghost_state(
        state, scheme.substencil_width, boundary
    )
    rhs, face_fluxes = _generated_bounded_line_rhs(
        ghosted, 1.0 / spacing, scheme, state.shape[-1]
    )
    boundary_fluxes = torch.stack(
        (face_fluxes[..., 0], face_fluxes[..., -1]), dim=-1
    )
    return rhs, boundary_fluxes


def euler1d_rhs(
    state: Tensor,
    dx: float,
    *,
    order: int = 5,
    boundary: str = "periodic",
    precision: WENOJSPrecisionPolicy | None = None,
) -> Tensor:
    """Return the one-dimensional characteristic WENO-JS Euler RHS."""
    rhs, _ = euler1d_rhs_with_boundary_fluxes(
        state, dx, order=order, boundary=boundary, precision=precision
    )
    return rhs


def euler1d_cfl_timestep(state: Tensor, dx: float, cfl: float = 0.1) -> Tensor:
    """Return the one-dimensional Euler CFL timestep on the state device."""
    _validate_state(state)
    spacing = _validate_spacing(dx)
    if isinstance(cfl, bool) or not isinstance(cfl, Real):
        raise TypeError("cfl must be a positive finite real scalar")
    cfl_value = float(cfl)
    if not math.isfinite(cfl_value) or cfl_value <= 0.0:
        raise ValueError("cfl must be a positive finite real scalar")
    density = state[0]
    velocity = state[1] / density
    pressure = (EULER_GAMMA - 1.0) * (
        state[2] - 0.5 * density * velocity.square()
    )
    sound_speed = torch.sqrt(EULER_GAMMA * pressure / density)
    return cfl_value * spacing / torch.amax(torch.abs(velocity) + sound_speed)


def euler1d_ssp_rk3_step(
    state: Tensor,
    dx: float,
    dt: float | Tensor,
    *,
    order: int = 5,
    boundary: str = "periodic",
    precision: WENOJSPrecisionPolicy | None = None,
) -> Tensor:
    """Advance one full three-stage SSP-RK3 step."""
    rhs0 = euler1d_rhs(
        state, dx, order=order, boundary=boundary, precision=precision
    )
    stage1 = state + dt * rhs0
    rhs1 = euler1d_rhs(
        stage1, dx, order=order, boundary=boundary, precision=precision
    )
    stage2 = 0.75 * state + 0.25 * (stage1 + dt * rhs1)
    rhs2 = euler1d_rhs(
        stage2, dx, order=order, boundary=boundary, precision=precision
    )
    return (state + 2.0 * (stage2 + dt * rhs2)) / 3.0
