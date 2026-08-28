"""Periodic scalar Burgers operators in ordinary PyTorch.

Finite difference stores point values and reconstructs globally split physical
flux. Finite volume stores physical cell averages, reconstructs face states,
and applies a two-state global Lax--Friedrichs/Rusanov flux. The distinct state
semantics are intentional and no point/average conversion occurs here.
"""

from __future__ import annotations

import math
from numbers import Real

import torch

from .fv_weno5 import fv_weno5_rhs
from .weno_js import WENOJS


BURGERS_FD_WENO5_FORMULATION_ID = (
    "fd_classical_js5_burgers_global_lf_periodic_v1"
)
BURGERS_FV_WENO5_FORMULATION_ID = (
    "fv_dimensional_js5_burgers_global_lf_periodic_v1"
)
_FD_WENO5 = WENOJS(5)


def _validate_state(state: torch.Tensor) -> None:
    if not isinstance(state, torch.Tensor):
        raise TypeError("Burgers state must already be a torch.Tensor")
    if state.dtype not in (torch.float32, torch.float64):
        raise TypeError("Burgers operators require float32 or float64 state")
    if state.ndim < 1:
        raise ValueError("Burgers state must have a spatial dimension")


def _validate_positive_scalar(
    value: float | torch.Tensor,
    *,
    name: str,
    reference: torch.Tensor,
) -> float | torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(f"tensor {name} must be scalar")
        if value.device != reference.device:
            raise ValueError(f"tensor {name} must remain on the state device")
        if value.dtype != reference.dtype:
            raise TypeError(f"tensor {name} must have the state dtype")
        return value
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or scalar tensor")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def burgers_flux(state: torch.Tensor) -> torch.Tensor:
    """Return the inviscid Burgers physical flux ``f(u)=u**2/2``."""
    _validate_state(state)
    return 0.5 * state.square()


def burgers_fd_weno5_rhs(
    point_values: torch.Tensor,
    dx: float | torch.Tensor,
    alpha: float | torch.Tensor,
    *,
    axis: int = -1,
) -> torch.Tensor:
    """Classical periodic FD-WENO-JS5 RHS for Burgers point values."""
    _validate_state(point_values)
    spacing = _validate_positive_scalar(
        dx, name="dx", reference=point_values
    )
    speed = _validate_positive_scalar(
        alpha, name="alpha", reference=point_values
    )
    return _FD_WENO5.rhs(
        point_values,
        spacing,
        burgers_flux,
        alpha=speed,
        axis=axis,
    )


def burgers_fv_weno5_rhs(
    cell_averages: torch.Tensor,
    dx: float | torch.Tensor,
    alpha: float | torch.Tensor,
    *,
    axis: int = -1,
) -> torch.Tensor:
    """Periodic FV-WENO-JS5 RHS for physical Burgers cell averages."""
    _validate_state(cell_averages)
    spacing = _validate_positive_scalar(
        dx, name="dx", reference=cell_averages
    )
    speed = _validate_positive_scalar(
        alpha, name="alpha", reference=cell_averages
    )
    return fv_weno5_rhs(
        cell_averages,
        spacing,
        burgers_flux,
        speed,
        axis=axis,
    )
