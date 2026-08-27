"""Scalar periodic finite-volume WENO-JS5 in ordinary PyTorch.

The persistent input is a physical cell average on a uniform periodic mesh.
This module reconstructs left/right point states at each face, evaluates a
global Lax--Friedrichs/Rusanov numerical flux, and differences that face flux
conservatively. It does not perform finite-difference physical-flux splitting.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from numbers import Real
from typing import TypeAlias

import torch

from .weno_js import WENOJS


TensorFunction: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
FV_WENO5_FORMULATION_ID = "fv_dimensional_js5_global_lf_periodic_v1"
_RECONSTRUCTION = WENOJS(5)


def _validate_state(cell_averages: torch.Tensor, axis: int) -> int:
    if not isinstance(cell_averages, torch.Tensor):
        raise TypeError("cell_averages must already be a torch.Tensor")
    if cell_averages.dtype not in (torch.float32, torch.float64):
        raise TypeError("finite-volume WENO-JS5 requires float32 or float64 input")
    if cell_averages.ndim < 1:
        raise ValueError("cell_averages must have a spatial dimension")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("axis must be an integer")
    if not -cell_averages.ndim <= axis < cell_averages.ndim:
        raise ValueError("axis is outside the input rank")
    normalized = axis % cell_averages.ndim
    if cell_averages.shape[normalized] < 5:
        raise ValueError("finite-volume WENO-JS5 requires at least five cells")
    return normalized


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
        # Positivity is a caller precondition. Inspecting a CUDA scalar here
        # would synchronize every numerical call; see Phase-3 amendment 1.
        return value
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or scalar tensor")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _validate_flux_result(
    result: torch.Tensor,
    reference: torch.Tensor,
    *,
    side: str,
) -> None:
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"flux({side}) must return a torch.Tensor")
    if result.shape != reference.shape:
        raise ValueError(f"flux({side}) must preserve shape")
    if result.device != reference.device:
        raise ValueError(f"flux({side}) must remain on the state device")
    if result.dtype != reference.dtype:
        raise TypeError(f"flux({side}) must preserve the state dtype")


def fv_weno5_face_states(
    cell_averages: torch.Tensor,
    *,
    axis: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct periodic left/right states at each cell's right face.

    Index ``i`` of each returned tensor is face ``i+1/2``, the right face of
    physical cell-average index ``i``. The coefficient algebra is shared with
    GradFlow's exact-generated order-five reconstruction, whose derivation from
    cell-average moments was independently checked in FD/FV Phase 2.
    """
    normalized = _validate_state(cell_averages, axis)
    left = _fv_weno5_reconstruct(cell_averages, bias="left", axis=normalized)
    right = _fv_weno5_reconstruct(cell_averages, bias="right", axis=normalized)
    return left, right


def _fv_weno5_reconstruct(
    cell_averages: torch.Tensor,
    *,
    bias: str,
    axis: int,
) -> torch.Tensor:
    """Apply the shared exact reconstruction under explicit FV semantics."""
    normalized = _validate_state(cell_averages, axis)
    if bias not in {"left", "right"}:
        raise ValueError("bias must be 'left' or 'right'")
    return _RECONSTRUCTION.reconstruct(
        cell_averages,
        bias=bias,
        axis=normalized,
    )


def fv_global_lax_friedrichs_flux(
    left: torch.Tensor,
    right: torch.Tensor,
    flux: TensorFunction,
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    """Evaluate the scalar global LF/Rusanov numerical flux at every face."""
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError("left and right states must be torch.Tensor values")
    if left.shape != right.shape:
        raise ValueError("left and right states must have the same shape")
    if left.device != right.device:
        raise ValueError("left and right states must share one device")
    if left.dtype != right.dtype:
        raise TypeError("left and right states must share one dtype")
    if left.dtype not in (torch.float32, torch.float64):
        raise TypeError("finite-volume flux requires float32 or float64 states")
    alpha_value = _validate_positive_scalar(
        alpha,
        name="alpha",
        reference=left,
    )
    left_flux = flux(left)
    right_flux = flux(right)
    _validate_flux_result(left_flux, left, side="left")
    _validate_flux_result(right_flux, right, side="right")
    return 0.5 * (
        left_flux + right_flux - alpha_value * (right - left)
    )


def fv_weno5_rhs(
    cell_averages: torch.Tensor,
    dx: float | torch.Tensor,
    flux: TensorFunction,
    alpha: float | torch.Tensor,
    *,
    axis: int = -1,
) -> torch.Tensor:
    """Return the conservative periodic FV-WENO-JS5 semidiscrete RHS."""
    normalized = _validate_state(cell_averages, axis)
    spacing = _validate_positive_scalar(
        dx,
        name="dx",
        reference=cell_averages,
    )
    left, right = fv_weno5_face_states(cell_averages, axis=normalized)
    face_flux = fv_global_lax_friedrichs_flux(left, right, flux, alpha)
    previous_face_flux = torch.roll(face_flux, shifts=1, dims=normalized)
    return (previous_face_flux - face_flux) / spacing
