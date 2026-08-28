"""Matched one-dimensional characteristic finite-volume WENO-JS5 Euler.

Public states are physical conservative cell averages. Face reconstruction
uses the exact-generated WENO-JS5 object and the same face-frozen Roe matrices
and line-global characteristic LF speeds as the qualified classical FD path.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .euler1d import (
    _ghost_state,
    _normalize_boundary,
    _validate_spacing,
    _validate_state,
    euler1d_cfl_timestep,
)
from .euler3d import EULER_GAMMA, _euler_weno_scheme, _flux_and_roe_faces


EULER1D_FV_FORMULATION_ID = (
    "fv_dimensional_characteristic_js5_global_matrix_lf_euler1d_v1"
)
_SCHEME = _euler_weno_scheme(5, None)


def _physical_flux(state: Tensor) -> Tensor:
    density = state[0]
    momentum = state[1]
    energy = state[2]
    velocity = momentum / density
    pressure = (EULER_GAMMA - 1.0) * (
        energy - 0.5 * momentum * velocity
    )
    return torch.stack(
        (momentum, momentum * velocity + pressure, velocity * (energy + pressure))
    )


def _characteristic_face_state(
    ghosted: Tensor,
    roe_left: Tensor,
    roe_right: Tensor,
    *,
    face_start: int,
    face_stop: int,
    bias: str,
) -> Tensor:
    offsets = _SCHEME.exact_coefficients.candidate_offsets
    candidate_stencils = []
    for candidate in offsets:
        stencil = []
        for offset in candidate:
            sample_offset = offset if bias == "left" else 1 - offset
            sample = ghosted[:, face_start + sample_offset : face_stop + sample_offset]
            sample_by_field = sample.movedim(-2, -1).unsqueeze(-2)
            stencil.append((roe_left * sample_by_field).sum(dim=-1))
        candidate_stencils.append(stencil)
    characteristic = _SCHEME.reconstruct_stencils(candidate_stencils)
    conservative = (roe_right * characteristic.unsqueeze(-2)).sum(dim=-1)
    return conservative.movedim(-1, -2)


def euler1d_fv_rhs_with_boundary_fluxes(
    cell_averages: Tensor,
    dx: float,
    *,
    boundary: str = "periodic",
) -> tuple[Tensor, Tensor]:
    """Return the matched FV Euler RHS and physical boundary fluxes.

    ``cell_averages`` has shape ``(3,cells)`` in conservative order. The
    boundary-flux result has shape ``(3,2)`` with left then right domain flux.
    """
    _validate_state(cell_averages, 5)
    spacing = _validate_spacing(dx)
    normalized_boundary = _normalize_boundary(boundary)
    width = _SCHEME.substencil_width
    ghosted = _ghost_state(cell_averages, width, normalized_boundary)
    _, alpha, all_left, all_right = _flux_and_roe_faces(ghosted)
    face_start = width - 1
    face_stop = face_start + cell_averages.shape[-1] + 1
    roe_left = all_left[face_start:face_stop]
    roe_right = all_right[face_start:face_stop]
    left_state = _characteristic_face_state(
        ghosted,
        roe_left,
        roe_right,
        face_start=face_start,
        face_stop=face_stop,
        bias="left",
    )
    right_state = _characteristic_face_state(
        ghosted,
        roe_left,
        roe_right,
        face_start=face_start,
        face_stop=face_stop,
        bias="right",
    )
    jump_by_field = (right_state - left_state).movedim(-2, -1)
    characteristic_jump = (
        roe_left * jump_by_field.unsqueeze(-2)
    ).sum(dim=-1)
    dissipation = (
        roe_right * (alpha * characteristic_jump).unsqueeze(-2)
    ).sum(dim=-1)
    face_flux = 0.5 * (
        _physical_flux(left_state)
        + _physical_flux(right_state)
        - dissipation.movedim(-1, -2)
    )
    rhs = (face_flux[:, :-1] - face_flux[:, 1:]) / spacing
    boundary_fluxes = torch.stack((face_flux[:, 0], face_flux[:, -1]), dim=-1)
    return rhs, boundary_fluxes


def euler1d_fv_rhs(
    cell_averages: Tensor,
    dx: float,
    *,
    boundary: str = "periodic",
) -> Tensor:
    """Return the matched characteristic FV-WENO-JS5 Euler RHS."""
    rhs, _ = euler1d_fv_rhs_with_boundary_fluxes(
        cell_averages, dx, boundary=boundary
    )
    return rhs


def euler1d_fv_cfl_timestep(
    cell_averages: Tensor,
    dx: float,
    cfl: float = 0.1,
) -> Tensor:
    """Return the existing on-device Euler CFL estimate for FV averages."""
    return euler1d_cfl_timestep(cell_averages, dx, cfl)


def euler1d_fv_ssp_rk3_step(
    cell_averages: Tensor,
    dx: float,
    dt: float | Tensor,
    *,
    boundary: str = "periodic",
) -> Tensor:
    """Advance the matched FV Euler state through one SSP-RK3 step."""
    rhs0 = euler1d_fv_rhs(cell_averages, dx, boundary=boundary)
    stage1 = cell_averages + dt * rhs0
    rhs1 = euler1d_fv_rhs(stage1, dx, boundary=boundary)
    stage2 = 0.75 * cell_averages + 0.25 * (stage1 + dt * rhs1)
    rhs2 = euler1d_fv_rhs(stage2, dx, boundary=boundary)
    return (cell_averages + 2.0 * (stage2 + dt * rhs2)) / 3.0
