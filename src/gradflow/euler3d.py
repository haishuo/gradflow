"""Generated characteristic finite-difference WENO-JS for compressible Euler.

The order-five path preserves the matched direct-PyTorch formulation used by
the Shu Euler bakeoff. Every qualified order shares the ancestral
duplicated-periodic grid, Roe characteristic reconstruction, line-wise global
LF policy, and SSP-RK3 algebra. The module contains no convolution or custom
native operation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from .weno_js import QUALIFIED_ORDERS, WENOJS

EULER_GAMMA = 1.4
EULER_WENO_EPSILON = 1.0e-6
EULER_LF_ENLARGEMENT = 1.1
QUALIFIED_EULER_WENO_ORDERS = QUALIFIED_ORDERS
_EULER_WENO_SCHEMES = {
    order: WENOJS(order, epsilon=EULER_WENO_EPSILON)
    for order in QUALIFIED_EULER_WENO_ORDERS
}


def _component_order(ndim: int, axis: int) -> tuple[int, ...]:
    momenta = list(range(1, ndim + 1))
    normal = momenta.pop(axis)
    return (0, normal, *momenta, ndim + 1)


def synchronize_duplicate_endpoints(state: Tensor) -> Tensor:
    """Copy each stored final periodic endpoint over its first endpoint."""
    synchronized = state
    ndim = state.ndim - 1
    for axis in range(ndim):
        tensor_axis = state.ndim - 1 - axis
        last = synchronized.narrow(
            tensor_axis, synchronized.shape[tensor_axis] - 1, 1
        )
        remainder = synchronized.narrow(
            tensor_axis, 1, synchronized.shape[tensor_axis] - 1
        )
        synchronized = torch.cat((last, remainder), dim=tensor_axis)
    return synchronized


def _flux_and_roe_faces(line: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return physical flux, line LF speeds, and Roe matrices at every face."""
    equations = line.shape[-2]
    ndim = equations - 2
    gamma_minus_one = EULER_GAMMA - 1.0

    density = line[..., 0, :]
    momenta = line[..., 1 : ndim + 1, :]
    energy = line[..., -1, :]
    velocity = momenta / density.unsqueeze(-2)
    velocity_squared = (velocity * velocity).sum(dim=-2)
    pressure = gamma_minus_one * (energy - 0.5 * density * velocity_squared)
    sound_speed = torch.sqrt(EULER_GAMMA * pressure / density)
    enthalpy = (pressure + energy) / density

    normal_velocity = velocity[..., 0, :]
    flux_parts = [
        momenta[..., 0, :],
        momenta[..., 0, :] * normal_velocity + pressure,
    ]
    for tangent in range(1, ndim):
        flux_parts.append(momenta[..., tangent, :] * normal_velocity)
    flux_parts.append(normal_velocity * (pressure + energy))
    flux = torch.stack(flux_parts, dim=-2)

    speed_minus = torch.amax(torch.abs(normal_velocity - sound_speed), dim=-1)
    speed_center = torch.amax(torch.abs(normal_velocity), dim=-1)
    speed_plus = torch.amax(torch.abs(normal_velocity + sound_speed), dim=-1)
    alpha = EULER_LF_ENLARGEMENT * torch.stack(
        (speed_minus, *([speed_center] * ndim), speed_plus), dim=-1
    )
    alpha = torch.clamp_min(alpha, 1.0e-15)

    right_density = torch.roll(density, shifts=-1, dims=-1)
    right_velocity = torch.roll(velocity, shifts=-1, dims=-1)
    right_enthalpy = torch.roll(enthalpy, shifts=-1, dims=-1)
    left_weight = torch.sqrt(density)
    right_weight = torch.sqrt(right_density)
    fraction = left_weight / (left_weight + right_weight)
    roe_velocity = (
        fraction.unsqueeze(-2) * velocity
        + (1.0 - fraction).unsqueeze(-2) * right_velocity
    )
    roe_enthalpy = fraction * enthalpy + (1.0 - fraction) * right_enthalpy
    roe_q = 0.5 * (roe_velocity * roe_velocity).sum(dim=-2)
    roe_sound = torch.sqrt(gamma_minus_one * (roe_enthalpy - roe_q))

    normal = roe_velocity[..., 0, :]
    tangential = [roe_velocity[..., index, :] for index in range(1, ndim)]
    one = torch.ones_like(normal)
    zero = torch.zeros_like(normal)

    right_columns = [
        torch.stack(
            (
                one,
                normal - roe_sound,
                *tangential,
                roe_enthalpy - normal * roe_sound,
            ),
            dim=-1,
        )
    ]
    for tangent_index, tangent_velocity in enumerate(tangential):
        momentum_entries = [zero] * ndim
        momentum_entries[tangent_index + 1] = one
        right_columns.append(
            torch.stack((zero, *momentum_entries, tangent_velocity), dim=-1)
        )
    right_columns.append(torch.stack((one, normal, *tangential, roe_q), dim=-1))
    right_columns.append(
        torch.stack(
            (
                one,
                normal + roe_sound,
                *tangential,
                roe_enthalpy + normal * roe_sound,
            ),
            dim=-1,
        )
    )
    right_eigenvectors = torch.stack(right_columns, dim=-1)

    reciprocal_sound = 1.0 / roe_sound
    b1 = gamma_minus_one * reciprocal_sound.square()
    b2 = roe_q * b1
    normal_over_sound = normal * reciprocal_sound
    b1_normal = b1 * normal
    half_b1 = 0.5 * b1
    tangential_b1 = [b1 * component for component in tangential]

    left_rows = [
        torch.stack(
            (
                0.5 * (b2 + normal_over_sound),
                -0.5 * (b1_normal + reciprocal_sound),
                *[-0.5 * value for value in tangential_b1],
                half_b1,
            ),
            dim=-1,
        )
    ]
    for tangent_index, tangent_velocity in enumerate(tangential):
        momentum_entries = [zero] * ndim
        momentum_entries[tangent_index + 1] = one
        left_rows.append(
            torch.stack((-tangent_velocity, *momentum_entries, zero), dim=-1)
        )
    left_rows.append(
        torch.stack((one - b2, b1_normal, *tangential_b1, -b1), dim=-1)
    )
    left_rows.append(
        torch.stack(
            (
                0.5 * (b2 - normal_over_sound),
                -0.5 * (b1_normal - reciprocal_sound),
                *[-0.5 * value for value in tangential_b1],
                half_b1,
            ),
            dim=-1,
        )
    )
    left_eigenvectors = torch.stack(left_rows, dim=-2)
    return flux, alpha, left_eigenvectors, right_eigenvectors


def _generated_line_rhs(
    line: Tensor,
    inverse_spacing: float,
    scheme: WENOJS,
) -> Tensor:
    """Apply generated WENO-JS to one family of component-ordered lines."""
    unique = line[..., :-1]
    flux, alpha, left, right = _flux_and_roe_faces(unique)
    offsets = scheme.exact_coefficients.candidate_offsets
    positive_offsets = tuple(
        sorted({offset for candidate in offsets for offset in candidate})
    )
    negative_offsets = tuple(
        sorted({1 - offset for candidate in offsets for offset in candidate})
    )

    def project(offset: int, sign: float) -> Tensor:
        state_sample = torch.roll(unique, shifts=-offset, dims=-1)
        flux_sample = torch.roll(flux, shifts=-offset, dims=-1)
        state_by_field = state_sample.movedim(-2, -1).unsqueeze(-2)
        flux_by_field = flux_sample.movedim(-2, -1).unsqueeze(-2)
        projected_state = (left * state_by_field).sum(dim=-1)
        projected_flux = (left * flux_by_field).sum(dim=-1)
        return 0.5 * (
            projected_flux + sign * alpha.unsqueeze(-2) * projected_state
        )

    positive = {offset: project(offset, 1.0) for offset in positive_offsets}
    negative = {offset: project(offset, -1.0) for offset in negative_offsets}
    positive_stencils = tuple(
        tuple(positive[offset] for offset in candidate) for candidate in offsets
    )
    negative_stencils = tuple(
        tuple(negative[1 - offset] for offset in candidate)
        for candidate in offsets
    )
    characteristic_flux = scheme.reconstruct_stencils(
        positive_stencils
    ) + scheme.reconstruct_stencils(negative_stencils)
    numerical_flux = (right * characteristic_flux.unsqueeze(-2)).sum(dim=-1)
    unique_derivative = (
        torch.roll(numerical_flux, shifts=1, dims=-2) - numerical_flux
    ) * inverse_spacing
    unique_derivative = unique_derivative.movedim(-1, -2)
    return torch.cat(
        (unique_derivative, unique_derivative[..., :1]), dim=-1
    )


def _generated_bounded_line_rhs(
    ghosted_line: Tensor,
    inverse_spacing: float,
    scheme: WENOJS,
    physical_points: int,
) -> tuple[Tensor, Tensor]:
    """Apply the shared characteristic algebra to explicitly ghosted lines.

    The returned pair is ``(physical_rhs, physical_face_fluxes)``. Component
    order is the penultimate dimension. Face fluxes include both domain faces,
    so their final dimension has ``physical_points + 1`` entries.
    """
    width = scheme.substencil_width
    expected_points = physical_points + 2 * width
    if ghosted_line.shape[-1] != expected_points:
        raise ValueError(
            f"expected {expected_points} ghosted points for "
            f"{physical_points} physical points"
        )

    flux, alpha, all_left, all_right = _flux_and_roe_faces(ghosted_line)
    face_start = width - 1
    face_stop = face_start + physical_points + 1
    left = all_left[..., face_start:face_stop, :, :]
    right = all_right[..., face_start:face_stop, :, :]
    offsets = scheme.exact_coefficients.candidate_offsets
    positive_offsets = tuple(
        sorted({offset for candidate in offsets for offset in candidate})
    )
    negative_offsets = tuple(
        sorted({1 - offset for candidate in offsets for offset in candidate})
    )

    def project(offset: int, sign: float) -> Tensor:
        sample_start = face_start + offset
        sample_stop = face_stop + offset
        state_sample = ghosted_line[..., :, sample_start:sample_stop]
        flux_sample = flux[..., :, sample_start:sample_stop]
        state_by_field = state_sample.movedim(-2, -1).unsqueeze(-2)
        flux_by_field = flux_sample.movedim(-2, -1).unsqueeze(-2)
        projected_state = (left * state_by_field).sum(dim=-1)
        projected_flux = (left * flux_by_field).sum(dim=-1)
        return 0.5 * (
            projected_flux + sign * alpha.unsqueeze(-2) * projected_state
        )

    positive = {offset: project(offset, 1.0) for offset in positive_offsets}
    negative = {offset: project(offset, -1.0) for offset in negative_offsets}
    positive_stencils = tuple(
        tuple(positive[offset] for offset in candidate) for candidate in offsets
    )
    negative_stencils = tuple(
        tuple(negative[1 - offset] for offset in candidate)
        for candidate in offsets
    )
    characteristic_flux = scheme.reconstruct_stencils(
        positive_stencils
    ) + scheme.reconstruct_stencils(negative_stencils)
    numerical_flux = (right * characteristic_flux.unsqueeze(-2)).sum(dim=-1)
    physical_rhs = (
        numerical_flux[..., :-1, :] - numerical_flux[..., 1:, :]
    ) * inverse_spacing
    return physical_rhs.movedim(-1, -2), numerical_flux.movedim(-1, -2)


def euler_weno_rhs(
    state: Tensor,
    spacing: Sequence[float],
    *,
    order: int = 5,
) -> Tensor:
    """Compute generated characteristic WENO-JS Euler RHS."""
    ndim = state.ndim - 1
    if ndim not in (2, 3):
        raise ValueError("state must be a 2-D or 3-D Euler field")
    if state.shape[0] != ndim + 2:
        raise ValueError(f"expected {ndim + 2} Euler components")
    if len(spacing) != ndim:
        raise ValueError(f"expected {ndim} grid spacings")
    if order not in QUALIFIED_EULER_WENO_ORDERS:
        raise ValueError(
            f"Euler WENO-JS order must be one of {QUALIFIED_EULER_WENO_ORDERS}"
        )
    if any(size - 1 < order for size in state.shape[1:]):
        raise ValueError(
            f"Euler WENO-JS order {order} requires at least {order} unique "
            "cells per axis"
        )

    state = synchronize_duplicate_endpoints(state)
    scheme = _EULER_WENO_SCHEMES[order]
    result = torch.zeros_like(state)
    for axis in range(ndim):
        order = _component_order(ndim, axis)
        tensor_axis = state.ndim - 1 - axis
        line = torch.movedim(state[list(order)], tensor_axis, -1)
        line = torch.movedim(line, 0, -2)
        line_result = _generated_line_rhs(line, 1.0 / spacing[axis], scheme)
        canonical_result = torch.movedim(line_result, -2, 0)
        canonical_result = torch.movedim(canonical_result, -1, tensor_axis)
        inverse_order = tuple(sorted(range(ndim + 2), key=order.__getitem__))
        result = result + canonical_result[list(inverse_order)]
    return result


def euler_weno5_rhs(state: Tensor, spacing: Sequence[float]) -> Tensor:
    """Compute generated characteristic JS-WENO-5 Euler RHS."""
    return euler_weno_rhs(state, spacing, order=5)


def euler_cfl_timestep(
    state: Tensor, spacing: Sequence[float], cfl: float
) -> Tensor:
    """Return Shu's sum-of-directional-speeds CFL timestep on-device."""
    ndim = state.ndim - 1
    density = state[0]
    velocity = state[1 : ndim + 1] / density.unsqueeze(0)
    energy = state[-1]
    pressure = (EULER_GAMMA - 1.0) * (
        energy - 0.5 * density * (velocity * velocity).sum(dim=0)
    )
    sound_speed = torch.sqrt(EULER_GAMMA * pressure / density)
    local_speed = torch.zeros_like(density)
    for axis in range(ndim):
        local_speed = local_speed + (
            torch.abs(velocity[axis]) + sound_speed
        ) / spacing[axis]
    interior = (slice(1, None),) * ndim
    return cfl / torch.amax(local_speed[interior])


def euler_ssp_rk3_step(
    state: Tensor,
    spacing: Sequence[float],
    dt: Tensor,
    *,
    order: int = 5,
) -> Tensor:
    """Advance one full three-stage SSP-RK3 step."""
    state = synchronize_duplicate_endpoints(state)
    rhs0 = euler_weno_rhs(state, spacing, order=order)
    stage1 = synchronize_duplicate_endpoints(state + dt * rhs0)
    rhs1 = euler_weno_rhs(stage1, spacing, order=order)
    stage2 = synchronize_duplicate_endpoints(
        0.75 * state + 0.25 * (stage1 + dt * rhs1)
    )
    rhs2 = euler_weno_rhs(stage2, spacing, order=order)
    return synchronize_duplicate_endpoints(
        (state + 2.0 * (stage2 + dt * rhs2)) / 3.0
    )


def periodic_vortex(
    intervals: Sequence[int], *, device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, tuple[float, ...]]:
    """Create the preserved periodic isentropic vortex test problem."""
    ndim = len(intervals)
    if ndim not in (2, 3):
        raise ValueError("intervals must contain nx, ny, and optionally nz")
    if any(number < 4 for number in intervals):
        raise ValueError("each axis needs at least four intervals")
    coordinates = [
        torch.arange(number + 1, device=device, dtype=dtype) * (10.0 / number)
        for number in intervals
    ]
    mesh = torch.meshgrid(*reversed(coordinates), indexing="ij")
    x = mesh[-1]
    y = mesh[-2]
    one = torch.tensor(1.0, device=device, dtype=dtype)
    pi = 4.0 * torch.atan(one)
    coefficient = 5.0 / (2.0 * pi * torch.exp(-0.5 * one))
    radius_squared = (x - 5.0).square() + (y - 5.0).square()
    exponential = torch.exp(-0.5 * radius_squared)
    x_velocity = -coefficient * exponential * (y - 5.0)
    y_velocity = coefficient * exponential * (x - 5.0)
    temperature = 1.0 - 0.5 * coefficient.square() * exponential.square() * (
        (EULER_GAMMA - 1.0) / EULER_GAMMA
    )
    pressure = temperature ** (EULER_GAMMA / (EULER_GAMMA - 1.0))
    density = pressure / temperature
    velocities = [x_velocity, y_velocity]
    if ndim == 3:
        velocities.append(torch.zeros_like(x_velocity))
    kinetic = sum(component.square() for component in velocities)
    energy = pressure / (EULER_GAMMA - 1.0) + 0.5 * density * kinetic
    state = torch.stack(
        (density, *[density * component for component in velocities], energy)
    )
    return state, tuple(10.0 / number for number in intervals)


def state_bytes(intervals: Sequence[int], *, dtype: torch.dtype) -> int:
    """Return bytes in one duplicated-endpoint Euler state tensor."""
    equations = len(intervals) + 2
    elements = equations * math.prod(number + 1 for number in intervals)
    return elements * torch.empty((), dtype=dtype).element_size()
