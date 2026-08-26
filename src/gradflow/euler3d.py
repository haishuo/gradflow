"""Characteristic finite-difference JS-WENO-5 for compressible Euler.

This is the package form of the matched direct-PyTorch formulation used by the
Shu Euler bakeoff. It deliberately preserves the ancestral duplicated-periodic
grid, Roe characteristic reconstruction, line-wise global LF policy, and
SSP-RK3 algebra. It contains no convolution or custom native operation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor


EULER_GAMMA = 1.4
EULER_WENO_EPSILON = 1.0e-6
EULER_LF_ENLARGEMENT = 1.1


def _component_order(ndim: int, axis: int) -> tuple[int, ...]:
    momenta = list(range(1, ndim + 1))
    normal = momenta.pop(axis)
    return (0, normal, *momenta, ndim + 1)


def _periodic_ghosts_with_duplicate_endpoint(line: Tensor) -> Tensor:
    if line.shape[-1] < 5:
        raise ValueError("each axis needs at least four intervals")
    return torch.cat((line[..., -4:-1], line, line[..., 1:4]), dim=-1)


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


def _flux_and_roe_matrices(line: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

    sqrt_density = torch.sqrt(density)
    left_weight = sqrt_density[..., 2:-3]
    right_weight = sqrt_density[..., 3:-2]
    fraction = left_weight / (left_weight + right_weight)
    roe_velocity = (
        fraction.unsqueeze(-2) * velocity[..., 2:-3]
        + (1.0 - fraction).unsqueeze(-2) * velocity[..., 3:-2]
    )
    roe_enthalpy = (
        fraction * enthalpy[..., 2:-3]
        + (1.0 - fraction) * enthalpy[..., 3:-2]
    )
    roe_q = 0.5 * (roe_velocity * roe_velocity).sum(dim=-2)
    roe_sound = torch.sqrt(gamma_minus_one * (roe_enthalpy - roe_q))

    normal = roe_velocity[..., 0, :]
    tangential = [roe_velocity[..., index, :] for index in range(1, ndim)]
    one = torch.ones_like(normal)
    zero = torch.zeros_like(normal)

    right_columns = [
        torch.stack(
            (one, normal - roe_sound, *tangential,
             roe_enthalpy - normal * roe_sound),
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
            (one, normal + roe_sound, *tangential,
             roe_enthalpy + normal * roe_sound),
            dim=-1,
        )
    )
    right_eigenvectors = torch.stack(right_columns, dim=-1)

    reciprocal_sound = 1.0 / roe_sound
    b1 = gamma_minus_one * reciprocal_sound * reciprocal_sound
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


def _nonlinear_flux_correction(h: Tensor) -> Tensor:
    t1 = h[..., 0, :] - h[..., 1, :]
    t2 = h[..., 1, :] - h[..., 2, :]
    t3 = h[..., 2, :] - h[..., 3, :]

    indicator1 = 13.0 * t1.square() + 3.0 * (
        h[..., 0, :] - 3.0 * h[..., 1, :]
    ).square()
    indicator2 = 13.0 * t2.square() + 3.0 * (
        h[..., 1, :] + h[..., 2, :]
    ).square()
    indicator3 = 13.0 * t3.square() + 3.0 * (
        3.0 * h[..., 2, :] - h[..., 3, :]
    ).square()

    # This is algebraically identical to Shu's product form
    # q2*q3 : 6*q1*q3 : 3*q1*q2, after division by q1*q2*q3.
    # The inverse form avoids a reciprocal of ~1e-24 in smooth float32 regions,
    # whose backward pass otherwise overflows before normalization cancels it.
    weight1 = 1.0 / (EULER_WENO_EPSILON + indicator1).square()
    weight2 = 6.0 / (EULER_WENO_EPSILON + indicator2).square()
    weight3 = 3.0 / (EULER_WENO_EPSILON + indicator3).square()
    reciprocal_sum = 1.0 / (weight1 + weight2 + weight3)
    weight1 = weight1 * reciprocal_sum
    weight3 = weight3 * reciprocal_sum
    return (
        weight1 * (t2 - t1)
        + (0.5 * weight3 - 0.25) * (t3 - t2)
    ) / 3.0


def _line_rhs(line: Tensor, inverse_spacing: float) -> Tensor:
    ghosted = _periodic_ghosts_with_duplicate_endpoint(line)
    flux, alpha, left, right = _flux_and_roe_matrices(ghosted)

    flux_difference = flux[..., 1:] - flux[..., :-1]
    state_difference = ghosted[..., 1:] - ghosted[..., :-1]
    split_positive = 0.5 * (
        flux_difference.unsqueeze(-3)
        + alpha[..., :, None, None] * state_difference.unsqueeze(-3)
    )
    split_negative = split_positive - flux_difference.unsqueeze(-3)
    positive_candidates = torch.stack(
        (
            split_positive[..., 0:-4], split_positive[..., 1:-3],
            split_positive[..., 2:-2], split_positive[..., 3:-1],
        ),
        dim=-2,
    )
    negative_candidates = torch.stack(
        (
            split_negative[..., 4:], split_negative[..., 3:-1],
            split_negative[..., 2:-2], split_negative[..., 1:-3],
        ),
        dim=-2,
    )
    left_by_field = left.movedim(-3, -1).unsqueeze(-2)
    projected_positive = (left_by_field * positive_candidates).sum(dim=-3)
    projected_negative = (left_by_field * negative_candidates).sum(dim=-3)
    characteristic_flux = (
        _nonlinear_flux_correction(projected_positive)
        + _nonlinear_flux_correction(projected_negative)
    )
    characteristic_flux = characteristic_flux.movedim(-1, -2).unsqueeze(-2)
    nonlinear_flux = (right * characteristic_flux).sum(dim=-1)
    central_flux = (
        -flux[..., 1:-4]
        + 7.0 * (flux[..., 2:-3] + flux[..., 3:-2])
        - flux[..., 4:-1]
    ).movedim(-2, -1) / 12.0
    numerical_flux = nonlinear_flux + central_flux
    derivative = (
        numerical_flux[..., :-1, :] - numerical_flux[..., 1:, :]
    ) * inverse_spacing
    return derivative.movedim(-1, -2)


def euler_weno5_rhs(state: Tensor, spacing: Sequence[float]) -> Tensor:
    """Compute characteristic JS-WENO-5 Euler RHS on a duplicated grid."""
    ndim = state.ndim - 1
    if ndim not in (2, 3):
        raise ValueError("state must be a 2-D or 3-D Euler field")
    if state.shape[0] != ndim + 2:
        raise ValueError(f"expected {ndim + 2} Euler components")
    if len(spacing) != ndim:
        raise ValueError(f"expected {ndim} grid spacings")

    state = synchronize_duplicate_endpoints(state)
    result = torch.zeros_like(state)
    for axis in range(ndim):
        order = _component_order(ndim, axis)
        tensor_axis = state.ndim - 1 - axis
        line = torch.movedim(state[list(order)], tensor_axis, -1)
        line = torch.movedim(line, 0, -2)
        line_result = _line_rhs(line, 1.0 / spacing[axis])
        canonical_result = torch.movedim(line_result, -2, 0)
        canonical_result = torch.movedim(canonical_result, -1, tensor_axis)
        inverse_order = tuple(sorted(range(ndim + 2), key=order.__getitem__))
        result = result + canonical_result[list(inverse_order)]
    return result


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
    state: Tensor, spacing: Sequence[float], dt: Tensor
) -> Tensor:
    """Advance one full three-stage SSP-RK3 step."""
    state = synchronize_duplicate_endpoints(state)
    rhs0 = euler_weno5_rhs(state, spacing)
    stage1 = synchronize_duplicate_endpoints(state + dt * rhs0)
    rhs1 = euler_weno5_rhs(stage1, spacing)
    stage2 = synchronize_duplicate_endpoints(
        0.75 * state + 0.25 * (stage1 + dt * rhs1)
    )
    rhs2 = euler_weno5_rhs(stage2, spacing)
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
