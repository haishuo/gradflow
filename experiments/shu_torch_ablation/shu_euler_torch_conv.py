"""Convolutional feature-bank representation of the matched Shu WENO-5 step.

This is an experimental representation, not a different numerical method.
Short grouped convolutions emit fixed linear stencil features; ordinary
pointwise tensor operations evaluate the nonlinear WENO weights in parallel.
The Roe projection remains explicit because its matrices vary by interface.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from shu_euler_torch import (
    _component_order,
    _flux_and_roe_matrices,
    _periodic_ghosts_with_duplicate_endpoint,
    synchronize_duplicate_endpoints,
)


def _grouped_stencil(value: Tensor, weight: Tensor) -> Tensor:
    """Apply one fixed 1-D stencil independently to every field."""
    channels = value.shape[-2]
    if weight.shape[0] != channels:
        raise ValueError("stencil weight does not match the field count")
    leading = value.shape[:-2]
    flattened = value.reshape(-1, channels, value.shape[-1])
    result = F.conv1d(flattened, weight, groups=channels)
    return result.reshape(*leading, channels, result.shape[-1])


def _nonlinear_correction_feature_bank(h: Tensor, weight: Tensor) -> Tensor:
    """Evaluate both WENO corrections from convolution-emitted features.

    ``h`` has shape ``(..., two_splits, four_samples, interfaces)``.  Every
    interface/split record becomes one four-sample convolution problem.  The
    six output channels are the three adjacent differences and the three
    additional linear forms used by the Jiang--Shu indicators.
    """
    samples = h.movedim(-1, -2)
    flattened = samples.reshape(-1, 1, 4)
    features = F.conv1d(flattened, weight).squeeze(-1)
    features = features.reshape(*samples.shape[:-1], 6).movedim(-1, -2)

    t1, t2, t3, s1, s2, s3 = features.unbind(dim=-2)
    indicator1 = 13.0 * t1.square() + 3.0 * s1.square()
    indicator2 = 13.0 * t2.square() + 3.0 * s2.square()
    indicator3 = 13.0 * t3.square() + 3.0 * s3.square()

    denominator1 = (1.0e-6 + indicator1).square()
    denominator2 = (1.0e-6 + indicator2).square()
    denominator3 = (1.0e-6 + indicator3).square()
    weight1 = denominator2 * denominator3
    weight2 = 6.0 * denominator1 * denominator3
    weight3 = 3.0 * denominator1 * denominator2
    reciprocal_sum = 1.0 / (weight1 + weight2 + weight3)
    weight1 = weight1 * reciprocal_sum
    weight3 = weight3 * reciprocal_sum

    correction = (
        weight1 * (t2 - t1)
        + (0.5 * weight3 - 0.25) * (t3 - t2)
    ) / 3.0
    return correction.sum(dim=-2)


def _line_rhs_conv(
    line: Tensor,
    inverse_spacing: float,
    difference_weight: Tensor,
    central_weight: Tensor,
    feature_weight: Tensor,
) -> Tensor:
    """Apply the exact characteristic reconstruction using feature banks."""
    ghosted = _periodic_ghosts_with_duplicate_endpoint(line)
    flux, alpha, left, right = _flux_and_roe_matrices(ghosted)

    differences = _grouped_stencil(
        torch.cat((flux, ghosted), dim=-2), difference_weight
    )
    equations = ghosted.shape[-2]
    flux_difference = differences[..., :equations, :]
    state_difference = differences[..., equations:, :]
    split_positive = 0.5 * (
        flux_difference.unsqueeze(-3)
        + alpha[..., :, None, None] * state_difference.unsqueeze(-3)
    )
    split_negative = split_positive - flux_difference.unsqueeze(-3)

    positive_candidates = torch.stack(
        (
            split_positive[..., 0:-4],
            split_positive[..., 1:-3],
            split_positive[..., 2:-2],
            split_positive[..., 3:-1],
        ),
        dim=-2,
    )
    negative_candidates = torch.stack(
        (
            split_negative[..., 4:],
            split_negative[..., 3:-1],
            split_negative[..., 2:-2],
            split_negative[..., 1:-3],
        ),
        dim=-2,
    )

    left_by_field = left.movedim(-3, -1).unsqueeze(-2)
    projected_positive = (left_by_field * positive_candidates).sum(dim=-3)
    projected_negative = (left_by_field * negative_candidates).sum(dim=-3)
    projected = torch.stack((projected_positive, projected_negative), dim=-3)
    characteristic_flux = _nonlinear_correction_feature_bank(
        projected, feature_weight
    )

    characteristic_flux = characteristic_flux.movedim(-1, -2).unsqueeze(-2)
    nonlinear_flux = (right * characteristic_flux).sum(dim=-1)
    central_flux = _grouped_stencil(flux, central_weight)[..., 1:-1]
    central_flux = central_flux.movedim(-2, -1)
    numerical_flux = nonlinear_flux + central_flux
    derivative = (
        numerical_flux[..., :-1, :] - numerical_flux[..., 1:, :]
    ) * inverse_spacing
    return derivative.movedim(-1, -2)


class ConvFeatureWenoStep(torch.nn.Module):
    """One exact 2-D or 3-D Shu SSP-RK3 step using convolutional features."""

    def __init__(
        self,
        dimension: int,
        spacing: Sequence[float],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if dimension not in (2, 3):
            raise ValueError("dimension must be two or three")
        if len(spacing) != dimension:
            raise ValueError("spacing does not match dimension")
        self.dimension = dimension
        self.spacing = tuple(float(value) for value in spacing)
        equations = dimension + 2

        difference = torch.tensor([-1.0, 1.0], dtype=dtype)
        difference = difference.reshape(1, 1, 2).repeat(2 * equations, 1, 1)
        central = torch.tensor([-1.0, 7.0, 7.0, -1.0], dtype=dtype) / 12.0
        central = central.reshape(1, 1, 4).repeat(equations, 1, 1)
        features = torch.tensor(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
                [1.0, -3.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 3.0, -1.0],
            ],
            dtype=dtype,
        ).reshape(6, 1, 4)
        self.register_buffer("difference_weight", difference)
        self.register_buffer("central_weight", central)
        self.register_buffer("feature_weight", features)

    def _rhs(self, state: Tensor) -> Tensor:
        state = synchronize_duplicate_endpoints(state)
        result = torch.zeros_like(state)
        for axis in range(self.dimension):
            order = _component_order(self.dimension, axis)
            tensor_axis = state.ndim - 1 - axis
            line = torch.movedim(state[list(order)], tensor_axis, -1)
            line = torch.movedim(line, 0, -2)
            line_result = _line_rhs_conv(
                line,
                1.0 / self.spacing[axis],
                self.difference_weight,
                self.central_weight,
                self.feature_weight,
            )
            canonical = torch.movedim(line_result, -2, 0)
            canonical = torch.movedim(canonical, -1, tensor_axis)
            inverse_order = tuple(
                sorted(range(self.dimension + 2), key=order.__getitem__)
            )
            result = result + canonical[list(inverse_order)]
        return result

    def forward(self, state: Tensor, dt: Tensor) -> Tensor:
        state = synchronize_duplicate_endpoints(state)
        rhs0 = self._rhs(state)
        stage1 = synchronize_duplicate_endpoints(state + dt * rhs0)
        rhs1 = self._rhs(stage1)
        stage2 = synchronize_duplicate_endpoints(
            0.75 * state + 0.25 * (stage1 + dt * rhs1)
        )
        rhs2 = self._rhs(stage2)
        return synchronize_duplicate_endpoints(
            (state + 2.0 * (stage2 + dt * rhs2)) / 3.0
        )
