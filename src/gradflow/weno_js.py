"""Generated odd-order finite-difference WENO-JS in ordinary PyTorch."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .weno5 import DEFAULT_EPSILON
from .weno_js_coefficients import (
    WENOJSCoefficients,
    generate_weno_js_coefficients,
)

TensorFunction = Callable[[torch.Tensor], torch.Tensor]
QUALIFIED_ORDERS = (5, 7, 9, 11, 13, 15)
SMOOTHNESS_SCALE = 12.0


def _linear_combination(
    coefficients: tuple[float, ...], values: list[torch.Tensor]
) -> torch.Tensor:
    result = coefficients[0] * values[0]
    for coefficient, value in zip(coefficients[1:], values[1:]):
        result = result + coefficient * value
    return result


class WENOJS:
    """One generated scalar finite-difference Jiang--Shu WENO scheme.

    Construction performs exact rational mathematics once. Numerical calls
    contain only fixed Python loops, periodic rolls, and tensor elementwise
    operations; a fixed instance is suitable for ``torch.compile``.
    """

    def __init__(
        self,
        order: int,
        *,
        epsilon: float = DEFAULT_EPSILON,
        nonlinear_power: int = 2,
    ) -> None:
        if not isinstance(epsilon, (int, float)) or epsilon <= 0:
            raise ValueError("epsilon must be a positive Python scalar")
        if (
            isinstance(nonlinear_power, bool)
            or not isinstance(nonlinear_power, int)
            or nonlinear_power < 1
        ):
            raise ValueError("nonlinear_power must be a positive integer")
        exact = generate_weno_js_coefficients(order)
        self.order = exact.order
        self.substencil_width = exact.substencil_width
        self.epsilon = float(epsilon)
        self.nonlinear_power = nonlinear_power
        self.exact_coefficients: WENOJSCoefficients = exact
        self._candidate_offsets = exact.candidate_offsets
        self._candidate_coefficients = tuple(
            tuple(float(value) for value in candidate)
            for candidate in exact.candidate_coefficients
        )
        self._optimal_weights = tuple(float(value) for value in exact.optimal_weights)
        self._smoothness_factors = tuple(
            tuple(
                (
                    SMOOTHNESS_SCALE * float(weight),
                    tuple(float(value) for value in coefficients),
                )
                for weight, coefficients in factors
            )
            for factors in exact.smoothness_factors
        )

    @staticmethod
    def _axis(values: torch.Tensor, axis: int) -> int:
        if not isinstance(values, torch.Tensor):
            raise TypeError("values must already be a torch.Tensor")
        if values.dtype not in (torch.float32, torch.float64):
            raise TypeError("WENO-JS requires float32 or float64 input")
        if values.ndim < 1:
            raise ValueError("values must have a spatial dimension")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("axis must be an integer")
        if not -values.ndim <= axis < values.ndim:
            raise ValueError("axis is outside the input rank")
        return axis % values.ndim

    @staticmethod
    def _shift(values: torch.Tensor, offset: int, axis: int) -> torch.Tensor:
        return torch.roll(values, shifts=-offset, dims=axis)

    def reconstruct(
        self,
        values: torch.Tensor,
        *,
        bias: str = "left",
        axis: int = -1,
    ) -> torch.Tensor:
        """Reconstruct the periodic interface value at ``i+1/2``.

        ``values`` are finite-difference flux samples on unique periodic nodes.
        ``bias="left"`` reconstructs from the upwind-left family;
        ``bias="right"`` uses its exact reflection about the interface.
        """
        normalized_axis = self._axis(values, axis)
        if values.shape[normalized_axis] < self.order:
            raise ValueError(
                f"WENO-JS order {self.order} requires at least {self.order} points"
            )
        if bias not in {"left", "right"}:
            raise ValueError("bias must be 'left' or 'right'")

        candidates = []
        indicators = []
        for offsets, coefficients, factors in zip(
            self._candidate_offsets,
            self._candidate_coefficients,
            self._smoothness_factors,
        ):
            if bias == "left":
                stencil = [
                    self._shift(values, offset, normalized_axis) for offset in offsets
                ]
            else:
                stencil = [
                    self._shift(values, 1 - offset, normalized_axis)
                    for offset in offsets
                ]
            candidates.append(_linear_combination(coefficients, stencil))
            indicator = None
            for factor_weight, factor_coefficients in factors:
                factor = _linear_combination(factor_coefficients, stencil)
                term = factor_weight * factor.square()
                indicator = term if indicator is None else indicator + term
            assert indicator is not None
            indicators.append(indicator)

        candidate_stack = torch.stack(candidates, dim=0)
        denominator = torch.stack(indicators, dim=0) + self.epsilon
        # Scaling by the smallest denominator preserves the normalized JS
        # weights while preventing float32 overflow on exactly constant data.
        scale = torch.amin(denominator, dim=0, keepdim=True)
        inverse_ratio = scale / denominator
        nonlinear = torch.stack(
            [
                weight * inverse_ratio[candidate].pow(self.nonlinear_power)
                for candidate, weight in enumerate(self._optimal_weights)
            ],
            dim=0,
        )
        weights = nonlinear / torch.sum(nonlinear, dim=0, keepdim=True)
        return torch.sum(weights * candidate_stack, dim=0)

    def rhs(
        self,
        u: torch.Tensor,
        dx: float | torch.Tensor,
        flux: TensorFunction,
        flux_derivative: TensorFunction | None = None,
        *,
        alpha: float | torch.Tensor | None = None,
        axis: int = -1,
    ) -> torch.Tensor:
        """Compute a conservative periodic scalar RHS with global LF splitting."""
        normalized_axis = self._axis(u, axis)
        if u.shape[normalized_axis] < self.order:
            raise ValueError(
                f"WENO-JS order {self.order} requires at least {self.order} points"
            )
        physical_flux = flux(u)
        if physical_flux.shape != u.shape:
            raise ValueError("flux(u) must have the same shape as u")
        if alpha is None:
            if flux_derivative is None:
                raise ValueError("provide flux_derivative when alpha is not explicit")
            alpha_value: float | torch.Tensor = torch.amax(
                torch.abs(flux_derivative(u))
            )
        else:
            alpha_value = alpha
        positive = 0.5 * (physical_flux + alpha_value * u)
        negative = 0.5 * (physical_flux - alpha_value * u)
        interface_flux = self.reconstruct(
            positive, bias="left", axis=normalized_axis
        ) + self.reconstruct(negative, bias="right", axis=normalized_axis)
        previous_interface = self._shift(interface_flux, -1, normalized_axis)
        return (previous_interface - interface_flux) / dx
